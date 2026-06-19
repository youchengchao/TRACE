"""The TRACE localization head and its building blocks (Token Reference Attention).

Core idea: a manipulated region is the part of an image that cannot be explained by the rest of
the same image. For each image-patch token, "reference attention" builds a reference vector from
the other tokens and measures how much the token DEVIATES from that reference; the deviation is
the localization signal. The mask is produced only from this deviation pathway (not from the raw
features), which structurally ties "fake" to "not explainable by the rest of the image".

Pipeline:
  backbone patch-token grids
    -> ReferenceAttention at one or more scales  (reference vector + deviation per token)
    -> a convolutional decoder upsamples the per-token signal to a per-pixel forgery-mask logit
    -> a separate classification head turns the image-level token into a real/fake logit

Contents of this file:
  ReferenceAttention            one scale of the reference-attention + deviation mechanism
  ResDecoder / GatedResDecoder  decoder variants (TRACE uses GatedResDecoder; 'res' is for ablation)
  TRACEHead                     the full head: reference attention -> decoder -> mask + class logit
  LoRALinear / inject_lora      Low-Rank Adaptation, used to fine-tune the frozen backbone cheaply
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _grid_to_tokens(x: torch.Tensor) -> torch.Tensor:
    # (B, C, H, W) -> (B, H*W, C)
    b, c, h, w = x.shape
    return x.permute(0, 2, 3, 1).reshape(b, h * w, c)


def _rel_offsets(h: int, w: int, device) -> torch.Tensor:
    """Normalised relative offsets (dr, dc) for every (query, key) pair -> (N, N, 2)."""
    ys, xs = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    coords = torch.stack([ys.reshape(-1), xs.reshape(-1)], dim=-1).float()  # (N, 2)
    diff = coords[:, None, :] - coords[None, :, :]  # (N, N, 2)
    diff[..., 0] /= max(h - 1, 1)
    diff[..., 1] /= max(w - 1, 1)
    return diff


class ReferenceAttention(nn.Module):
    """One scale of differentiable same-image clean-reference attention + residual bottleneck."""

    def __init__(self, dim: int, attn_dim: int = 128, res_dim: int = 128, temp_init: float = 1.0):
        super().__init__()
        self.q = nn.Linear(dim, attn_dim)
        self.k = nn.Linear(dim, attn_dim)
        self.v = nn.Linear(dim, attn_dim)
        self.g_src = nn.Sequential(nn.Linear(dim, attn_dim), nn.GELU(), nn.Linear(attn_dim, 1))
        self.pos_bias = nn.Sequential(nn.Linear(2, 32), nn.GELU(), nn.Linear(32, 1))
        self.log_temp = nn.Parameter(torch.tensor(float(torch.log(torch.tensor(temp_init)))))
        self.residual = nn.Sequential(nn.Linear(3 * attn_dim, res_dim), nn.GELU(), nn.Linear(res_dim, res_dim))
        self.authority = nn.Sequential(nn.Linear(3, 32), nn.GELU(), nn.Linear(32, 1))
        self.fake_head = nn.Linear(res_dim, 1)
        self.res_dim = res_dim
        self.ablate_ref = False   # null test: if True, ref:=0 -> residual=f(v) only (no clean-reference graph)
        # optional robust-reference variant (unused by default; ref_mode="attn"). ref_mode in {"attn","robust","ablate"}.
        # "robust" replaces similarity-weighted ref with deviation-gated reference:
        # sources that deviate from the real consensus (fake) get Tukey weight -> 0.
        self.ref_mode = "attn"
        self.robust_iters = 2
        # cutoff = softplus(log_c) * median(deviation); init softplus(2.0)=2.13 -> only the
        # tail (~fake) beyond ~2x median deviation is excluded, real bulk retained.
        self.log_c = nn.Parameter(torch.full((1,), 2.0))
        # res_input: "full" = MLP([v, ref, v-ref]) (raw-v shortcut, graph bypassable);
        # "dev" = MLP([v-ref]) ONLY -> decoder sees only the clean-reference deviation, so the
        # graph is load-bearing by construction (no raw-v path to read out localization from).
        self.res_input = "full"
        self.residual_dev = nn.Sequential(nn.Linear(attn_dim, res_dim), nn.GELU(),
                                          nn.Linear(res_dim, res_dim))

    def forward(self, type_grid: torch.Tensor, value_grid: torch.Tensor):
        b, c, h, w = type_grid.shape
        t = _grid_to_tokens(type_grid)   # (B, N, C)
        v_in = _grid_to_tokens(value_grid)
        q = self.q(t); k = self.k(t); v = self.v(v_in)            # (B, N, A)
        scale = q.shape[-1] ** -0.5
        scores = torch.matmul(q, k.transpose(1, 2)) * scale       # (B, N, N)
        pb = self.pos_bias(_rel_offsets(h, w, t.device)).squeeze(-1)  # (N, N)
        base = scores + pb.unsqueeze(0)                            # contextual candidate structure
        temp = self.log_temp.exp().clamp_min(1e-2)
        log_gsrc = F.logsigmoid(self.g_src(t).squeeze(-1))        # (B, N) clean-source gate (log)
        self._gsrc_log = log_gsrc.detach()                         # (B,N) for eval gsrc_gap

        if self.ref_mode == "robust":
            # robust-reference variant: weight source tokens by how much they DEVIATE
            # from the real consensus (not by similarity) -> fake self-excludes.
            a = (base / temp).softmax(dim=-1)
            ref = torch.matmul(a, v)
            for _ in range(self.robust_iters):
                r = torch.linalg.vector_norm(v - ref, dim=-1)      # (B,N) deviation of each token
                med = r.median(dim=1, keepdim=True).values.detach()  # robust scale (outlier-insensitive)
                cc = F.softplus(self.log_c) * (med + 1e-6)
                u = (r / (cc + 1e-6)).clamp(max=1.0)               # >= cutoff -> 1
                wreal = (1.0 - u * u) ** 2                          # Tukey biweight, 0 beyond cutoff
                logw = torch.log(wreal + 1e-6)                     # (B,N) source reweight (fake -> -inf)
                a = (base / temp + logw.unsqueeze(1)).softmax(dim=-1)
                ref = torch.matmul(a, v)
            self._real_w = wreal.detach()                          # per-token real-ness (1-w ~ fakeness)
        else:
            scores2 = (base + log_gsrc.unsqueeze(1)) / temp        # down-weight unreliable sources
            a = scores2.softmax(dim=-1)                            # (B, N, N)
            ref = torch.matmul(a, v)                               # (B, N, A) similarity-weighted ref
        self._attn_a = a                                           # kept with gradient
        if self.ablate_ref:                                        # NULL TEST: kill the graph
            ref = torch.zeros_like(ref)                            # residual becomes f(v) only
        delta    = v - ref
        self._delta = delta.detach()                               # (B,N,A) for eval delta_gap
        if self.res_input == "dev":
            residual = self.residual_dev(delta)                    # decoder sees ONLY v-ref (graph load-bearing)
        else:
            residual = self.residual(torch.cat([v, ref, delta], dim=-1))  # (B, N, R)
        # soft authority from attention statistics
        a_max   = a.max(dim=-1).values
        a_top   = a.topk(min(5, a.shape[-1]), dim=-1).values.sum(dim=-1)
        neg_ent = (a * (a + 1e-9).log()).sum(dim=-1)
        authority = torch.sigmoid(self.authority(torch.stack([a_max, a_top, neg_ent], dim=-1)).squeeze(-1))
        raw_fe = self.fake_head(residual).squeeze(-1)              # (B, N) raw logit pre-authority
        self._raw_fe = raw_fe.reshape(b, 1, h, w)                 # pre-gate forgery logit (used for mask supervision)
        fake_evidence = raw_fe * authority                         # (B, N) <<< bottleneck
        fe  = fake_evidence.reshape(b, 1, h, w)
        res = residual.transpose(1, 2).reshape(b, self.res_dim, h, w)
        return fe, res, residual, authority


class _ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.GroupNorm(8, ch), nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.GroupNorm(8, ch),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))


class ResDecoder(nn.Module):
    """Progressive residual decoder 32->64->out_size with residual blocks at each stage.
    forward(evidences, residuals) -> mask logit."""

    def __init__(self, res_dim: int, n_scales: int, hidden: int = 64, out_size: int = 112):
        super().__init__()
        self.out_size = out_size
        self.proj   = nn.ModuleList([nn.Conv2d(res_dim + 1, hidden, 1) for _ in range(n_scales)])
        self.res32  = nn.Sequential(_ResBlock(hidden), _ResBlock(hidden))
        self.res64  = _ResBlock(hidden)
        self.resout = _ResBlock(hidden)
        self.head   = nn.Conv2d(hidden, 1, 1)
        self._acc: torch.Tensor | None = None   # side-channel for v6b FCOS

    def forward(self, evidences, residuals):
        acc = None
        for proj, fe, res in zip(self.proj, evidences, residuals):
            x = proj(torch.cat([res, fe], dim=1))
            x = F.interpolate(x, (32, 32), mode="bilinear", align_corners=False)
            acc = x if acc is None else acc + x
        acc = self.res32(acc)
        acc = F.interpolate(acc, (64, 64), mode="bilinear", align_corners=False)
        acc = self.res64(acc)
        acc = F.interpolate(acc, (self.out_size, self.out_size), mode="bilinear", align_corners=False)
        acc = self.resout(acc)
        self._acc = acc          # [B, hidden, out_size, out_size] for v6b
        return self.head(acc)    # [B, 1, out_size, out_size]


class GatedResDecoder(nn.Module):
    """Gated decoder fused with progressive upsampling (32 -> 64 -> 112). This is the decoder
    TRACE uses.

    At each resolution, the residual feature is multiplied by a structural gate
    sigmoid(raw_forgery_logit + bias) BEFORE being projected, so the forgery evidence is a
    necessary gate on the localization signal (the decoder cannot bypass it via a raw-feature
    shortcut). The gated features are then upsampled progressively to the output resolution.
    """

    def __init__(self, res_dim: int, n_scales: int, hidden: int = 64, out_size: int = 112,
                 dynamic_upsample: bool = False):
        super().__init__()
        self.out_size = out_size
        # dynamic_upsample=True: progressively upsample from the INPUT grid (in -> 2*in -> out),
        # no hardcoded 32. Lets v2 (32 grid) and v3 (28 grid) each start from their native grid
        # with no spurious resample. Default False keeps the legacy 32->64->out behavior.
        self.dynamic_upsample = dynamic_upsample
        self.gate_bias = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(n_scales)])
        self.proj   = nn.ModuleList([nn.Conv2d(res_dim + 1, hidden, 1) for _ in range(n_scales)])
        self.res32  = nn.Sequential(_ResBlock(hidden), _ResBlock(hidden))
        self.res64  = _ResBlock(hidden)
        self.resout = _ResBlock(hidden)
        self.head   = nn.Conv2d(hidden, 1, 1)
        self._acc: torch.Tensor | None = None   # side-channel for v6b FCOS

    def forward(self, raw_fes, residuals):
        acc = None
        in_hw = residuals[0].shape[-2:]                 # native input grid (e.g. 32 for v2, 28 for v3)
        for bias, proj, raw_fe, res in zip(self.gate_bias, self.proj, raw_fes, residuals):
            gate      = torch.sigmoid(raw_fe + bias)   # structural gate
            gated_res = res * gate                      # residual gated by fake-evidence
            x = proj(torch.cat([gated_res, raw_fe], dim=1))
            tgt = in_hw if self.dynamic_upsample else (32, 32)
            x = F.interpolate(x, tgt, mode="bilinear", align_corners=False)
            acc = x if acc is None else acc + x
        acc = self.res32(acc)
        mid = (in_hw[0] * 2, in_hw[1] * 2) if self.dynamic_upsample else (64, 64)
        acc = F.interpolate(acc, mid, mode="bilinear", align_corners=False)
        acc = self.res64(acc)
        acc = F.interpolate(acc, (self.out_size, self.out_size), mode="bilinear", align_corners=False)
        acc = self.resout(acc)
        self._acc = acc
        return self.head(acc)


class TRACEHead(nn.Module):
    """The TRACE localization head: turns backbone features into a per-pixel forgery-probability
    map and an image-level real/fake score, using Token Reference Attention (ReferenceAttention)
    followed by a convolutional decoder.

    decoder_type selects the decoder variant:
      'gatedres' — GatedResDecoder (gated + progressive upsampling; the default used by TRACE)
      'res'      — ResDecoder (progressive upsampling 32->64->112; used for the decoder ablation)
    """

    def __init__(self, dim: int, scales=(16, 32), attn_dim: int = 128, res_dim: int = 128,
                 out_size: int = 64, cls_dim: int | None = None, decoder_type: str = "gatedres"):
        super().__init__()
        self.scales = tuple(scales)
        self.attn = nn.ModuleList([ReferenceAttention(dim, attn_dim, res_dim) for _ in self.scales])
        if decoder_type == "gatedres":
            self.decoder = GatedResDecoder(res_dim, len(self.scales), out_size=out_size)
        elif decoder_type == "res":
            self.decoder = ResDecoder(res_dim, len(self.scales), out_size=out_size)
        else:
            raise ValueError(f"unknown decoder_type {decoder_type!r} (use 'gatedres' or 'res')")
        self.decoder_type = decoder_type
        self.cls_head = nn.Sequential(nn.LayerNorm(cls_dim or dim), nn.Linear(cls_dim or dim, 2))

    def forward(self, type_grid: torch.Tensor, value_grid: torch.Tensor, cls_token: torch.Tensor):
        evidences, residuals, raw_fes, res_tokens_per_scale = [], [], [], []
        for s, attn in zip(self.scales, self.attn):
            tg = F.interpolate(type_grid, size=(s, s), mode="bilinear", align_corners=False)
            vg = F.interpolate(value_grid, size=(s, s), mode="bilinear", align_corners=False)
            fe, res, res_tok, _auth = attn(tg, vg)
            evidences.append(fe)
            residuals.append(res)
            raw_fes.append(attn._raw_fe)           # (B,1,s,s) pre-authority logit
            res_tokens_per_scale.append(res_tok)

        if self.decoder_type == "gatedres":
            mask_logit = self.decoder(raw_fes, residuals)
        else:  # "res"
            fe_in = raw_fes if getattr(self, "res_fe_channel", "evidence") == "raw" else evidences
            mask_logit = self.decoder(fe_in, residuals)

        image_logit = self.cls_head(cls_token)     # (B,2)
        return {"mask_logit": mask_logit, "image_logit": image_logit,
                "evidences": evidences, "raw_fes": raw_fes,
                "res_tokens": res_tokens_per_scale}


class LoRALinear(nn.Module):
    """Low-rank adapter over a frozen nn.Linear: y = base(x) + (alpha/r) * B(A(x))."""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.A = nn.Linear(base.in_features, r, bias=False)
        self.B = nn.Linear(r, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.B.weight)
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + self.scale * self.B(self.A(x))


def inject_lora(backbone, rank: int, alpha: int = 16, targets=("qkv", "fc1", "fc2"), block_ids=None):
    # block_ids=None -> all blocks (legacy). Pass an iterable (e.g. range(0,18)) to restrict LoRA to
    # specific depths (e.g. <=L18 so LoRA only touches blocks that feed the localization taps).
    n = 0
    allow = None if block_ids is None else set(block_ids)
    for bi, blk in enumerate(backbone.blocks):
        if allow is not None and bi not in allow:
            continue
        if "qkv" in targets:
            blk.attn.qkv = LoRALinear(blk.attn.qkv, rank, alpha); n += 1
        if "proj" in targets:
            blk.attn.proj = LoRALinear(blk.attn.proj, rank, alpha); n += 1
        if "fc1" in targets:
            blk.mlp.fc1 = LoRALinear(blk.mlp.fc1, rank, alpha); n += 1
        if "fc2" in targets:
            blk.mlp.fc2 = LoRALinear(blk.mlp.fc2, rank, alpha); n += 1
    return n
