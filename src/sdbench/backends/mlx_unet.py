"""SD 1.5 UNet forward pass in MLX (R3.4.1).

Upstream ``ml-explore/mlx-examples`` ships SD 2.1-base / SDXL / FLUX configs but
no SD 1.5. This module supplies the SD 1.5 UNet on the canonical architecture as
a thin, functional MLX implementation: the weights dict *is* the model, so every
tensor shape is pinned by the diffusers state_dict that feeds it (no shape
guesswork). Keys mirror the diffusers ``UNet2DConditionModel`` state_dict, which
makes the weight remap in ``load_weights`` mechanical.

Layout convention: MLX convolutions are NHWC, so the adapter transposes the
input latent to NHWC on entry and back to NCHW on exit; 1x1 convolutions
(proj_in/proj_out/conv_shortcut) are folded to channel-linear weights at load.
"""

import math
from dataclasses import dataclass

import mlx.core as mx


@dataclass(frozen=True)
class UNetConfig:
    in_channels: int = 4
    out_channels: int = 4
    block_out_channels: tuple = (320, 640, 1280, 1280)
    layers_per_block: int = 2
    cross_attention_dim: int = 768
    num_heads: int = 8
    norm_num_groups: int = 32
    down_block_types: tuple = (
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "DownBlock2D",
    )
    up_block_types: tuple = (
        "UpBlock2D",
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
    )
    # diffusers eps: ResnetBlock2D and Transformer2DModel GroupNorm use 1e-6,
    # the final conv_norm_out uses norm_eps=1e-5, transformer LayerNorms use 1e-5.
    resnet_eps: float = 1e-6
    attn_groupnorm_eps: float = 1e-6
    out_groupnorm_eps: float = 1e-5
    layernorm_eps: float = 1e-5


def load_weights(state_dict, dtype):
    """Remap a diffusers UNet state_dict into an MLX weights dict.

    4D conv weights are transposed NCHW -> NHWC; 1x1 convs are folded to
    channel-linear matrices ``(out, in)``. Accepts torch tensors or numpy arrays;
    only numpy is required at runtime."""
    import numpy as np

    weights = {}
    for key, tensor in state_dict.items():
        arr = tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else np.asarray(tensor)
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 4:
            out_c, in_c, kh, kw = arr.shape
            if kh == 1 and kw == 1:
                arr = arr.reshape(out_c, in_c)  # 1x1 conv -> channel linear
            else:
                arr = np.transpose(arr, (0, 2, 3, 1))  # NCHW -> NHWC
        weights[key] = mx.array(arr).astype(dtype)
    return weights


def _silu(x):
    return x * mx.sigmoid(x)


def _gelu(x):
    # Exact GELU (diffusers GEGLU default), via erf rather than the tanh approx.
    return 0.5 * x * (1.0 + mx.erf(x * 0.7071067811865476))


def _group_norm(x, weight, bias, groups, eps):
    # x is NHWC; normalize over spatial dims and the channels within each group.
    # The reduction runs in fp32: fp16 variance over a large group is numerically
    # unstable and would dominate the equivalence error.
    dtype = x.dtype
    b, h, w, c = x.shape
    xf = x.astype(mx.float32).reshape(b, h, w, groups, c // groups)
    mean = mx.mean(xf, axis=(1, 2, 4), keepdims=True)
    var = mx.var(xf, axis=(1, 2, 4), keepdims=True)
    xf = ((xf - mean) * mx.rsqrt(var + eps)).reshape(b, h, w, c)
    return (xf * weight.astype(mx.float32) + bias.astype(mx.float32)).astype(dtype)


def _layer_norm(x, weight, bias, eps):
    return mx.fast.layer_norm(x, weight, bias, eps)


def _channel_linear(x, weight, bias=None):
    # weight is (out, in); apply over the last axis. Covers Linear layers and the
    # folded 1x1 convs (proj_in/proj_out/conv_shortcut).
    y = x @ mx.swapaxes(weight, -1, -2)
    return y + bias if bias is not None else y


def _conv(x, weight, bias, stride, padding):
    y = mx.conv2d(x, weight, stride=stride, padding=padding)
    return y + bias


def _timestep_embedding(timestep, dim):
    half = dim // 2
    freqs = mx.exp(-math.log(10000.0) * mx.arange(half, dtype=mx.float32) / half)
    # timestep is passed as a 0-d mx array so a compiled graph is not re-traced
    # when its value changes (a recompile inside the timed loop would be invalid).
    if not isinstance(timestep, mx.array):
        timestep = mx.array(float(timestep))
    args = timestep.astype(mx.float32) * freqs
    # flip_sin_to_cos=True, freq_shift=0 -> [cos, sin]
    return mx.concatenate([mx.cos(args), mx.sin(args)])


def _attention(weights, prefix, hidden, context, num_heads):
    q = _channel_linear(hidden, weights[f"{prefix}.to_q.weight"])
    k = _channel_linear(context, weights[f"{prefix}.to_k.weight"])
    v = _channel_linear(context, weights[f"{prefix}.to_v.weight"])

    b, lq, c = q.shape
    lk = k.shape[1]
    d = c // num_heads
    q = q.reshape(b, lq, num_heads, d).transpose(0, 2, 1, 3)
    k = k.reshape(b, lk, num_heads, d).transpose(0, 2, 1, 3)
    v = v.reshape(b, lk, num_heads, d).transpose(0, 2, 1, 3)

    # Fused SDPA already accumulates scores/softmax in fp32 internally, so the
    # fp16 inputs are safe from the SD 1.5 attention overflow and no manual upcast
    # is needed (an explicit fp32 cast here only doubled the attention cost).
    scale = 1.0 / math.sqrt(d)
    out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
    out = out.transpose(0, 2, 1, 3).reshape(b, lq, c)
    return _channel_linear(out, weights[f"{prefix}.to_out.0.weight"], weights[f"{prefix}.to_out.0.bias"])


def _feed_forward(weights, prefix, x):
    # net.0 = GEGLU(proj: dim -> 2*inner), net.2 = Linear(inner -> dim).
    proj = _channel_linear(x, weights[f"{prefix}.net.0.proj.weight"], weights[f"{prefix}.net.0.proj.bias"])
    a, gate = mx.split(proj, 2, axis=-1)
    h = a * _gelu(gate)
    return _channel_linear(h, weights[f"{prefix}.net.2.weight"], weights[f"{prefix}.net.2.bias"])


def _basic_transformer_block(weights, prefix, x, context, cfg):
    eps = cfg.layernorm_eps
    h = _layer_norm(x, weights[f"{prefix}.norm1.weight"], weights[f"{prefix}.norm1.bias"], eps)
    x = x + _attention(weights, f"{prefix}.attn1", h, h, cfg.num_heads)  # self-attention
    h = _layer_norm(x, weights[f"{prefix}.norm2.weight"], weights[f"{prefix}.norm2.bias"], eps)
    x = x + _attention(weights, f"{prefix}.attn2", h, context, cfg.num_heads)  # cross-attention
    h = _layer_norm(x, weights[f"{prefix}.norm3.weight"], weights[f"{prefix}.norm3.bias"], eps)
    return x + _feed_forward(weights, f"{prefix}.ff", h)


def _transformer_2d(weights, prefix, x, context, cfg):
    residual = x
    b, h, w, c = x.shape
    x = _group_norm(
        x, weights[f"{prefix}.norm.weight"], weights[f"{prefix}.norm.bias"], cfg.norm_num_groups, cfg.attn_groupnorm_eps
    )
    x = _channel_linear(x, weights[f"{prefix}.proj_in.weight"], weights[f"{prefix}.proj_in.bias"])
    x = x.reshape(b, h * w, c)
    x = _basic_transformer_block(weights, f"{prefix}.transformer_blocks.0", x, context, cfg)
    x = x.reshape(b, h, w, c)
    x = _channel_linear(x, weights[f"{prefix}.proj_out.weight"], weights[f"{prefix}.proj_out.bias"])
    return residual + x


def _resnet(weights, prefix, x, temb, cfg):
    eps = cfg.resnet_eps
    groups = cfg.norm_num_groups
    h = _group_norm(x, weights[f"{prefix}.norm1.weight"], weights[f"{prefix}.norm1.bias"], groups, eps)
    h = _silu(h)
    h = _conv(h, weights[f"{prefix}.conv1.weight"], weights[f"{prefix}.conv1.bias"], stride=1, padding=1)

    t = _channel_linear(_silu(temb), weights[f"{prefix}.time_emb_proj.weight"], weights[f"{prefix}.time_emb_proj.bias"])
    h = h + t[:, None, None, :]

    h = _group_norm(h, weights[f"{prefix}.norm2.weight"], weights[f"{prefix}.norm2.bias"], groups, eps)
    h = _silu(h)
    h = _conv(h, weights[f"{prefix}.conv2.weight"], weights[f"{prefix}.conv2.bias"], stride=1, padding=1)

    shortcut = f"{prefix}.conv_shortcut.weight"
    if shortcut in weights:
        x = _channel_linear(x, weights[shortcut], weights[f"{prefix}.conv_shortcut.bias"])
    return x + h


def _upsample_nearest(x):
    b, h, w, c = x.shape
    x = mx.broadcast_to(x.reshape(b, h, 1, w, 1, c), (b, h, 2, w, 2, c))
    return x.reshape(b, h * 2, w * 2, c)


def unet_forward(weights, cfg, sample_nchw, timestep, encoder_hidden_states):
    """One UNet forward pass. Inputs/outputs are NCHW numpy-convention MLX arrays;
    encoder_hidden_states is (B, 77, cross_attention_dim)."""
    x = mx.transpose(sample_nchw, (0, 2, 3, 1))  # NCHW -> NHWC
    context = encoder_hidden_states

    temb = _timestep_embedding(timestep, cfg.block_out_channels[0])
    temb = _channel_linear(temb, weights["time_embedding.linear_1.weight"], weights["time_embedding.linear_1.bias"])
    temb = _silu(temb)
    temb = _channel_linear(temb, weights["time_embedding.linear_2.weight"], weights["time_embedding.linear_2.bias"])
    # The sinusoidal embedding is built in fp32; settle to the working dtype so the
    # rest of the graph stays in a single precision instead of promoting to fp32.
    temb = mx.broadcast_to(temb[None, :], (x.shape[0], temb.shape[0])).astype(x.dtype)

    x = _conv(x, weights["conv_in.weight"], weights["conv_in.bias"], stride=1, padding=1)
    residuals = [x]

    for i, block_type in enumerate(cfg.down_block_types):
        has_attn = block_type.startswith("CrossAttn")
        for j in range(cfg.layers_per_block):
            x = _resnet(weights, f"down_blocks.{i}.resnets.{j}", x, temb, cfg)
            if has_attn:
                x = _transformer_2d(weights, f"down_blocks.{i}.attentions.{j}", x, context, cfg)
            residuals.append(x)
        if i < len(cfg.down_block_types) - 1:  # downsampler on every block but the last
            conv = f"down_blocks.{i}.downsamplers.0.conv"
            x = _conv(x, weights[f"{conv}.weight"], weights[f"{conv}.bias"], stride=2, padding=1)
            residuals.append(x)

    x = _resnet(weights, "mid_block.resnets.0", x, temb, cfg)
    x = _transformer_2d(weights, "mid_block.attentions.0", x, context, cfg)
    x = _resnet(weights, "mid_block.resnets.1", x, temb, cfg)

    for i, block_type in enumerate(cfg.up_block_types):
        has_attn = block_type.startswith("CrossAttn")
        for j in range(cfg.layers_per_block + 1):
            x = mx.concatenate([x, residuals.pop()], axis=-1)  # channel-wise skip
            x = _resnet(weights, f"up_blocks.{i}.resnets.{j}", x, temb, cfg)
            if has_attn:
                x = _transformer_2d(weights, f"up_blocks.{i}.attentions.{j}", x, context, cfg)
        if i < len(cfg.up_block_types) - 1:  # upsampler on every block but the last
            x = _upsample_nearest(x)
            conv = f"up_blocks.{i}.upsamplers.0.conv"
            x = _conv(x, weights[f"{conv}.weight"], weights[f"{conv}.bias"], stride=1, padding=1)

    x = _group_norm(
        x, weights["conv_norm_out.weight"], weights["conv_norm_out.bias"], cfg.norm_num_groups, cfg.out_groupnorm_eps
    )
    x = _silu(x)
    x = _conv(x, weights["conv_out.weight"], weights["conv_out.bias"], stride=1, padding=1)
    return mx.transpose(x, (0, 3, 1, 2))  # NHWC -> NCHW
