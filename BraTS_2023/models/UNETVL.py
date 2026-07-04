import math
import warnings
from enum import Enum
import einops
import torch
import torch.nn.functional as F
from torch import nn
from deepkan import ChebyshevKANLayer as ChebyKANLayer
import numpy as np
import collections
import itertools
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            assert len(x) == n
            return x
        return tuple(itertools.repeat(x, n))
    return parse
def to_ntuple(x, n):
    return _ntuple(n=n)(x)
def interpolate_sincos(embed, seqlens, mode="bicubic"):
    assert embed.ndim - 2 == len(seqlens)
    embed = F.interpolate(
        einops.rearrange(embed, "1 ... dim -> 1 dim ..."),
        size=seqlens,
        mode=mode,
    )
    embed = einops.rearrange(embed, "1 dim ... -> 1 ... dim")
    return embed
class SequenceConv2d(nn.Conv2d):
    def __init__(self, *args, seqlens=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.seqlens = seqlens
    def forward(self, x):
        assert x.ndim == 3
        if self.seqlens is None:
            h = math.sqrt(x.size(1))
            assert h.is_integer()
            h = int(h)
        else:
            assert len(self.seqlens) == 2
            h = self.seqlens[0]
        x = einops.rearrange(x, "b (h w) d -> b d h w", h=h)
        x = super().forward(x)
        x = einops.rearrange(x, "b d h w -> b (h w) d")
        return x
class SequenceConv3d(nn.Conv3d):
    def __init__(self, *args, seqlens=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.seqlens = seqlens
    def forward(self, x):
        assert x.ndim == 3                                              
        if self.seqlens is None:
            d = int(math.pow(x.size(1), 1/3))
            assert d ** 3 == x.size(1), "Input sequence length is not a perfect cube"
        else:
            assert len(self.seqlens) == 3, "seqlens must be a tuple of three dimensions (D, H, W)"
            d, h, w = self.seqlens
        x = einops.rearrange(x, "b (d h w) c -> b c d h w", d=d, h=h, w=w)
        x = super().forward(x)
        x = einops.rearrange(x, "b c d h w -> b (d h w) c")
        return x
class VitPatchEmbed(nn.Module):
    def __init__(self, dim, num_channels, resolution, patch_size, stride=None, init_weights="xavier_uniform"):
        super().__init__()
        self.resolution = resolution
        self.init_weights = init_weights
        self.ndim = len(resolution)
        self.patch_size = to_ntuple(patch_size, n=self.ndim)
        if stride is None:
            self.stride = self.patch_size
        else:
            self.stride = to_ntuple(stride, n=self.ndim)
        for i in range(self.ndim):
            assert resolution[i] % self.patch_size[i] == 0,\
                f"resolution[{i}] % patch_size[{i}] != 0 (resolution={resolution} patch_size={patch_size})"
        self.seqlens = [resolution[i] // self.patch_size[i] for i in range(self.ndim)]
        if self.patch_size == self.stride:
            self.num_patches = int(np.prod(self.seqlens))
        else:
            if self.ndim == 1:
                conv_func = F.conv1d
            elif self.ndim == 2:
                conv_func = F.conv2d
            elif self.ndim == 3:
                conv_func = F.conv3d
            else:
                raise NotImplementedError
            self.num_patches = conv_func(
                input=torch.zeros(1, 1, *resolution),
                weight=torch.zeros(1, 1, *self.patch_size),
                stride=self.stride,
            ).numel()
        if self.ndim == 1:
            conv_ctor = nn.Conv1d
        elif self.ndim == 2:
            conv_ctor = nn.Conv2d
        elif self.ndim == 3:
            conv_ctor = nn.Conv3d
        else:
            raise NotImplementedError
        self.proj = conv_ctor(num_channels, dim, kernel_size=self.patch_size, stride=self.stride)
        self.reset_parameters()
    def reset_parameters(self):
        if self.init_weights == "torch":
            pass
        elif self.init_weights == "xavier_uniform":
            w = self.proj.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            nn.init.zeros_(self.proj.bias)
        else:
            raise NotImplementedError
    def forward(self, x):
        assert all(x.size(i + 2) % self.patch_size[i] == 0 for i in range(self.ndim)),\
            f"x.shape={x.shape} incompatible with patch_size={self.patch_size}"
        x = self.proj(x)
        x = einops.rearrange(x, "b c ... -> b ... c")
        return x
class VitPosEmbed2d(nn.Module):
    def __init__(self, seqlens, dim: int, allow_interpolation: bool = True):
        super().__init__()
        self.seqlens = seqlens
        self.dim = dim
        self.allow_interpolation = allow_interpolation
        self.embed = nn.Parameter(torch.zeros(1, *seqlens, dim))
        self.reset_parameters()
    @property
    def _expected_x_ndim(self):
        return len(self.seqlens) + 2
    def reset_parameters(self):
        nn.init.trunc_normal_(self.embed, std=.02)
    def forward(self, x):
        assert x.ndim == self._expected_x_ndim
        if x.shape[1:] != self.embed.shape[1:]:
            assert self.allow_interpolation
            embed = interpolate_sincos(embed=self.embed, seqlens=x.shape[1:-1])
        else:
            embed = self.embed
        return x + embed
class VitPosEmbed3d(nn.Module):
    def __init__(self, seqlens, dim: int, allow_interpolation: bool = True):
        super().__init__()
        self.seqlens = seqlens                                                           
        self.dim = dim
        self.allow_interpolation = allow_interpolation
        self.embed = nn.Parameter(torch.zeros(1, *seqlens, dim))
        self.reset_parameters()
    @property
    def _expected_x_ndim(self):
        return len(self.seqlens) + 2
    def reset_parameters(self):
        nn.init.trunc_normal_(self.embed, std=.02)
    def forward(self, x):
        assert x.ndim == self._expected_x_ndim, "Input tensor does not have expected number of dimensions"
        if x.shape[1:] != self.embed.shape[1:]:
            assert self.allow_interpolation, "Interpolation must be allowed to adjust positional embeddings"
            embed = interpolate_sincos(embed=self.embed, seqlens=x.shape[2:-1], mode="trilinear")                          
        else:
            embed = self.embed
        return x + embed
class DropPath(nn.Sequential):
    """"""
    def __init__(self, *args, drop_prob: float = 0., scale_by_keep: bool = True, stochastic_drop_prob: bool = False):
        super().__init__(*args)
        assert 0. <= drop_prob < 1.
        self._drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep
        self.stochastic_drop_prob = stochastic_drop_prob
    @property
    def drop_prob(self):
        return self._drop_prob
    @drop_prob.setter
    def drop_prob(self, value):
        assert 0. <= value < 1.
        self._drop_prob = value
    @property
    def keep_prob(self):
        return 1. - self.drop_prob
    def forward(self, x, residual_path=None, residual_path_kwargs=None):
        assert (len(self) == 0) ^ (residual_path is None)
        residual_path_kwargs = residual_path_kwargs or {}
        if self.drop_prob == 0. or not self.training:
            if residual_path is None:
                return x + super().forward(x, **residual_path_kwargs)
            else:
                return x + residual_path(x, **residual_path_kwargs)
        bs = len(x)
        if self.stochastic_drop_prob:
            perm = torch.empty(bs, device=x.device).bernoulli_(self.keep_prob).nonzero().squeeze(1)
            scale = 1 / self.keep_prob
        else:
            keep_count = max(int(bs * self.keep_prob), 1)
            scale = bs / keep_count
            perm = torch.randperm(bs, device=x.device)[:keep_count]
        if self.scale_by_keep:
            alpha = scale
        else:
            alpha = 1.
        residual_path_kwargs = {
            key: value[perm] if torch.is_tensor(value) else value
            for key, value in residual_path_kwargs.items()
        }
        if residual_path is None:
            residual = super().forward(x[perm], **residual_path_kwargs)
        else:
            residual = residual_path(x[perm], **residual_path_kwargs)
        return torch.index_add(
            x.flatten(start_dim=1),
            dim=0,
            index=perm,
            source=residual.to(x.dtype).flatten(start_dim=1),
            alpha=alpha,
        ).view_as(x)
    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob, 3):0.3f}'
class SequenceTraversal(Enum):
    ROWWISE_FROM_TOP_LEFT = "rowwise_from_top_left"
    ROWWISE_FROM_BOT_RIGHT = "rowwise_from_bot_right"
def bias_linspace_init_(param: torch.Tensor, start: float = 3.4, end: float = 6.0) -> torch.Tensor:
    """"""
    assert param.dim() == 1, f"param must be 1-dimensional (typically a bias), got {param.dim()}"
    n_dims = param.shape[0]
    init_vals = torch.linspace(start, end, n_dims)
    with torch.no_grad():
        param.copy_(init_vals)
    return param
def small_init_(param: torch.Tensor, dim: int) -> torch.Tensor:
    """"""
    std = math.sqrt(2 / (5 * dim))
    torch.nn.init.normal_(param, mean=0.0, std=std)
    return param
def wang_init_(param: torch.Tensor, dim: int, num_blocks: int):
    """"""
    std = 2 / num_blocks / math.sqrt(dim)
    torch.nn.init.normal_(param, mean=0.0, std=std)
    return param
def parallel_stabilized_simple(
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        igate_preact: torch.Tensor,
        fgate_preact: torch.Tensor,
        lower_triangular_matrix: torch.Tensor = None,
        stabilize_rowwise: bool = True,
        eps: float = 1e-6,
) -> torch.Tensor:
    """"""
    B, NH, S, DH = queries.shape
    _dtype, _device = queries.dtype, queries.device
    log_fgates = torch.nn.functional.logsigmoid(fgate_preact)                 
    if lower_triangular_matrix is None or S < lower_triangular_matrix.size(-1):
        ltr = torch.tril(torch.ones((S, S), dtype=torch.bool, device=_device))
    else:
        ltr = lower_triangular_matrix
    assert ltr.dtype == torch.bool, f"lower_triangular_matrix must be of dtype bool, got {ltr.dtype}"
    log_fgates_cumsum = torch.cat(
        [
            torch.zeros((B, NH, 1, 1), dtype=_dtype, device=_device),
            torch.cumsum(log_fgates, dim=-2),
        ],
        dim=-2,
    )                   
    rep_log_fgates_cumsum = log_fgates_cumsum.repeat(1, 1, 1, S + 1)                     
    _log_fg_matrix = rep_log_fgates_cumsum - rep_log_fgates_cumsum.transpose(-2, -1)                     
    log_fg_matrix = torch.where(ltr, _log_fg_matrix[:, :, 1:, 1:], -float("inf"))                 
    log_D_matrix = log_fg_matrix + igate_preact.transpose(-2, -1)                 
    if stabilize_rowwise:
        max_log_D, _ = torch.max(log_D_matrix, dim=-1, keepdim=True)                 
    else:
        max_log_D = torch.max(log_D_matrix.view(B, NH, -1), dim=-1, keepdim=True)[0].unsqueeze(-1)
    log_D_matrix_stabilized = log_D_matrix - max_log_D                 
    D_matrix = torch.exp(log_D_matrix_stabilized)                 
    keys_scaled = keys / math.sqrt(DH)
    qk_matrix = queries @ keys_scaled.transpose(-2, -1)                 
    C_matrix = qk_matrix * D_matrix                 
    normalizer = torch.maximum(C_matrix.sum(dim=-1, keepdim=True).abs(), torch.exp(-max_log_D))                 
    C_matrix_normalized = C_matrix / (normalizer + eps)
    h_tilde_state = C_matrix_normalized @ values                  
    return h_tilde_state
class LinearHeadwiseExpand(nn.Module):
    """"""
    def __init__(self, dim, num_heads, bias=False):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        dim_per_head = dim // num_heads
        self.weight = nn.Parameter(torch.empty(num_heads, dim_per_head, dim_per_head))
        if bias:
            self.bias = nn.Parameter(torch.empty(dim))
        else:
            self.bias = None
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.normal_(self.weight.data, mean=0.0, std=math.sqrt(2 / 5 / self.weight.shape[-1]))
        if self.bias is not None:
            nn.init.zeros_(self.bias.data)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = einops.rearrange(x, "... (nh d) -> ... nh d", nh=self.num_heads)
        x = einops.einsum(
            x,
            self.weight,
            "... nh d, nh out_d d -> ... nh out_d",
        )
        x = einops.rearrange(x, "... nh out_d -> ... (nh out_d)")
        if self.bias is not None:
            x = x + self.bias
        return x
    def extra_repr(self):
        return (
            f"dim={self.dim}, "
            f"num_heads={self.num_heads}, "
            f"bias={self.bias is not None}, "
        )
class CausalConv1d(nn.Module):
    """"""
    def __init__(self, dim, kernel_size=4, bias=True):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.bias = bias
        self.pad = kernel_size - 1
        self.conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            padding=self.pad,
            groups=dim,
            bias=bias,
        )
        self.reset_parameters()
    def reset_parameters(self):
        self.conv.reset_parameters()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = einops.rearrange(x, "b l d -> b d l")
        x = self.conv(x)
        x = x[:, :, :-self.pad]
        x = einops.rearrange(x, "b d l -> b l d")
        return x
class LayerNorm(nn.Module):
    """"""
    def __init__(
            self,
            ndim: int = -1,
            weight: bool = True,
            bias: bool = False,
            eps: float = 1e-5,
            residual_weight: bool = True,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(ndim)) if weight else None
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        self.eps = eps
        self.residual_weight = residual_weight
        self.ndim = ndim
        self.reset_parameters()
    @property
    def weight_proxy(self) -> torch.Tensor:
        if self.weight is None:
            return None
        if self.residual_weight:
            return 1.0 + self.weight
        else:
            return self.weight
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x,
            normalized_shape=(self.ndim,),
            weight=self.weight_proxy,
            bias=self.bias,
            eps=self.eps,
        )
    def reset_parameters(self):
        if self.weight_proxy is not None:
            if self.residual_weight:
                nn.init.zeros_(self.weight)
            else:
                nn.init.ones_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
class MultiHeadLayerNorm(LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4, "Input must be 4D tensor (B, NH, S, DH)"
        B, NH, S, DH = x.shape
        gn_in_1 = x.transpose(1, 2)                  
        gn_in_2 = gn_in_1.reshape(B * S, NH * DH)                    
        out = F.group_norm(
            gn_in_2,
            num_groups=NH,
            weight=self.weight_proxy,
            bias=self.bias,
            eps=self.eps,
        )                
        out = out.view(B, S, NH, DH).transpose(1, 2)
        return out
class MatrixLSTMCell(nn.Module):
    def __init__(self, dim, num_heads, norm_bias=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.igate = nn.Linear(3 * dim, num_heads)
        self.fgate = nn.Linear(3 * dim, num_heads)
        self.outnorm = MultiHeadLayerNorm(ndim=dim, weight=True, bias=norm_bias)
        self.causal_mask_cache = {}
        self.reset_parameters()
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        B, S, _ = q.shape             
        if_gate_input = torch.cat([q, k, v], dim=-1)
        q = q.view(B, S, self.num_heads, -1)                  
        k = k.view(B, S, self.num_heads, -1)                  
        v = v.view(B, S, self.num_heads, -1)                  
        q = q.transpose(1, 2)                  
        k = k.transpose(1, 2)                  
        v = v.transpose(1, 2)                  
        igate_preact = self.igate(if_gate_input)              
        igate_preact = igate_preact.transpose(-1, -2).unsqueeze(-1)                 
        fgate_preact = self.fgate(if_gate_input)              
        fgate_preact = fgate_preact.transpose(-1, -2).unsqueeze(-1)                  
        if S in self.causal_mask_cache:
            causal_mask = self.causal_mask_cache[(S, str(q.device))]
        else:
            causal_mask = torch.tril(torch.ones(S, S, dtype=torch.bool, device=q.device))
            self.causal_mask_cache[(S, str(q.device))] = causal_mask
        h_state = parallel_stabilized_simple(
            queries=q,
            keys=k,
            values=v,
            igate_preact=igate_preact,
            fgate_preact=fgate_preact,
            lower_triangular_matrix=causal_mask,
        )                  
        h_state_norm = self.outnorm(h_state)                  
        h_state_norm = h_state_norm.transpose(1, 2).reshape(B, S, -1)                                                 
        return h_state_norm
    def reset_parameters(self):
        self.outnorm.reset_parameters()
        torch.nn.init.zeros_(self.fgate.weight)
        bias_linspace_init_(self.fgate.bias, start=3.0, end=6.0)
        torch.nn.init.zeros_(self.igate.weight)
        torch.nn.init.normal_(self.igate.bias, mean=0.0, std=0.1)
class ViLLayer(nn.Module):
    def __init__(
            self,
            dim,
            direction,
            expansion=2,
            qkv_block_size=4,
            proj_bias=True,
            norm_bias=True,
            conv_bias=True,
            conv_kernel_size=4,
            conv_kind="3d",
            seqlens=None,
            no_kan=True,
    ):
        super().__init__()
        assert dim % qkv_block_size == 0
        self.dim = dim
        self.direction = direction
        self.expansion = expansion
        self.qkv_block_size = qkv_block_size
        self.proj_bias = proj_bias
        self.conv_bias = conv_bias
        self.conv_kernel_size = conv_kernel_size
        self.conv_kind = conv_kind
        self.no_kan = no_kan
        inner_dim = expansion * dim
        num_heads = inner_dim // qkv_block_size
        if no_kan:
            self.proj_up = nn.Linear(
                in_features=dim,
                out_features=2 * inner_dim,
                bias=proj_bias,
            )
        else:
            self.proj_up = ChebyKANLayer(
                input_dim=dim,
                output_dim=2 * inner_dim,
                degree=3,
            )
        self.q_proj = LinearHeadwiseExpand(
            dim=inner_dim,
            num_heads=num_heads,
            bias=proj_bias,
        )
        self.k_proj = LinearHeadwiseExpand(
            dim=inner_dim,
            num_heads=num_heads,
            bias=proj_bias,
        )
        self.v_proj = LinearHeadwiseExpand(
            dim=inner_dim,
            num_heads=num_heads,
            bias=proj_bias,
        )
        if conv_kind == "causal1d":
            self.conv = CausalConv1d(
                dim=inner_dim,
                kernel_size=conv_kernel_size,
                bias=conv_bias,
            )
        elif conv_kind == "2d":
            assert conv_kernel_size % 2 == 1,\
                f"same output shape as input shape is required -> even kernel sizes not supported"
            self.conv = SequenceConv2d(
                in_channels=inner_dim,
                out_channels=inner_dim,
                kernel_size=conv_kernel_size,
                padding=conv_kernel_size // 2,
                groups=inner_dim,
                bias=conv_bias,
                seqlens=seqlens,
            )
        elif conv_kind == "3d":
            assert conv_kernel_size % 2 == 1,\
                f"same output shape as input shape is required -> even kernel sizes not supported"
            self.conv = SequenceConv3d(
                in_channels=inner_dim,
                out_channels=inner_dim,
                kernel_size=conv_kernel_size,
                padding=conv_kernel_size // 2,
                groups=inner_dim,
                bias=conv_bias,
                seqlens=seqlens,
            )
        else:
            raise NotImplementedError
        self.mlstm_cell = MatrixLSTMCell(
            dim=inner_dim,
            num_heads=qkv_block_size,
            norm_bias=norm_bias,
        )
        self.learnable_skip = nn.Parameter(torch.ones(inner_dim))
        if no_kan:
            self.proj_down = nn.Linear(
                in_features=inner_dim,
                out_features=dim,
                bias=proj_bias,
            )
        else:
            self.proj_down = ChebyKANLayer(
                input_dim=inner_dim,
                output_dim=dim,
                degree=3,
            )
        if self.no_kan:
            self.reset_parameters()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        if self.direction == SequenceTraversal.ROWWISE_FROM_TOP_LEFT:
            pass
        elif self.direction == SequenceTraversal.ROWWISE_FROM_BOT_RIGHT:
            x = x.flip(dims=[1])
        else:
            raise NotImplementedError
        x_inner = self.proj_up(x)
        if not self.no_kan:    
            x_inner = einops.rearrange(x_inner, "(b p) h -> b p h", p=S, h=x_inner.shape[-1])
        x_mlstm, z = torch.chunk(x_inner, chunks=2, dim=-1)
        x_mlstm_conv = self.conv(x_mlstm)
        x_mlstm_conv_act = F.silu(x_mlstm_conv)
        q = self.q_proj(x_mlstm_conv_act)
        k = self.k_proj(x_mlstm_conv_act)
        v = self.v_proj(x_mlstm)
        h_tilde_state = self.mlstm_cell(q=q, k=k, v=v)
        h_tilde_state_skip = h_tilde_state + (self.learnable_skip * x_mlstm_conv_act)
        h_state = h_tilde_state_skip * F.silu(z)
        x = self.proj_down(h_state)
        if not self.no_kan:
            x = einops.rearrange(x, "(b p) h -> b p h", p=S, h=x.shape[-1])
        if self.direction == SequenceTraversal.ROWWISE_FROM_TOP_LEFT:
            pass
        elif self.direction == SequenceTraversal.ROWWISE_FROM_BOT_RIGHT:
            x = x.flip(dims=[1])
        else:
            raise NotImplementedError
        return x
    def reset_parameters(self):
        if self.no_kan:
            small_init_(self.proj_up.weight, dim=self.dim)
            if self.proj_up.bias is not None:
                nn.init.zeros_(self.proj_up.bias)
            wang_init_(self.proj_down.weight, dim=self.dim, num_blocks=1)
            if self.proj_down.bias is not None:
                nn.init.zeros_(self.proj_down.bias)
        nn.init.ones_(self.learnable_skip)
        def _init_qkv_proj(qkv_proj: LinearHeadwiseExpand):
            small_init_(qkv_proj.weight, dim=self.dim)
            if qkv_proj.bias is not None:
                nn.init.zeros_(qkv_proj.bias)
        _init_qkv_proj(self.q_proj)
        _init_qkv_proj(self.k_proj)
        _init_qkv_proj(self.v_proj)
        self.mlstm_cell.reset_parameters()
class ViLBlock(nn.Module):
    def __init__(
            self,
            dim,
            direction,
            drop_path=0.0,
            conv_kind="3d",
            conv_kernel_size=3,
            proj_bias=True,
            norm_bias=True,
            seqlens=None,
            no_kan=True,
    ):
        super().__init__()
        self.dim = dim
        self.direction = direction
        self.drop_path = drop_path
        self.conv_kind = conv_kind
        self.conv_kernel_size = conv_kernel_size
        self.drop_path = DropPath(drop_prob=drop_path)
        self.norm = LayerNorm(ndim=dim, weight=True, bias=norm_bias)
        self.layer = ViLLayer(
            dim=dim,
            direction=direction,
            conv_kind=conv_kind,
            conv_kernel_size=conv_kernel_size,
            seqlens=seqlens,
            norm_bias=norm_bias,
            proj_bias=proj_bias,
            no_kan=no_kan,
        )
        self.seqlens = seqlens
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=dim, out_channels=2*dim, kernel_size=3, padding=1),
            nn.LayerNorm([2*dim, *seqlens]),
            nn.SiLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=2*dim, out_channels=dim, kernel_size=3, padding=1),
            nn.LayerNorm([dim, *seqlens]),
            nn.SiLU()
        )
        self.reset_parameters()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x2 = self.norm(x)
        x2 = self.layer(x2)
        d, h, w = self.seqlens
        x1 = einops.rearrange(x, "b (d h w) c -> b c d h w", d=d, h=h, w=w)
        x1 = self.conv1(x1)
        x1 = self.conv2(x1)
        x1 = einops.rearrange(x1, "b c d h w -> b (d h w) c")
        out = x1 + x2
        return out
    def reset_parameters(self):
        self.layer.reset_parameters()
        self.norm.reset_parameters()
class ViLBlockPair(nn.Module):
    def __init__(
            self,
            dim,
            drop_path=0.0,
            conv_kind="3d",
            conv_kernel_size=3,
            proj_bias=True,
            norm_bias=True,
            seqlens=None,
            no_kan=True,
    ):
        super().__init__()
        self.rowwise_from_top_left = ViLBlock(
            dim=dim,
            direction=SequenceTraversal.ROWWISE_FROM_TOP_LEFT,
            drop_path=drop_path,
            conv_kind=conv_kind,
            conv_kernel_size=conv_kernel_size,
            proj_bias=proj_bias,
            norm_bias=norm_bias,
            seqlens=seqlens,
            no_kan=no_kan
        )
        self.rowwise_from_bot_right = ViLBlock(
            dim=dim,
            direction=SequenceTraversal.ROWWISE_FROM_BOT_RIGHT,
            drop_path=drop_path,
            conv_kind=conv_kind,
            conv_kernel_size=conv_kernel_size,
            proj_bias=proj_bias,
            norm_bias=norm_bias,
            seqlens=seqlens,
            no_kan=no_kan
        )
    def forward(self, x):
        x = self.rowwise_from_top_left(x)
        x = self.rowwise_from_bot_right(x)
        return x
class VisionLSTM2(nn.Module):
    def __init__(
            self,
            dim=192,
            input_shape=(3, 224, 224),
            patch_size=16,
            depth=12,
            output_shape=(1000,),
            mode="classifier",
            pooling="bilateral_flatten",
            drop_path_rate=0.0,
            drop_path_decay=False,
            stride=None,
            legacy_norm=False,
            conv_kind="3d",
            conv_kernel_size=3,
            proj_bias=True,
            norm_bias=True,
            no_kan=True,
    ):
        if depth == 24 and dim < 1024:
            warnings.warn(
                "A single VisionLSTM2 block consists of two subblocks (one for each traversal direction). "
                "ViL-T, ViL-S and ViL-B therefore use depth=12 instead of depth=24, are you sure you want to use "
                "depth=24?"
            )
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        ndim = len(self.input_shape) - 1
        self.patch_size = to_ntuple(patch_size, n=ndim)
        self.dim = dim
        self.depth = depth
        self.stride = stride
        self.mode = mode
        self.pooling = pooling
        self.drop_path_rate = drop_path_rate
        self.drop_path_decay = drop_path_decay
        self.conv_kind = conv_kind
        self.conv_kernel_size = conv_kernel_size
        self.proj_bias = proj_bias
        self.norm_bias = norm_bias
        self.no_kan = no_kan
        self.patch_embed = VitPatchEmbed(
            dim=dim,
            stride=stride,
            num_channels=self.input_shape[0],
            resolution=self.input_shape[1:],
            patch_size=self.patch_size,
        )
        if ndim == 2:
            self.pos_embed = VitPosEmbed2d(seqlens=self.patch_embed.seqlens, dim=dim)
        elif ndim == 3:
            self.pos_embed = VitPosEmbed3d(seqlens=self.patch_embed.seqlens, dim=dim)
        if drop_path_decay and drop_path_rate > 0.:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        else:
            dpr = [drop_path_rate] * depth
        self.blocks = nn.ModuleList(
            [
                ViLBlockPair(
                    dim=dim,
                    drop_path=dpr[i],
                    conv_kind=conv_kind,
                    seqlens=self.patch_embed.seqlens,
                    proj_bias=proj_bias,
                    norm_bias=norm_bias,
                    no_kan=no_kan,
                )
                for i in range(depth)
            ],
        )
        if pooling == "bilateral_flatten":
            head_dim = dim * 2
        else:
            head_dim = dim
        self.norm = LayerNorm(dim, bias=norm_bias, eps=1e-6)
        if legacy_norm:
            self.legacy_norm = nn.LayerNorm(head_dim)
        else:
            self.legacy_norm = nn.Identity()
        self.norm1 = LayerNorm(dim, bias=norm_bias, eps=1e-6)
        self.norm2 = LayerNorm(dim, bias=norm_bias, eps=1e-6)
        self.norm3 = LayerNorm(dim, bias=norm_bias, eps=1e-6)
        self.norm4 = LayerNorm(dim, bias=norm_bias, eps=1e-6)
        if mode == "features":
            assert self.output_shape is None
            self.head = None
            if self.pooling is None:
                self.output_shape = (self.patch_embed.num_patches, dim)
            elif self.pooling == "to_image":
                self.output_shape = (dim, *self.patch_embed.seqlens)
            else:
                raise NotImplementedError(f"invalid pooling '{pooling}' for mode '{mode}'")
        elif mode == "classifier":
            assert self.output_shape is not None and len(self.output_shape) == 1,\
                f"define number of classes via output_shape=(num_classes,) (e.g. output_shape=(1000,) for ImageNet-1K"
            self.head = nn.Linear(head_dim, self.output_shape[0])
            nn.init.trunc_normal_(self.head.weight, std=2e-5)
            nn.init.zeros_(self.head.bias)
        else:
            raise NotImplementedError
    def load_state_dict(self, state_dict, strict=True):
        old_pos_embed = state_dict["pos_embed.embed"]
        if old_pos_embed.shape != self.pos_embed.embed.shape:
            state_dict["pos_embed.embed"] = interpolate_sincos(embed=old_pos_embed, seqlens=self.pos_embed.seqlens)
        return super().load_state_dict(state_dict=state_dict, strict=strict)
    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed.embed"}
    def forward(self, x):
        x = self.patch_embed(x)
        x = self.pos_embed(x)
        x = einops.rearrange(x, "b ... d -> b (...) d")
        idx = self.depth // 4
        skip_connection_index = [int(idx), int(2 * idx), int(3 * idx), int(4 * idx)]
        skip_connections = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i+1 in skip_connection_index:
                skip_connections.append(x)
        x = self.norm(x)
        if self.pooling is None:
            x = self.legacy_norm(x)
        elif self.pooling == "to_image":
            x = self.legacy_norm(x)
            seqlen_h, seqlen_w = self.patch_embed.seqlens
            x = einops.rearrange(
                x,
                "b (seqlen_h seqlen_w) dim -> b dim seqlen_h seqlen_w",
                seqlen_h=seqlen_h,
                seqlen_w=seqlen_w,
            )
        elif self.pooling == "bilateral_avg":
            x = (x[:, 0] + x[:, -1]) / 2
            x = self.legacy_norm(x)
        elif self.pooling == "bilateral_flatten":
            x = torch.concat([x[:, 0], x[:, -1]], dim=1)
            x = self.legacy_norm(x)
        else:
            raise NotImplementedError(f"pooling '{self.pooling}' is not implemented")
        if self.head is not None:
            x = self.head(x)
        skip_connections[0] = self.norm1(skip_connections[0])
        skip_connections[1] = self.norm2(skip_connections[1])
        skip_connections[2] = self.norm3(skip_connections[2])
        skip_connections[3] = self.norm4(skip_connections[3])
        return skip_connections
class SingleDeconv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.block = nn.ConvTranspose3d(in_planes, out_planes, kernel_size=2, stride=2, padding=0, output_padding=0)
    def forward(self, x):
        return self.block(x)
class SingleConv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size):
        super().__init__()
        self.block = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=1,
                               padding=((kernel_size - 1) // 2))
    def forward(self, x):
        return self.block(x)
class Conv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            SingleConv3DBlock(in_planes, out_planes, kernel_size),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True)
        )
    def forward(self, x):
        return self.block(x)
class Deconv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            SingleDeconv3DBlock(in_planes, out_planes),
            SingleConv3DBlock(out_planes, out_planes, kernel_size),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True)
        )
    def forward(self, x):
        return self.block(x)
class UNETR_LSTM(nn.Module):
    def __init__(self, img_shape=(128, 128, 128), input_dim=2, output_dim=2, embed_dim=192, patch_size=16, depth=12, dropout=0.0, no_kan=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.img_shape = img_shape
        self.patch_size = patch_size
        self.depth = depth
        self.dropout = dropout
        self.patch_dim = [int(x / patch_size) for x in img_shape]
        self.ViL = VisionLSTM2(
            dim = embed_dim,
            input_shape = (input_dim, *self.img_shape),
            depth = depth,
            mode = "features",
            output_shape = None,
            drop_path_rate=self.dropout,
            pooling = None,
            no_kan = no_kan)
        self.decoder0 =\
            nn.Sequential(
                Conv3DBlock(input_dim, 32, 3),
                Conv3DBlock(32, 64, 3)
            )
        self.decoder3 =\
            nn.Sequential(
                Deconv3DBlock(embed_dim, 512),
                Deconv3DBlock(512, 256),
                Deconv3DBlock(256, 128)
            )
        self.decoder6 =\
            nn.Sequential(
                Deconv3DBlock(embed_dim, 512),
                Deconv3DBlock(512, 256),
            )
        self.decoder9 =\
            Deconv3DBlock(embed_dim, 512)
        self.decoder12_upsampler =\
            SingleDeconv3DBlock(embed_dim, 512)
        self.decoder9_upsampler =\
            nn.Sequential(
                Conv3DBlock(1024, 512),
                Conv3DBlock(512, 512),
                Conv3DBlock(512, 512),
                SingleDeconv3DBlock(512, 256)
            )
        self.decoder6_upsampler =\
            nn.Sequential(
                Conv3DBlock(512, 256),
                Conv3DBlock(256, 256),
                SingleDeconv3DBlock(256, 128)
            )
        self.decoder3_upsampler =\
            nn.Sequential(
                Conv3DBlock(256, 128),
                Conv3DBlock(128, 128),
                SingleDeconv3DBlock(128, 64)
            )
        self.decoder0_header =\
            nn.Sequential(
                Conv3DBlock(128, 64),
                Conv3DBlock(64, 64),
                SingleConv3DBlock(64, output_dim, 1)
            )
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=512, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        z = self.ViL(x)
        z0, z3, z6, z9, z12 = x, *z
        z3 = z3.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        z6 = z6.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        z9 = z9.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        z12 = z12.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        z12 = self.decoder12_upsampler(z12)
        z9 = self.decoder9(z9)
        z9 = self.decoder9_upsampler(torch.cat([z9, z12], dim=1))
        z6 = self.decoder6(z6)
        z6 = self.decoder6_upsampler(torch.cat([z6, z9], dim=1))
        z3 = self.decoder3(z3)
        z3 = self.decoder3_upsampler(torch.cat([z3, z6], dim=1))
        z0 = self.decoder0(z0)
        output = self.decoder0_header(torch.cat([z0, z3], dim=1))
        return output
if __name__ == "__main__":
    device = torch.device("cuda:1")
    model = UNETR_LSTM(input_dim=4,output_dim=5)
    x = torch.randn(1, 4, 128, 128, 128)
    out = model(x)
    print("Output shape:", out.shape)
    from thop import profile
    flops, params = profile(model, inputs=(x,))
    print("FLOPs: {:.2f} M".format(flops / 1e6))
    print("Params: {:.2f} M".format(params / 1e6))
