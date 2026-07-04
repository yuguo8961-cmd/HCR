from mmengine.model import BaseModule
import torch.utils.checkpoint as cp
from torch.utils.cpp_extension import load
from .utils.drop import DropPath
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
wkv_cuda = load(name="bi_wkv", sources=["./models/bi_wkv.cpp", "./models/bi_wkv_kernel.cu"],
                verbose=True,
                extra_cuda_cflags=['-res-usage', '--maxrregcount 128', '--use_fast_math', '-O3', '-Xptxas -O3',
                                   '-gencode arch=compute_86,code=sm_86'])
class WKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, u, k, v):
        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)
        ctx.save_for_backward(w, u, k, v)
        w = w.float().contiguous()
        u = u.float().contiguous()
        k = k.float().contiguous()
        v = v.float().contiguous()
        y = wkv_cuda.bi_wkv_forward(w, u, k, v)
        if half_mode:
            y = y.half()
        elif bf_mode:
            y = y.bfloat16()
        return y
    @staticmethod
    def backward(ctx, gy):
        w, u, k, v = ctx.saved_tensors
        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)
        gw, gu, gk, gv = wkv_cuda.bi_wkv_backward(w.float().contiguous(),
                                                  u.float().contiguous(),
                                                  k.float().contiguous(),
                                                  v.float().contiguous(),
                                                  gy.float().contiguous())
        if half_mode:
            return (gw.half(), gu.half(), gk.half(), gv.half())
        elif bf_mode:
            return (gw.bfloat16(), gu.bfloat16(), gk.bfloat16(), gv.bfloat16())
        else:
            return (gw, gu, gk, gv)
def RUN_CUDA(w, u, k, v):
    return WKV.apply(w.cuda(), u.cuda(), k.cuda(), v.cuda())


def q_shift_3d(input, shift_pixel=1, gamma=1 / 6, patch_resolution=None):
    assert gamma <= 1 / 6, "3D requires 6 directions, gamma should be <= 1/6"
    assert patch_resolution is not None and len(patch_resolution) == 3,\
        "patch_resolution must be a 3-tuple (D, H, W)"
    B, N, C = input.shape
    D, H, W = patch_resolution
    assert N == D * H * W, f"N={N} must equal D*H*W={D * H * W}"
    input = input.transpose(1, 2).reshape(B, C, D, H, W)
    output = torch.zeros_like(input)
    c1 = int(C * gamma)
    c2 = int(C * gamma * 2)
    c3 = int(C * gamma * 3)
    c4 = int(C * gamma * 4)
    c5 = int(C * gamma * 5)
    c6 = int(C * gamma * 6)
    output[:, 0:c1, :, :, shift_pixel:W] = input[:, 0:c1, :, :, 0:W - shift_pixel]
    output[:, c1:c2, :, :, 0:W - shift_pixel] = input[:, c1:c2, :, :, shift_pixel:W]
    output[:, c2:c3, :, shift_pixel:H, :] = input[:, c2:c3, :, 0:H - shift_pixel, :]
    output[:, c3:c4, :, 0:H - shift_pixel, :] = input[:, c3:c4, :, shift_pixel:H, :]
    output[:, c4:c5, shift_pixel:D, :, :] = input[:, c4:c5, 0:D - shift_pixel, :, :]
    output[:, c5:c6, 0:D - shift_pixel, :, :] = input[:, c5:c6, shift_pixel:D, :, :]
    output[:, c6:, ...] = input[:, c6:, ...]
    return output.flatten(2).transpose(1, 2)
class VRWKV_SpatialMix_3D(BaseModule):
    def __init__(self, n_embd, n_layer, layer_id, shift_mode='q_shift_3d',
                 channel_gamma=1 / 6, shift_pixel=1, init_mode='fancy', k_norm=True):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.device = None
        attn_sz = n_embd
        self._init_weights(init_mode)
        self.shift_pixel = shift_pixel
        self.shift_mode = shift_mode
        self.channel_gamma = channel_gamma
        if shift_pixel > 0:
            self.shift_func = q_shift_3d
        else:
            self.spatial_mix_k = None
            self.spatial_mix_v = None
            self.spatial_mix_r = None
        self.key = nn.Linear(n_embd, attn_sz, bias=False)
        self.value = nn.Linear(n_embd, attn_sz, bias=False)
        self.receptance = nn.Linear(n_embd, attn_sz, bias=False)
        if k_norm:
            self.key_norm = nn.LayerNorm(attn_sz)
        else:
            self.key_norm = None
        self.output = nn.Linear(attn_sz, n_embd, bias=False)
        self.key.scale_init = 0
        self.receptance.scale_init = 0
        self.output.scale_init = 0
        self.value.scale_init = 1
    def _init_weights(self, init_mode):
        if init_mode == 'fancy':
            with torch.no_grad():
                ratio_0_to_1 = (self.layer_id / (self.n_layer - 1))
                ratio_1_to_almost0 = (1.0 - (self.layer_id / self.n_layer))
                decay_speed = torch.ones(self.n_embd)
                for h in range(self.n_embd):
                    decay_speed[h] = -5 + 8 * (h / (self.n_embd - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
                self.spatial_decay = nn.Parameter(decay_speed)
                import math
                zigzag = (torch.tensor([(i + 1) % 3 - 1 for i in range(self.n_embd)]) * 0.5)
                self.spatial_first = nn.Parameter(torch.ones(self.n_embd) * math.log(0.3) + zigzag)
                x = torch.ones(1, 1, self.n_embd)
                for i in range(self.n_embd):
                    x[0, 0, i] = i / self.n_embd
                self.spatial_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
                self.spatial_mix_v = nn.Parameter(torch.pow(x, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
                self.spatial_mix_r = nn.Parameter(torch.pow(x, 0.5 * ratio_1_to_almost0))
        elif init_mode == 'local':
            self.spatial_decay = nn.Parameter(torch.ones(self.n_embd))
            self.spatial_first = nn.Parameter(torch.ones(self.n_embd))
            self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]))
            self.spatial_mix_v = nn.Parameter(torch.ones([1, 1, self.n_embd]))
            self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]))
        elif init_mode == 'global':
            self.spatial_decay = nn.Parameter(torch.zeros(self.n_embd))
            self.spatial_first = nn.Parameter(torch.zeros(self.n_embd))
            self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
            self.spatial_mix_v = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
            self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
        else:
            raise NotImplementedError
    def jit_func(self, x, patch_resolution):
        B, T, C = x.size()
        if self.shift_pixel > 0:
            xx = self.shift_func(x, self.shift_pixel, self.channel_gamma, patch_resolution)
            xk = x * self.spatial_mix_k + xx * (1 - self.spatial_mix_k)
            xv = x * self.spatial_mix_v + xx * (1 - self.spatial_mix_v)
            xr = x * self.spatial_mix_r + xx * (1 - self.spatial_mix_r)
        else:
            xk = x
            xv = x
            xr = x
        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)
        sr = torch.sigmoid(r)
        return sr, k, v
    def forward(self, x, patch_resolution=None):
        B, T, C = x.size()
        self.device = x.device
        sr, k, v = self.jit_func(x, patch_resolution)
        rwkv = RUN_CUDA(self.spatial_decay / T, self.spatial_first / T, k, v)
        if self.key_norm is not None:
            rwkv = self.key_norm(rwkv)
        rwkv = sr * rwkv
        rwkv = self.output(rwkv)
        return rwkv
class VRWKV_ChannelMix_3D(BaseModule):
    def __init__(self, n_embd, n_layer, layer_id, shift_mode='q_shift_3d',
                 channel_gamma=1 / 6, shift_pixel=1, hidden_rate=4, init_mode='fancy',
                 k_norm=True):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self._init_weights(init_mode)
        self.shift_pixel = shift_pixel
        self.shift_mode = shift_mode
        self.channel_gamma = channel_gamma
        if shift_pixel > 0:
            self.shift_func = q_shift_3d
        else:
            self.spatial_mix_k = None
            self.spatial_mix_r = None
        hidden_sz = hidden_rate * n_embd
        self.key = nn.Linear(n_embd, hidden_sz, bias=False)
        if k_norm:
            self.key_norm = nn.LayerNorm(hidden_sz)
        else:
            self.key_norm = None
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, n_embd, bias=False)
        self.value.scale_init = 0
        self.receptance.scale_init = 0
        self.key.scale_init = 1
    def _init_weights(self, init_mode):
        if init_mode == 'fancy':
            with torch.no_grad():
                ratio_1_to_almost0 = (1.0 - (self.layer_id / self.n_layer))
                x = torch.ones(1, 1, self.n_embd)
                for i in range(self.n_embd):
                    x[0, 0, i] = i / self.n_embd
                self.spatial_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
                self.spatial_mix_r = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
        elif init_mode == 'local':
            self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]))
            self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]))
        elif init_mode == 'global':
            self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
            self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
        else:
            raise NotImplementedError
    def forward(self, x, patch_resolution=None):
        if self.shift_pixel > 0:
            xx = self.shift_func(x, self.shift_pixel, self.channel_gamma, patch_resolution)
            xk = x * self.spatial_mix_k + xx * (1 - self.spatial_mix_k)
            xr = x * self.spatial_mix_r + xx * (1 - self.spatial_mix_r)
        else:
            xk = x
            xr = x
        k = self.key(xk)
        k = torch.square(torch.relu(k))
        if self.key_norm is not None:
            k = self.key_norm(k)
        kv = self.value(k)
        rkv = torch.sigmoid(self.receptance(xr)) * kv
        return rkv
class VRWKV_Block_3D(BaseModule):
    def __init__(self, n_embd, n_layer, layer_id,
                 shift_mode='q_shift_3d',
                 channel_gamma=1 / 6,
                 shift_pixel=1,
                 drop_path=0.,
                 hidden_rate=4,
                 init_mode='fancy',
                 init_values=None,
                 post_norm=False,
                 k_norm=True,
                 with_cp=False):
        super().__init__()
        self.layer_id = layer_id
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(n_embd)
        self.att = VRWKV_SpatialMix_3D(n_embd, n_layer, layer_id, shift_mode,
                                       channel_gamma, shift_pixel, init_mode,
                                       k_norm=k_norm)
        self.ffn = VRWKV_ChannelMix_3D(n_embd, n_layer, layer_id, shift_mode,
                                       channel_gamma, shift_pixel, hidden_rate,
                                       init_mode, k_norm=k_norm)
        self.layer_scale = (init_values is not None)
        self.post_norm = post_norm
        if self.layer_scale:
            self.gamma1 = nn.Parameter(init_values * torch.ones((n_embd)), requires_grad=True)
            self.gamma2 = nn.Parameter(init_values * torch.ones((n_embd)), requires_grad=True)
        self.with_cp = with_cp
    def forward(self, x, patch_resolution=None):
        B, T, C = x.shape
        D, H, W = patch_resolution
        def _inner_forward(x):
            if self.layer_id == 0:
                x = self.ln0(x)
            if self.post_norm:
                if self.layer_scale:
                    x = x + self.drop_path(self.gamma1 * self.ln1(self.att(x, patch_resolution)))
                    x = x + self.drop_path(self.gamma2 * self.ln2(self.ffn(x, patch_resolution)))
                else:
                    x = x + self.drop_path(self.ln1(self.att(x, patch_resolution)))
                    x = x + self.drop_path(self.ln2(self.ffn(x, patch_resolution)))
            else:
                if self.layer_scale:
                    x = x + self.drop_path(self.gamma1 * self.att(self.ln1(x), patch_resolution))
                    x = x + self.drop_path(self.gamma2 * self.ffn(self.ln2(x), patch_resolution))
                else:
                    x = x + self.drop_path(self.att(self.ln1(x), patch_resolution))
                    x = x + self.drop_path(self.ffn(self.ln2(x), patch_resolution))
            return x
        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x
class VRWKVLayer(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id,
                 channel_gamma=1 / 6,
                 shift_pixel=1,
                 drop_path=0.1,
                 hidden_rate=4,
                 init_mode='fancy',
                 init_values=None,
                 post_norm=False,
                 k_norm=True,
                 with_cp=False):
        super().__init__()
        self.dim = n_embd
        self.norm = nn.LayerNorm(n_embd)
        self.rwkv_block = VRWKV_Block_3D(
            n_embd=n_embd,
            n_layer=n_layer,
            layer_id=layer_id,
            channel_gamma=channel_gamma,
            shift_pixel= shift_pixel,
            drop_path=drop_path,
            hidden_rate=hidden_rate,
            init_mode=init_mode,
            init_values=init_values,
            post_norm=post_norm,
            k_norm=k_norm,
            with_cp=with_cp
        )
    def rwkv_forward(self, x):
        B, C = x.shape[:2]
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x = self.norm(x)
        patch_res = img_dims
        x = self.rwkv_block(x, patch_resolution=patch_res)
        x = x.transpose(-1, -2).reshape(B, C, *img_dims)
        return x
    def forward(self, x):
        B, C, D, H, W = x.shape
        x_skip = x
        out_x_1 = self.rwkv_forward(x)
        x_2 = rearrange(x, "b c d w h -> b c w d h")
        out_x_2 = self.rwkv_forward(x_2)
        out_x_2 = rearrange(out_x_2, "b c w d h -> b c d w h")
        x_3 = rearrange(x, "b c d w h -> b c h w d")
        out_x_3 = self.rwkv_forward(x_3)
        out_x_3 = rearrange(out_x_3, "b c h w d -> b c d w h")
        out = out_x_1 + out_x_2 + out_x_3
        out = out + x_skip
        return out
