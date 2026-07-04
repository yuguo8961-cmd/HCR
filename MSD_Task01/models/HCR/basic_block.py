import torch
import torch.nn as nn
import torch.nn.functional as F
from .RWKV import VRWKV_ChannelMix_3D, VRWKV_SpatialMix_3D

class CrossScaleRWKVChannelMix(nn.Module):
    def __init__(self, src_channels, tgt_channels, hidden_rate=2,
                 layer_id=0, n_layer=6, dropout_rate=0.0):
        super().__init__()

        self.src_proj = nn.Conv3d(src_channels, tgt_channels, kernel_size=1, bias=False)
        self.tgt_norm = nn.InstanceNorm3d(tgt_channels)

        self.channel_mix = VRWKV_ChannelMix_3D(
            n_embd=tgt_channels,
            n_layer=n_layer,
            layer_id=layer_id,
            channel_gamma=1 / 6,
            shift_pixel=1,
            hidden_rate=hidden_rate,
            init_mode='fancy',
            k_norm=True
        )

        self.out_norm = nn.InstanceNorm3d(tgt_channels)

    def forward(self, src_feat, tgt_feat):
        if src_feat.shape[2:] != tgt_feat.shape[2:]:
            src_feat = F.interpolate(
                src_feat,
                size=tgt_feat.shape[2:],
                mode='trilinear',
                align_corners=False
            )

        src_feat = self.src_proj(src_feat)

        x = tgt_feat + src_feat
        x = self.tgt_norm(x)

        B, C, D, H, W = x.shape


        x_seq = x.flatten(2).transpose(1, 2)

        mixed = self.channel_mix(x_seq, patch_resolution=(D, H, W))

        mixed = mixed.transpose(1, 2).reshape(B, C, D, H, W)
        mixed = self.out_norm(mixed)

        return mixed

class TriDirectionalSpatialMix3D(nn.Module):
    def __init__(self, channels, layer_id=0, n_layer=2):
        super().__init__()

        self.norm = nn.LayerNorm(channels)

        self.spatial_mix = VRWKV_SpatialMix_3D(
            n_embd=channels,
            n_layer=n_layer,
            layer_id=layer_id,
            channel_gamma=1 / 6,
            shift_pixel=1,
            init_mode='fancy',
            k_norm=True
        )
        self.out_norm = nn.InstanceNorm3d(channels)

    def _spatial_forward(self, x):
        B, C = x.shape[:2]
        spatial_size = x.shape[2:]

        x_seq = x.flatten(2).transpose(1, 2)  # [B, N, C]
        x_seq = self.norm(x_seq)

        y = self.spatial_mix(x_seq, patch_resolution=spatial_size)

        y = y.transpose(1, 2).reshape(B, C, *spatial_size)
        return y

    def forward(self, x):
        identity = x


        y1 = self._spatial_forward(x)

        x2 = x.permute(0, 1, 3, 2, 4).contiguous()
        y2 = self._spatial_forward(x2)
        y2 = y2.permute(0, 1, 3, 2, 4).contiguous()

        x3 = x.permute(0, 1, 4, 3, 2).contiguous()
        y3 = self._spatial_forward(x3)
        y3 = y3.permute(0, 1, 4, 3, 2).contiguous()

        out = identity + y1 + y2 + y3
        out = self.out_norm(out)

        return out

class LightweightLPBlock3D(nn.Module):
    def __init__(self, channels, dropout=0.2, kernel_size=(1, 5, 5)):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)

        padding = tuple(k // 2 for k in kernel_size)

        self.pw1 = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(channels),
            nn.ReLU(inplace=True)
        )

        self.dw = nn.Sequential(
            nn.Conv3d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=padding,
                groups=channels,
                bias=False
            ),
            nn.InstanceNorm3d(channels),
            nn.ReLU(inplace=True)
        )

        self.pw2 = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(channels)
        )

        self.dropout = nn.Dropout3d(dropout)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        x = self.pw1(x)

        x = x + self.dw(x)

        x = self.pw2(x)
        x = self.dropout(x)

        x = self.relu(x + identity)

        return x

class CrossLayerInteraction4(nn.Module):

    def __init__(self, dim1, dim2, dim3, dim4, reduction=8):

        super().__init__()

        self.alpha_4to3 = nn.Parameter(torch.tensor(0.3))
        self.alpha_3to2 = nn.Parameter(torch.tensor(0.25))
        self.alpha_2to1 = nn.Parameter(torch.tensor(0.2))
        self.alpha_1to2 = nn.Parameter(torch.tensor(0.15))
        self.alpha_2to3 = nn.Parameter(torch.tensor(0.15))
        self.alpha_3to4 = nn.Parameter(torch.tensor(0.1))


        self.guide_4to3 = CrossScaleRWKVChannelMix(dim4, dim3, hidden_rate=2, layer_id=0)
        self.guide_3to2 = CrossScaleRWKVChannelMix(dim3, dim2, hidden_rate=2, layer_id=1)
        self.guide_2to1 = CrossScaleRWKVChannelMix(dim2, dim1, hidden_rate=2, layer_id=2)

        self.refine_1to2 = CrossScaleRWKVChannelMix(dim1, dim2, hidden_rate=2, layer_id=3)
        self.refine_2to3 = CrossScaleRWKVChannelMix(dim2, dim3, hidden_rate=2, layer_id=4)
        self.refine_3to4 = CrossScaleRWKVChannelMix(dim3, dim4, hidden_rate=2, layer_id=5)

        self.norm1 = nn.InstanceNorm3d(dim1)
        self.norm2 = nn.InstanceNorm3d(dim2)
        self.norm3 = nn.InstanceNorm3d(dim3)
        self.norm4 = nn.InstanceNorm3d(dim4)

    def forward(self, out1, out2, out3, out4):
        alpha_4to3 = torch.sigmoid(self.alpha_4to3)
        mix_4to3 = self.guide_4to3(out4, out3)
        out3_guided = self.norm3(out3 + alpha_4to3 * mix_4to3)

        alpha_3to2 = torch.sigmoid(self.alpha_3to2)
        mix_3to2 = self.guide_3to2(out3_guided, out2)
        out2_guided = self.norm2(out2 + alpha_3to2 * mix_3to2)

        alpha_2to1 = torch.sigmoid(self.alpha_2to1)
        mix_2to1 = self.guide_2to1(out2_guided, out1)
        out1_guided = self.norm1(out1 + alpha_2to1 * mix_2to1)

        alpha_1to2 = torch.sigmoid(self.alpha_1to2)
        mix_1to2 = self.refine_1to2(out1_guided, out2_guided)
        out2_refined = self.norm2(out2_guided + alpha_1to2 * mix_1to2)

        alpha_2to3 = torch.sigmoid(self.alpha_2to3)
        mix_2to3 = self.refine_2to3(out2_refined, out3_guided)
        out3_refined = self.norm3(out3_guided + alpha_2to3 * mix_2to3)

        alpha_3to4 = torch.sigmoid(self.alpha_3to4)
        mix_3to4 = self.refine_3to4(out3_refined, out4)
        out4_refined = self.norm4(out4 + alpha_3to4 * mix_3to4)

        return out1_guided, out2_refined, out3_refined, out4_refined


class SimpleChannelAttention(nn.Module):


    def __init__(self, src_channels, tgt_channels, reduction=8):
        super().__init__()


        hidden_dim = max(tgt_channels // reduction, 4)


        self.avg_pool = nn.AdaptiveAvgPool3d(1)


        self.mlp = nn.Sequential(
            nn.Conv3d(src_channels, hidden_dim, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_dim, tgt_channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, src_feat, target_spatial_size=None):

        pooled = self.avg_pool(src_feat)

        weight = self.mlp(pooled)

        return weight


class _ConvINReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, p=0.2):

        super(_ConvINReLU3D, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding),
            nn.InstanceNorm3d(out_channels),
            nn.Dropout3d(p=p, inplace=True),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class _ConvIN3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(_ConvIN3D, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding),
            nn.InstanceNorm3d(out_channels)
        )

    def forward(self, x):
        return self.block(x)

class DepthwiseSeparableConv3d(nn.Module):


    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, bias=False):
        super().__init__()


        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(padding, int):
            padding = (padding, padding, padding)
        if isinstance(stride, int):
            stride = (stride, stride, stride)


        self.depthwise = nn.Conv3d(
            in_channels, in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False
        )

        self.pointwise = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=1,
            bias=bias
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class DSConvINReLU3D(nn.Module):


    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            DepthwiseSeparableConv3d(in_channels, out_channels,
                                     kernel_size, stride, padding),
            nn.InstanceNorm3d(out_channels),
            nn.Dropout3d(p=dropout, inplace=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class DSConvIN3D(nn.Module):


    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0):
        super().__init__()
        self.block = nn.Sequential(
            DepthwiseSeparableConv3d(in_channels, out_channels,
                                     kernel_size, stride, padding),
            nn.InstanceNorm3d(out_channels)
        )

    def forward(self, x):
        return self.block(x)



class Encoder(nn.Module):
    def __init__(self, in_chns, out_chns, k=1, p=0, dropout=0.2):
        super().__init__()

        self.conv1 = nn.Sequential(
            _ConvINReLU3D(in_channels=in_chns, out_channels=out_chns, kernel_size=k, padding=p, p=dropout),
            _ConvIN3D(in_channels=out_chns, out_channels=out_chns, kernel_size=k, padding=p),
        )

        if in_chns != out_chns:
            self.skip = nn.Conv3d(in_chns, out_chns, kernel_size=1, stride=1, padding=0)
        else:
            self.skip = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.skip(x)
        out = self.conv1(x)
        out = out + identity
        return self.relu(out)

class Encoder2(nn.Module):

    def __init__(self, in_chns, out_chns, k=3, p=1, dropout=0.2):
        super().__init__()

        self.conv1 = nn.Sequential(
            DSConvINReLU3D(in_chns, out_chns, k, padding=p, dropout=dropout),
            DSConvIN3D(out_chns, out_chns, k, padding=p),
        )
        self.conv2 = nn.Sequential(
            DSConvINReLU3D(out_chns, out_chns, k, padding=p, dropout=dropout),
            DSConvIN3D(out_chns, out_chns, k, padding=p),
        )

        if in_chns != out_chns:
            self.skip = nn.Conv3d(in_chns, out_chns, 1)
        else:
            self.skip = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.skip(x)
        out = self.conv1(x)
        return self.relu(out + identity)


class DownSample(nn.Module):


    def __init__(self, in_chns, out_chns,kernels,strides):
        super(DownSample, self).__init__()


        self.down = nn.Sequential(
                nn.InstanceNorm3d(in_chns),
                nn.Conv3d(in_chns, out_chns, kernel_size=kernels, stride=strides)
            )

    def forward(self, x):
        return self.down(x)


class Decoder(nn.Module):

    def __init__(self, in_chns, out_chns, dropout):
        super().__init__()
        self.conv1 = nn.Sequential(

            DSConvINReLU3D(in_chns, out_chns, (1, 3, 3), padding=(0, 1, 1), dropout=dropout),
            DSConvIN3D(out_chns, out_chns, (3, 1, 1), padding=(1, 0, 0)),
        )
        self.conv2 = nn.Conv3d(in_chns, out_chns, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x = self.conv2(x)
        return self.relu(x1 + x)

class LightweightUp_sum(nn.Module):

    def __init__(self, in_chns, out_chns, kernel, stride, dropout,
                 attention_block=None, halves=True):
        super().__init__()
        up_chns = in_chns // 2 if halves else in_chns


        self.up = nn.Sequential(

            nn.ConvTranspose3d(
                in_chns, in_chns,
                kernel_size=kernel,
                stride=stride,
                groups=in_chns,
                bias=False
            ),

            nn.Conv3d(in_chns, up_chns, 1),
            nn.InstanceNorm3d(up_chns)
        )

        self.attention = attention_block
        self.convs = Decoder(up_chns, out_chns, dropout)

    def forward(self, x1, x2):
        if x2 is not None:
            x_1 = self.up(x1)


            if x_1.shape[2:] != x2.shape[2:]:
                x_1 = F.interpolate(x_1, size=x2.shape[2:],
                                    mode='trilinear', align_corners=False)

            x = x_1 + x2

            if self.attention is not None:
                x, w = self.attention(x)
                x = self.convs(x)
                return x, w
            else:
                x = self.convs(x)
                return x
        else:
            x_1 = self.up(x1)
            x = self.convs(x_1)
            return x

class conv_layer(nn.Module):

    def __init__(self, dim, res_ratio, dropout_rate, input_size=(128, 128, 128)):
        super().__init__()

        self.is_anisotropic = []
        D, H, W = input_size


        is_aniso_0 = res_ratio >= 2 ** 0 + 2 ** (-1)
        self.is_anisotropic.append(is_aniso_0)

        if not is_aniso_0:
            self.encoder1 = Encoder(dim[0], dim[1], k=3, p=1, dropout=dropout_rate)
        else:
            self.encoder1 = Encoder(dim[0], dim[1], k=(1, 3, 3), p=(0, 1, 1), dropout=dropout_rate)


        is_aniso_1 = res_ratio >= 2 ** 1 + 2 ** 0
        self.is_anisotropic.append(is_aniso_1)

        if not is_aniso_0:
            self.pool1 = DownSample(dim[1], dim[1], kernels=2, strides=2)
        else:
            self.pool1 = DownSample(dim[1], dim[1], kernels=(1, 2, 2), strides=(1, 2, 2))


        self.attn1 = LightweightLPBlock3D(
            channels=dim[1],
            dropout=dropout_rate,
            kernel_size=3,
        )

        if not is_aniso_1:
            self.encoder2 = Encoder(dim[1], dim[2], k=3, p=1, dropout=dropout_rate)
        else:
            self.encoder2 = Encoder(dim[1], dim[2], k=(1, 3, 3), p=(0, 1, 1), dropout=dropout_rate)


        is_aniso_2 = res_ratio >= 2 ** 2 + 2 ** 1
        self.is_anisotropic.append(is_aniso_2)

        if not is_aniso_1:
            self.pool2 = DownSample(dim[2], dim[2], kernels=2, strides=2)
        else:
            self.pool2 = DownSample(dim[2], dim[2], kernels=(1, 2, 2), strides=(1, 2, 2))  # √


        self.attn2 = TriDirectionalSpatialMix3D(
            channels=dim[2],
            layer_id=0,
            n_layer=2
        )

        if not is_aniso_2:
            self.encoder3 = Encoder2(dim[2], dim[3], k=3, p=1, dropout=dropout_rate)
        else:
            self.encoder3 = Encoder2(dim[2], dim[3], k=(1, 3, 3), p=(0, 1, 1), dropout=dropout_rate)


        is_aniso_3 = res_ratio >= 2 ** 3 + 2 ** 2
        self.is_anisotropic.append(is_aniso_2)

        if not is_aniso_2:
            self.pool3 = DownSample(dim[3], dim[3], kernels=2, strides=2)
        else:
            self.pool3 = DownSample(dim[3], dim[3], kernels=(1, 2, 2), strides=(1, 2, 2))


        self.attn3 = TriDirectionalSpatialMix3D(
            channels=dim[3],
            layer_id=1,
            n_layer=2
        )

        if not is_aniso_3:
            self.encoder4 = Encoder2(dim[3], dim[4], k=3, p=1, dropout=dropout_rate)
        else:
            self.encoder4 = Encoder2(dim[3], dim[4], k=(1, 3, 3), p=(0, 1, 1), dropout=dropout_rate)

        self.interaction = CrossLayerInteraction4(
            dim1=dim[1],
            dim2=dim[2],
            dim3=dim[3],
            dim4=dim[4],
            reduction=8
        )

    def forward(self, x):
        out1 = self.encoder1(x)

        x2 = self.pool1(out1)
        x2 = self.attn1(x2)
        out2 = self.encoder2(x2)

        x3 = self.pool2(out2)
        x3 = self.attn2(x3)
        out3 = self.encoder3(x3)

        x4 = self.pool3(out3)
        x4 = self.attn3(x4)
        out4 = self.encoder4(x4)

        out1_enhanced, out2_enhanced, out3_enhanced, out4_enhanced = self.interaction(
            out1, out2, out3, out4
        )

        return [out1_enhanced, out2_enhanced, out3_enhanced, out4_enhanced]

class deconv_layer(nn.Module):
    def __init__(self, embed_dims, res_ratio, dropout_rate):
        super(deconv_layer, self).__init__()
        self.network = []
        for i in range(len(embed_dims) - 1, 0, -1):
            is_half = True if embed_dims[i] == 2 * embed_dims[i - 1] else False
            if i <= 3 and res_ratio >= 2 ** (i - 1) + 2 ** (i - 2):
                self.network.append(
                    LightweightUp_sum(in_chns=embed_dims[i], out_chns=embed_dims[i - 1], kernel=(2, 2, 1), stride=(2, 2, 1),
                           dropout=dropout_rate,
                           halves=is_half))
            else:
                self.network.append(
                    LightweightUp_sum(in_chns=embed_dims[i], out_chns=embed_dims[i - 1], kernel=2, stride=2, dropout=dropout_rate,
                           halves=is_half))
        self.network = nn.Sequential(*self.network)

    def forward(self, hidden_states, return_intermediate=False):

        if return_intermediate:
            outputs = []
            for i in range(len(self.network)):
                if i == 0:
                    x = self.network[i](hidden_states[0], hidden_states[1])
                else:
                    x = self.network[i](x, hidden_states[i + 1])
                outputs.append(x)
            return outputs
        else:
            for i in range(len(self.network)):
                if i == 0:
                    x = self.network[i](hidden_states[0], hidden_states[1])
                else:
                    x = self.network[i](x, hidden_states[i + 1])
            return x



