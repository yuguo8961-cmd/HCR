import torch
import torch.nn as nn
import torch.nn.functional as F
from .basic_block import conv_layer, deconv_layer


class hcr(nn.Module):
    def __init__(self, res_ratio, in_channels, out_channels,
                 embed_dims=(20, 40, 80, 80, 160), dropout_rate=0.2,
                 deep_supervision=True):
        super().__init__()


        conv_dim = [in_channels, embed_dims[0], embed_dims[1], embed_dims[2],embed_dims[4]]

        self.deep_supervision = deep_supervision

        self.conv = conv_layer(
            dim=conv_dim,
            res_ratio=res_ratio,
            dropout_rate=dropout_rate
        )


        self.deconv = deconv_layer(
            embed_dims=(embed_dims[0], embed_dims[1], embed_dims[2], embed_dims[4]),
            res_ratio=res_ratio,
            dropout_rate=dropout_rate
        )


        self.final_conv = nn.Conv3d(embed_dims[0], out_channels, kernel_size=1)

        if self.deep_supervision:
            self.aux_head0 = nn.Conv3d(embed_dims[2], out_channels, 1)
            self.aux_head1 = nn.Conv3d(embed_dims[1], out_channels, 1)

    def forward(self, x):
        conv_hidden_states = self.conv(x)


        hidden_states = [
            conv_hidden_states[3],
            conv_hidden_states[2],
            conv_hidden_states[1],
            conv_hidden_states[0],
        ]


        if self.deep_supervision and self.training:
            decoder_outputs = self.deconv(hidden_states, return_intermediate=True)
            logits_main = self.final_conv(decoder_outputs[2])
            logits_aux1 = self.aux_head1(decoder_outputs[1])
            logits_aux2 = self.aux_head0(decoder_outputs[0])

            return logits_main, logits_aux1, logits_aux2

        else:
            u0 = self.deconv(hidden_states, return_intermediate=False)
            logits = self.final_conv(u0)
            return logits



