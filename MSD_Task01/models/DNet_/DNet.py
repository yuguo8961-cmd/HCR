import torch
import torch.nn as nn
from .DNet_blocks import Encoder, Decoder
class DNet(nn.Module):
    def __init__(
        self,
        input_channels,
        n_stages,
        features_per_stage,
        conv_op,
        num_classes,
        deep_supervision,
        depths=[2, 2, 2, 2, 2],
        feat_size=[48, 96, 192, 384, 768],
        drop_path_rate=0            
    ) -> None:
        super().__init__()     
        self.in_channels = input_channels
        self.out_channels = num_classes
        self.deep_supervision=deep_supervision
        self.depths = depths
        self.drop_path_rate = drop_path_rate
        self.feat_size = feat_size
        self.encoder = Encoder(
            in_chans=self.in_channels,
            depths=self.depths,
            dims=self.feat_size,
            drop_path_rate=self.drop_path_rate
        )
        self.decoder = Decoder(
            out_channels=self.out_channels,
            depths=self.depths,
            dims=self.feat_size,
            drop_path_rate=self.drop_path_rate,
            deep_supervision=self.deep_supervision
        )
    def forward(self, x):
        skips, out = self.encoder(x)
        return self.decoder(skips, out)
if __name__ == "__main__":
    device = torch.device("cuda:1")
    model = DNet(
        input_channels=4,
        n_stages=5,
        features_per_stage=[32, 64, 128, 256, 512],
        conv_op=nn.Conv3d,
        num_classes=4,   
        deep_supervision=True,
        depths=[2, 2, 2, 2, 2],
        feat_size=[32, 64, 128, 256, 512],
        drop_path_rate=0.0
    )
    x = torch.randn(1, 4, 128, 128, 128)
    out = model(x)
    print("Output shape:", out.shape)
    from thop import profile
    flops, params = profile(model, inputs=(x,))
    print("FLOPs: {:.2f} M".format(flops / 1e6))
    print("Params: {:.2f} M".format(params / 1e6))
