import torch
import torch.nn as nn
from torch.nn import functional as F
class GridAttentionGate3d(nn.Module):
    def __init__(self, F_l, F_g, F_int=None, mode="concatenation", sub_sample_factor=2):
        super(GridAttentionGate3d, self).__init__()
        if F_int is None:
            F_int = F_l // 2
            if F_int == 0:
                F_int = 1
        self.W = nn.Sequential(
            nn.Conv3d(in_channels=F_l, out_channels=F_l, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(F_l)
        )
        self.theta = nn.Conv3d(in_channels=F_l, out_channels=F_int, kernel_size=sub_sample_factor,
                               stride=sub_sample_factor, padding=0, bias=False)
        self.phi = nn.Conv3d(in_channels=F_g, out_channels=F_int, kernel_size=1, stride=1, padding=0, bias=True)
        self.psi = nn.Conv3d(in_channels=F_int, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)
        if mode == 'concatenation':
            self.operation_function = self._concatenation
        elif mode == 'concatenation_debug':
            self.operation_function = self._concatenation_debug
        elif mode == 'concatenation_residual':
            self.operation_function = self._concatenation_residual
        else:
            raise NotImplementedError('Unknown operation function！')
    def forward(self, x, g):
        output = self.operation_function(x, g)
        return output
    def _concatenation(self, x, g):
        input_size = x.size()
        bs = input_size[0]
        assert bs == g.size(0)
        theta_x = self.theta(x)
        theta_x_size = theta_x.size()
        phi_g = F.interpolate(self.phi(g), size=theta_x_size[2:], mode="trilinear", align_corners=True)
        f = F.relu(theta_x + phi_g, inplace=True)
        sigm_psi_f = torch.sigmoid(self.psi(f))
        sigm_psi_f = F.interpolate(sigm_psi_f, size=input_size[2:], mode="trilinear", align_corners=True)
        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)
        return W_y, sigm_psi_f
    def _concatenation_debug(self, x, g):
        input_size = x.size()
        bs = input_size[0]
        assert bs == g.size(0)
        theta_x = self.theta(x)
        theta_x_size = theta_x.size()
        phi_g = F.interpolate(self.phi(g), size=theta_x_size[2:], mode="trilinear", align_corners=True)
        f = F.softplus(theta_x + phi_g)
        sigm_psi_f = torch.sigmoid(self.psi(f))
        sigm_psi_f = F.interpolate(sigm_psi_f, size=input_size[2:], mode="trilinear", align_corners=True)
        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)
        return W_y, sigm_psi_f
    def _concatenation_residual(self, x, g):
        input_size = x.size()
        bs = input_size[0]
        assert bs == g.size(0)
        theta_x = self.theta(x)
        theta_x_size = theta_x.size()
        phi_g = F.interpolate(self.phi(g), size=theta_x_size[2:], mode="trilinear", align_corners=True)
        f = F.relu(theta_x + phi_g, inplace=True)
        f = self.psi(f).view(bs, 1, -1)
        softmax_psi_f = torch.softmax(f, dim=2).view(bs, 1, *theta_x_size[2:])
        softmax_psi_f = F.interpolate(softmax_psi_f, size=input_size[2:], mode="trilinear", align_corners=True)
        y = softmax_psi_f.expand_as(x) * x
        W_y = self.W(y)
        return W_y, softmax_psi_f
if __name__ == '__main__':
    model = GridAttentionGate3d(128, 256, 64, mode="concatenation", sub_sample_factor=2)
    x = torch.rand((4, 128, 80, 80, 48))
    g = torch.rand((4, 256, 40, 40, 24))
    W_y, sigm_psi_f = model(x, g)
    print(x.size())
    print(g.size())
    print(W_y.size())
    print(sigm_psi_f.size())
