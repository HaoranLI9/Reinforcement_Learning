from turtle import width
from utils import *
from torch.nn import functional as F
from math import log, pi, exp
import numpy as np
from scipy import linalg as la
import torch.distributions.normal as normal
logabs = lambda x: torch.log(torch.abs(x))

class ActNorm(AffineFlowModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_dep_init_done = False
    
    def f(self, x):
        # first batch is used for init
        if not self.data_dep_init_done:
            assert self.s is not None and self.t is not None # for now
            self.s.data = (-torch.log(x.std(dim=0, keepdim=True))).detach()
            self.t.data = (-(x * torch.exp(self.s)).mean(dim=0, keepdim=True)).detach()
            self.data_dep_init_done = True
        return super().f(x)

    def invf(self, x):
        # first batch is used for init
        if not self.data_dep_init_done:
            assert self.s is not None and self.t is not None # for now
            self.s.data = (-torch.log(x.std(dim=0, keepdim=True))).detach()
            self.t.data = (-(x * torch.exp(self.s)).mean(dim=0, keepdim=True)).detach()
            self.data_dep_init_done = True
        return super().invf(x)

class AffineCoupling(nn.Module):
    def __init__(self, dim, parity, net_class=MLP, nh=24, scale=True, shift=True):
        super().__init__()
        self.dim = dim
        self.parity = parity
        self.s_cond = lambda x: x.new_zeros(x.size(0), self.dim // 2)
        self.t_cond = lambda x: x.new_zeros(x.size(0), self.dim // 2)
        if scale:
            self.s_cond = net_class(self.dim // 2, self.dim // 2, nh)
        if shift:
            self.t_cond = net_class(self.dim // 2, self.dim // 2, nh)
        
    def f(self, x):
        #x0, x1 = x[:,::2], x[:,1::2]
        x0, x1 = x[:,:x.size(1)//2], x[:,x.size(1)//2:] 
        if self.parity:
            x0, x1 = x1, x0
        s = self.s_cond(x0)
        t = self.t_cond(x0)
        z0 = x0 # untouched half
        z1 = torch.exp(s) * x1 + t # transform this half as a function of the other
        if self.parity:
            z0, z1 = z1, z0
        z = torch.cat([z0, z1], dim=1)
        log_det = torch.sum(s, dim=1)
        return z, log_det
    
    def invf(self, z):
        #z0, z1 = z[:,::2], z[:,1::2]
        z0, z1 = z[:,:z.size(1)//2], z[:,z.size(1)//2:]
        if self.parity:
            z0, z1 = z1, z0
        s = self.s_cond(z0)
        t = self.t_cond(z0)
        x0 = z0 # this was the same
        x1 = (z1 - t) * torch.exp(-s) # reverse the transform on this half
        if self.parity:
            x0, x1 = x1, x0
        x = torch.cat([x0, x1], dim=1)
        log_det = torch.sum(-s, dim=1)
        return x, log_det

class InvConv1dLU(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        weight = np.random.randn(in_channel, in_channel)
        q, _ = la.qr(weight)
        w_p, w_l, w_u = la.lu(q.astype(np.float32))
        w_s = np.diag(w_u)
        w_u = np.triu(w_u, 1)
        u_mask = np.triu(np.ones_like(w_u), 1)
        l_mask = u_mask.T

        w_p = torch.from_numpy(w_p)
        w_l = torch.from_numpy(w_l)
        w_s = torch.from_numpy(w_s)
        w_u = torch.from_numpy(w_u)

        self.register_buffer("w_p", w_p)
        self.register_buffer("u_mask", torch.from_numpy(u_mask))
        self.register_buffer("l_mask", torch.from_numpy(l_mask))
        self.register_buffer("s_sign", torch.sign(w_s))
        self.register_buffer("l_eye", torch.eye(l_mask.shape[0]))
        self.w_l = nn.Parameter(w_l)
        self.w_s = nn.Parameter(logabs(w_s))
        self.w_u = nn.Parameter(w_u)

    def f(self, input):
        #_, _, height, width = input.shape
        height = 1
        width = 1
        input = input.unsqueeze(2).unsqueeze(3)
        weight = self.calc_weight()
        out = F.conv1d(input.double(), weight.double())
        logdet = torch.tensor([height * width * torch.sum(self.w_s)])
        out = out.squeeze(-1).squeeze(-1)
        return out, logdet


    def calc_weight(self):
        weight = (
            self.w_p
            @ (self.w_l * self.l_mask + self.l_eye)
            @ ((self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(self.w_s)))
        )

        return weight.unsqueeze(2).unsqueeze(3)

    def invf(self, output):
        weight = self.calc_weight()
        height = 1
        width =1
        logdet = torch.tensor([height * width * torch.sum(-self.w_s)])
        inv_input = F.conv1d(output.unsqueeze(2).unsqueeze(3).double(), weight.squeeze().inverse().unsqueeze(2).unsqueeze(3).double())
        return inv_input.squeeze(-1).squeeze(-1), logdet

"""
if __name__ == '__main__':
  ActNorm(dim=10)
  nn.Sequential(ActNorm(dim=10), FlowModules(dim=10, *flow))
"""