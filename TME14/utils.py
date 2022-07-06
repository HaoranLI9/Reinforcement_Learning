from torch.utils.data import DataLoader
import torch.nn as nn
import enum
from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")
import seaborn as sns

sns.set_theme()


def toDataFrame(t: torch.Tensor, origin: str):
    t = t.cpu().detach().numpy()
    df = pd.DataFrame(data=t, columns=(f"x{ix}" for ix in range(t.shape[1])))
    df['ix'] = df.index * 1.
    df["origin"] = origin
    return df



def scatterplots(samples: List[Tuple[str, torch.Tensor]], col_wrap=4):
    """Draw the 

    Args:
        samples (List[Tuple[str, torch.Tensor]]): The list of samples with their types
        col_wrap (int, optional): Number of columns in the graph. Defaults to 4.

    Raises:
        NotImplementedError: If the dimension of the data is not supported
    """
    # Convert data into pandas dataframes
    _, dim = samples[0][1].shape
    samples = [toDataFrame(sample, name) for name, sample in samples]
    data = pd.concat(samples, ignore_index=True)

    g = sns.FacetGrid(data, height=2, col_wrap=col_wrap, col="origin", sharex=False, sharey=False)
    g.set_titles(col_template="{col_name}", row_template="{row_name}")

    if dim == 1:
        g.map(sns.kdeplot, "distribution")
        plt.show()
    elif dim == 2:
        g.map(sns.scatterplot, "x0", "x1", alpha=0.6)
        plt.show()
    else:
        raise NotImplementedError()


def iter_data(dataset, bs):
    """Infinite iterator"""
    while True:
        loader = DataLoader(dataset, batch_size=bs, shuffle=True)
        yield from iter(loader)


class MLP(nn.Module):
    """Réseau simple 4 couches"""
    def __init__(self, nin, nout, nh):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nin, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nout),
        )
    def forward(self, x):
        x = x.float()
        return self.net(x)


# --- Modules de base

class AffineFlowModule(nn.Module):
    def __init__(self, dim, scale=True, shift=True):
        super().__init__()
        self.s = nn.Parameter(torch.randn(1, dim, requires_grad=True)) if scale else None
        self.t = nn.Parameter(torch.randn(1, dim, requires_grad=True)) if shift else None

    def invf(self, y) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns f^-1(x) and log |det J_f^-1(x)|"""
        s = self.s if self.s is not None else y.new_zeros(y.size())
        t = self.t if self.t is not None else y.new_zeros(y.size())
        x = (y - t) * torch.exp(-s)
        log_det = torch.sum(-s, dim=1)
        return x, log_det
        
    def f(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns f(x) and log |det J_f(x)|"""
        s = self.s if self.s is not None else x.new_zeros(x.size())
        t = self.t if self.t is not None else x.new_zeros(x.size())
        y = x * torch.exp(s) + t
        log_det = torch.sum(s, dim=1)
        return y, log_det

class FlowModules(AffineFlowModule):
    """A container for a succession of flow modules"""
    def __init__(self, dim, *flows: AffineFlowModule):
        super().__init__(dim)
        self.flows = nn.ModuleList(flows)

    def apply(self, modules_iter, caller, x):
        #print(type(x), x.shape, len())
        m, _ = x.shape
        logdet = torch.zeros(m, device=x.device)
        zs = [x]
        for module in modules_iter:
            x, _logdet = caller(module, x)
            zs.append(x)
            logdet += _logdet
        return zs, logdet            

    def modulenames(self, backward=False):
        return [f"L{ix} {module.__class__.__name__}" for ix, module in enumerate(reversed(self.flows) if backward else self.flows)]

    def f(self, x):
        zs, logdet = self.apply(self.flows, lambda m, x: m.f(x), x)
        return zs[-1], logdet[-1]

    def invf(self, y):
        zs, logdet = self.apply(reversed(self.flows), lambda m, y: m.invf(y), y)
        return zs[-1], logdet[-1]


class FlowModel(FlowModules):
    """Flow model = prior + flow modules"""
    def __init__(self, prior, dim, *flows: AffineFlowModule):
        super().__init__(dim,*flows)
        self.prior = prior

    def invf(self, x):
        # Just computes the prior
        zs, logdet = super().invf(x)
        logprob = self.prior.log_prob(zs[-1])
        return logprob, zs, logdet

    def f(self, x):
        zs, logdet = super().f(x)
        logprob = self.prior.log_prob(zs)
        return logprob, zs, logdet

