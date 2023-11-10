import math
from typing import Optional

import numpy as np
import torch

from gpytorch.kernels.kernel import Kernel
from linear_operator import operators, to_linear_operator
from scipy.special import kv, kvp, loggamma


class Bessel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, nu):
        inp = inp.detach()
        ctx.nu = nu
        ctx.save_for_backward(inp)
        out = kv(nu,inp.numpy())
        return (torch.as_tensor(out, dtype=inp.dtype))

    @staticmethod
    def backward(ctx, grad_out):
        grad_out = grad_out.detach()
        nu = ctx.nu
        inp, = ctx.saved_tensors
        grad_in = grad_out.numpy() * kvp(nu,inp.numpy())
        return (torch.as_tensor(grad_in, dtype=inp.dtype), None)

class MaternKernel(Kernel):

    has_lengthscale = True

    def __init__(self, nu: Optional[float] = 2.01, **kwargs):
        if (nu < 0):
            raise RuntimeError("nu cannot be negative")
        self.nu = nu
        super(MaternKernel, self).__init__(**kwargs)

    def forward(self, x1:torch.Tensor, x2:torch.Tensor, diag=False, **params):
        x1_ = x1.reshape(-1,1).div(self.lengthscale)
        x2_ = x2.reshape(-1,1).div(self.lengthscale)
        r_ = math.sqrt(2.0*self.nu) * self.covar_dist(x1_, x2_, diag=diag, **params)
        r_ = r_.clamp(1e-15)
        C_ = torch.pow(r_,self.nu) * Bessel.apply(r_,self.nu)
        C_ = C_ * (math.pow(2.0,1-self.nu) * math.exp(-loggamma(self.nu)))
        return (C_)

    def dCdx1(self, x1:torch.Tensor, x2:torch.Tensor) -> operators.DenseLinearOperator:
        x1 = x1.squeeze()
        x2 = x2.squeeze()
        if ((x1.ndimension()!=1) or (x2.ndimension()!=1)):
            # only support 1D data for now
            raise RuntimeError("Kernel gradient only support 1D input")
        x1_ = x1.div(self.lengthscale)
        x2_ = x2.div(self.lengthscale)
        rd_ = x1_.reshape(-1,1) - x2_.reshape(1,-1)
        r_ = math.sqrt(2.0*self.nu) * rd_.abs()
        r_ = r_.clamp(1e-15)
        dC_ = -r_ * torch.pow(r_,self.nu-1) * Bessel.apply(r_,self.nu-1)
        dC_ = dC_ * (math.pow(2.0,1-self.nu) * math.exp(-loggamma(self.nu)))
        dC_ = dC_ * (math.sqrt(2.0*self.nu) / self.lengthscale)
        dC_ = dC_ * torch.sign(rd_)
        return (to_linear_operator(dC_))

    def dCdx2(self, x1:torch.Tensor, x2:torch.Tensor) -> operators.DenseLinearOperator:
        return (self.dCdx1(x1,x2).t())

    def d2Cdx1dx2(self, x1:torch.Tensor, x2:torch.Tensor) -> operators.DenseLinearOperator:
        x1 = x1.squeeze()
        x2 = x2.squeeze()
        if ((x1.ndimension()!=1) or (x2.ndimension()!=1)):
            # only support 1D data for now
            raise RuntimeError("Kernel gradient only support 1D input")
        x1_ = x1.div(self.lengthscale)
        x2_ = x2.div(self.lengthscale)
        rd_ = x1_.reshape(-1,1) - x2_.reshape(1,-1)
        r_ = math.sqrt(2.0*self.nu) * rd_.abs()
        r_ = r_.clamp(1e-15)
        d2C_ = -torch.pow(r_,self.nu-1) * Bessel.apply(r_,self.nu-1)
        d2C_ = d2C_ + torch.pow(r_,2.0) * torch.pow(r_,self.nu-2) * Bessel.apply(r_,self.nu-2)
        d2C_ = d2C_ * (math.pow(2.0,1-self.nu) * math.exp(-loggamma(self.nu)))
        d2C_ = d2C_ * (2.0 * self.nu / torch.square(self.lengthscale))
        d2C_ = -d2C_
        return (to_linear_operator(d2C_))