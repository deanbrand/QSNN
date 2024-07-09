import torch
import torch.nn as nn


class Alpha(nn.Module):
    def __init__(self, threshold=0.75, T1=60, N=1, device=torch.device("cpu")):
        super(Alpha, self).__init__()

        self.mem = torch.zeros(N, dtype=torch.float32)
        self.threshold = threshold
        self.T1 = T1
        self.device = device

    def forward(self, theta, tau, x):
        theta = theta.to(self.device)
        tau = tau.to(self.device)
        x = x.to(self.device)

        spike_out, self.mem = AlphaSurrogate.apply(theta, tau, x, self.mem,
                                                   self.threshold, self.T1)
        return spike_out, self.mem

    def reset_mem(self):
        self.mem = torch.zeros_like(self.mem, device=self.device)
        return self.mem


class AlphaSurrogate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, theta, tau, x, mem, threshold, T1):
        ctx.save_for_backward(theta, tau, x, mem)
        ctx.threshold = threshold
        ctx.T1 = T1

        gamma = -2*torch.arcsin(torch.sqrt(mem * torch.exp(-tau/T1)))
        phi = torch.where(mem < threshold,
                          2 * torch.arcsin(torch.sqrt(mem)), 0.0)

        Theta = torch.sin((theta + phi) * x/2)**2
        Gamma = torch.sin((gamma + phi) * (1 - x)/2)**2
        mem = torch.where(mem < threshold, Gamma + Theta, 0.0)

        spike_out = torch.where(mem > threshold,
                                1.0, 0.0)

        mem = torch.where(mem > 1, 0.0, mem)

        return spike_out, mem

    @staticmethod
    def backward(ctx, grad_spike_out, grad_output):
        theta, tau, x, mem = ctx.saved_tensors
        T1 = ctx.T1
        grad_mem = 1 / (1 + (torch.pi * mem).pow_(2)) * grad_output

        sqrt_mem = torch.sqrt(mem)
        sqrt_mem_exp = torch.sqrt(mem * torch.exp(-tau/T1))

        dtheta = (x * torch.sin(x * (theta + 2 * torch.arcsin(sqrt_mem)))) / 2
        dtau = ((x - 1) * sqrt_mem_exp * torch.sin(2 * (x - 1) * (torch.arcsin(sqrt_mem) -
                torch.arcsin(sqrt_mem_exp)))) / (2 * T1 * torch.sqrt(1 - mem * torch.exp(-tau/T1)))

        grad_theta = (dtheta * grad_mem)
        grad_tau = (dtau * grad_mem)

        return grad_theta, grad_tau, grad_mem, None, None, None
