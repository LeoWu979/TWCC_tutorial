import math
import pdb

import numpy as np

import torch
import torch.nn as nn
#import torch.nn.functional as F
from torch.autograd import Function, Variable
from torch.autograd.function import Function, InplaceFunction
from torch.nn.parameter import Parameter


def _mean(p, dim):
    """Computes the mean over all dimensions except dim"""
    if dim is None:
        return p.mean()
    elif dim == 0:
        output_size = (p.size(0),) + (1,) * (p.dim() - 1)
        return p.contiguous().view(p.size(0), -1).mean(dim=1).view(*output_size)
    elif dim == p.dim() - 1:
        output_size = (1,) * (p.dim() - 1) + (p.size(-1),)
        return p.contiguous().view(-1, p.size(-1)).mean(dim=0).view(*output_size)
    else:
        return _mean(p.transpose(0, dim), 0).transpose(0, dim)


class UniformQuantize(InplaceFunction):

    @classmethod
    def forward(cls, ctx, input, num_bits=1, min_value=None, max_value=None,
                stochastic=False, inplace=False, enforce_true_zero=False, num_chunks=None, out_half=False):

        num_chunks = num_chunks = input.shape[
            0] if num_chunks is None else num_chunks
        if min_value is None or max_value is None:
            B = input.shape[0]
            y = input.view(B // num_chunks, -1)
        if min_value is None:
            min_value = y.min(-1)[0].mean(-1)  # C
            #min_value = float(input.view(input.size(0), -1).min(-1)[0].mean())
        if max_value is None:
            #max_value = float(input.view(input.size(0), -1).max(-1)[0].mean())
            max_value = y.max(-1)[0].mean(-1)  # C
        ctx.inplace = inplace
        ctx.num_bits = num_bits
        ctx.min_value = min_value
        ctx.max_value = max_value
        ctx.stochastic = stochastic

        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()

        qmin = 0.
        qmax = 2.**num_bits - 1.
        scale = (max_value - min_value) / (qmax - qmin)

        scale = max(scale, 1e-8)

        if enforce_true_zero:
            initial_zero_point = qmin - min_value / scale
            zero_point = 0.
            # make zero exactly represented
            if initial_zero_point < qmin:
                zero_point = qmin
            elif initial_zero_point > qmax:
                zero_point = qmax
            else:
                zero_point = initial_zero_point
            zero_point = int(zero_point)
            output.div_(scale).add_(zero_point)
        else:
            output.add_(-min_value).div_(scale).add_(qmin)

        if ctx.stochastic:
            noise = output.new(output.shape).uniform_(-0.5, 0.5)
            output.add_(noise)
        output.clamp_(qmin, qmax).round_()  # quantize

        if enforce_true_zero:
            output.add_(-zero_point).mul_(scale)  # dequantize
        else:
            output.add_(-qmin).mul_(scale).add_(min_value)  # dequantize
        if out_half and num_bits <= 16:
            output = output.half()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # straight-through estimator
        grad_input = grad_output
        return grad_input, None, None, None, None, None, None


class UniformQuantizeGrad(InplaceFunction):

    @classmethod
    def forward(cls, ctx, input, num_bits=1, min_value=None, max_value=None, stochastic=True, inplace=False):
        ctx.inplace = inplace
        ctx.num_bits = num_bits
        ctx.min_value = min_value
        ctx.max_value = max_value
        ctx.stochastic = stochastic
        return input

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.min_value is None:
            min_value = float(grad_output.min())
        else:
            min_value = ctx.min_value
        if ctx.max_value is None:
            max_value = float(grad_output.max())
        else:
            max_value = ctx.max_value
        grad_input = UniformQuantize().apply(grad_output, ctx.num_bits,
                                             min_value, max_value, ctx.stochastic, ctx.inplace)
        return grad_input, None, None, None, None, None


def conv2d_biprec(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, num_bits_grad=None):
    out1 = nn.conv2d(input.detach(), weight, bias,
                    stride, padding, dilation, groups)
    out2 = nn.conv2d(input, weight.detach(), bias.detach() if bias is not None else None,
                    stride, padding, dilation, groups)
    out2 = quantize_grad(out2, num_bits=num_bits_grad)
    return out1 + out2 - out1.detach()


def linear_biprec(input, weight, bias=None, num_bits_grad=None):
    out1 = nn.linear(input.detach(), weight, bias)
    out2 = nn.linear(input, weight.detach(), bias.detach()
                    if bias is not None else None)
    out2 = quantize_grad(out2, num_bits=num_bits_grad)
    return out1 + out2 - out1.detach()


def quantize(x, num_bits=1, min_value=None, max_value=None, num_chunks=None, stochastic=False, inplace=False):
    return UniformQuantize().apply(x, num_bits, min_value, max_value, num_chunks, stochastic, inplace)


def quantize_grad(x, num_bits=8, min_value=None, max_value=None, stochastic=True, inplace=False):
    return UniformQuantizeGrad().apply(x, num_bits, min_value, max_value, stochastic, inplace)


class QuantMeasure(nn.Module):
    """docstring for QuantMeasure."""

    def __init__(self, num_bits=1, momentum=0.1):
        super(QuantMeasure, self).__init__()
        self.register_buffer('running_min', torch.zeros(1))
        self.register_buffer('running_max', torch.zeros(1))
        self.momentum = momentum
        self.num_bits = num_bits

    def forward(self, input):
        if self.training:
            min_value = input.detach().view(
                input.size(0), -1).min(-1)[0].mean()
            max_value = input.detach().view(
                input.size(0), -1).max(-1)[0].mean()
            self.running_min.mul_(self.momentum).add_(
                min_value * (1 - self.momentum))
            self.running_max.mul_(self.momentum).add_(
                max_value * (1 - self.momentum))
        else:
            min_value = self.running_min
            max_value = self.running_max
        return quantize(input, self.num_bits, min_value=float(min_value), max_value=float(max_value), num_chunks=16)


def Binarize(tensor, quant_mode='det'):
    if quant_mode=='det':
        return tensor.sign()
    else:
        return tensor.add_(1).div_(2).add_(torch.rand(tensor.size()).add(-0.5)).clamp_(0,1).round().mul_(2).add_(-1)

#import torch.nn._functions as tnnf

class QuantLinear(nn.Linear):

    def __init__(self, *kargs, **kwargs):
        self.prog = kwargs.pop('progressive', None)
        self.layer = kwargs.pop('layer', None)
        self.bits = kwargs.pop('bits', None)
        self.fmap = kwargs.pop('fmap', 1)               # Input feature map flatten multiplier
        self.out_ch = kwargs.pop('out_ch', 1)
        self.in_ch = kwargs.pop('in_ch', 1)
        self.ch = kargs

        # Call init function
        super(QuantLinear, self).__init__(*kargs, **kwargs)
        self.quantizer = QuantMeasure(self.bits)

    def forward(self, input):
        
        # Quantize input
        if self.bits != 1:
            input.data = self.quantizer(input.data)
        else:
            input.data = Binarize(input.data)
        
        if not hasattr(self.weight,'org'):
            self.weight.org = self.weight.data.clone()

        # Quantize weight
        if self.bits != 1:
            self.weight.data = quantize(self.weight.org, num_bits=self.bits,
                                        min_value=float(self.weight.min()),
                                        max_value=float(self.weight.max()))
        else:
            self.weight.data = Binarize(self.weight.org)
            
        out = nn.functional.linear(input, self.weight)

        if self.bias is not None:
            out += self.bias.view(1, -1).expand_as(out)

        return out


class QuantConv2d(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        self.prog = kwargs.pop('progressive', None)
        self.layer = kwargs.pop('layer', None)
        self.ksize = kwargs.get('kernel_size', 3)
        self.bits = kwargs.pop('bits', None)
        self.ch = kargs
        
        # Call init function
        super(QuantConv2d, self).__init__(*kargs, **kwargs)   
        self.quantizer = QuantMeasure(self.bits)
        
    def forward(self, input):
        
        # Quantize input
        if self.bits != 1:
            input.data = self.quantizer(input.data)
        else:
            input.data = Binarize(input.data)
        
        if not hasattr(self.weight, 'org'):
            self.weight.org = self.weight.data.clone()
            
        # Quantize weight
        if self.bits != 1:
            self.weight.data = quantize(self.weight.org, num_bits=self.bits,
                                        min_value=float(self.weight.min()),
                                        max_value=float(self.weight.max()))
        else:
            self.weight.data = Binarize(self.weight.org)
        
        out = nn.functional.conv2d(input, self.weight, None, self.stride, self.padding, self.dilation, self.groups)

        if self.bias is not None:
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out
