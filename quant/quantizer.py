import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from util.logger_utils import logger

logger = logging.getLogger(__name__)


def _stash_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return value


def _use_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    return value

def lp_loss(pred, tgt, p=2.0, reduction='none'):
    """
    loss function measured in L_p Norm
    """
    if reduction == 'none':
        return (pred-tgt).abs().pow(p).sum(1).mean()
    else:
        return (pred-tgt).abs().pow(p).mean()


class UniformQuantizer_ar(nn.Module):
    """
    PyTorch module for asymmetric quantization with channel-wise quantization and dynamic initialization.
    """
    def __init__(self, n_bits: int = 8, channel_wise: bool = False, i=None):
        super(UniformQuantizer_ar, self).__init__()
        assert 2 <= n_bits <= 8, 'Unsupported bit width'
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits
        self.delta = None
        self.zero_point = None
        self.inited = False
        self.channel_wise = channel_wise
        self.i = None

        # Initialize the device
        # self.device = 'cuda'
        self.device = torch.device('cpu')
        # Initialize activation parameters

        self.other_activations_zs = torch.zeros(64, device=self.device)
        self.other_activations_ss = torch.zeros(64, device=self.device)
        self.other_weights_inits = torch.zeros(1, dtype=torch.bool, device=self.device)
        self.other_weights_zs = [torch.zeros(1, device=self.device) for _ in range(1)]
        self.other_weights_ss = [torch.zeros(1, device=self.device) for _ in range(1)]


    def forward(self, x: torch.Tensor, i=None, step=None, calib5=False, is_weight=False, adjustment = False, group_name=None):
        device = x.device  # Get the input tensor device

        if is_weight:  # Process weights
            index = step
            if calib5 or adjustment:
                self.delta, self.zero_point = self.init_quantization_scale(x, self.channel_wise,is_weight = is_weight)
                self.other_weights_ss = _stash_cpu(self.delta)
                self.other_weights_zs = _stash_cpu(self.zero_point)
            else:
                self.zero_point = self.other_weights_zs
                self.delta = self.other_weights_ss
        else:  # Process activations
            index = step
            if calib5:
                self.delta, self.zero_point = self.init_quantization_scale(x, self.channel_wise)
                self.other_activations_ss[index], self.other_activations_zs[index] = self.delta, self.zero_point
            else:
                self.zero_point = self.other_activations_zs[index]
                self.delta = self.other_activations_ss[index]
        self.zero_point = _use_device(self.zero_point, x.device)
        self.delta = _use_device(self.delta, x.device)
        if (isinstance(self.delta, torch.Tensor) and torch.all(self.delta == 0)) or (not isinstance(self.delta, torch.Tensor) and self.delta == 0):
            x_int = x + self.zero_point
        else:
            x_int = torch.round(x / self.delta) + self.zero_point
        x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        x_dequant = (x_quant - self.zero_point) * self.delta
        return x_dequant

# Original
    def init_quantization_scale(self, x: torch.Tensor, channel_wise: bool = False, is_weight=False):
        delta = torch.tensor(0.0)  # can be torch.tensor(0) for an integer zero or torch.tensor(0.0) for a floating-point zero
        zero_point = torch.tensor(0.0)
        if channel_wise:
            x_clone = x.clone().detach()
            n_channels = x_clone.shape[-1] if (len(x.shape) == 3 or len(x.shape) == 2) else x_clone.shape[0]
            if len(x.shape) == 4:
                x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
            elif len(x.shape) == 2:
                x_max = x_clone.abs().max(dim=0)[0]
            elif len(x.shape) == 3:
                x_max = x_clone.abs().max(dim=0)[0].max(dim=0)[0]
            else:
                raise NotImplementedError

            delta = x_max.clone()
            zero_point = x_max.clone()
            # determine the scale and zero point channel-by-channel
            for c in range(n_channels):
                if len(x.shape) == 3:
                    delta[c], zero_point[c] = self.init_quantization_scale(x_clone[:,:,c], channel_wise=False,is_weight = is_weight)
                else:
                    delta[c], zero_point[c] = self.init_quantization_scale(x_clone[:,c], channel_wise=False,is_weight = is_weight)
            if len(x.shape) == 4:
                delta = delta.view(-1, 1, 1, 1)
                zero_point = zero_point.view(-1, 1, 1, 1)
            elif len(x.shape) == 2:
                delta = delta.view(1, -1)
                zero_point = zero_point.view(1, -1)
            elif len(x.shape) == 3:
                delta = delta.view(1, 1, -1)
                zero_point = zero_point.view(1, 1, -1)
            else:
                raise NotImplementedError
        else:
            x_clone = x.clone().detach()
            x_max = x_clone.max()
            x_min = x_clone.min()
            delta = (x_max - x_min) / (2 ** self.n_bits - 1)  # Compute the quantization step size
            zero_point = (-x_min / delta).round()  # Compute the zero point
        return delta, zero_point



class UniformQuantizer_diff(nn.Module):
    """
    PyTorch module for asymmetric quantization with channel-wise quantization and dynamic initialization.
    """
    def __init__(self, n_bits: int = 8, channel_wise: bool = False, i=None):
        super(UniformQuantizer_diff, self).__init__()
        assert 2 <= n_bits <= 8, 'Unsupported bit width'
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits
        self.delta = None
        self.zero_point = None
        self.inited = False
        self.channel_wise = channel_wise
        self.i = None

        # Initialize the device
        # self.device = 'cuda'
        self.device = torch.device('cpu')
        # Initialize activation parameters
        self.diffloss_zs = torch.zeros(6400, device=self.device)
        self.diffloss_ss = torch.zeros(6400, device=self.device)
        self.diffloss_weights_zs = [torch.zeros(1, device=self.device) for _ in range(6400)]
        self.diffloss_weights_ss = [torch.zeros(1, device=self.device) for _ in range(6400)]

    def forward(self, x: torch.Tensor, i=None, step=None, calib5=False, is_weight=False, adjustment = False, group_name=None,threshold = None):
        self.i = i
        device = x.device  # Get the input tensor device

        if is_weight:  # Process weights
            if i is not None and i >= 0:  # diffloss weights
                index = step * 100 + i
                if calib5 or adjustment:
                        # Initialize
                        self.delta, self.zero_point = self.init_quantization_scale(x, self.channel_wise,is_weight = is_weight)
                        self.diffloss_weights_ss[index] = _stash_cpu(self.delta)
                        self.diffloss_weights_zs[index] = _stash_cpu(self.zero_point)
                else:
                    self.zero_point = self.diffloss_weights_zs[index]
                    self.delta = self.diffloss_weights_ss[index]
        else:  # Process activations
            if i is not None and i >= 0:  # diffloss activations
                index = step * 100 + i
                if calib5:
                        self.delta, self.zero_point = self.init_quantization_scale(x, self.channel_wise,threshold = threshold)
                        self.diffloss_ss[index], self.diffloss_zs[index] = self.delta, self.zero_point
                else:
                    self.zero_point = self.diffloss_zs[index]
                    self.delta = self.diffloss_ss[index]
        self.zero_point = _use_device(self.zero_point, x.device)
        self.delta = _use_device(self.delta, x.device)
        if (isinstance(self.delta, torch.Tensor) and torch.all(self.delta == 0)) or (not isinstance(self.delta, torch.Tensor) and self.delta == 0):
            x_int = x + self.zero_point
        else:
            x_int = torch.round(x / self.delta) + self.zero_point
        x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        x_dequant = (x_quant - self.zero_point) * self.delta
        return x_dequant

    def init_quantization_scale(self, x: torch.Tensor, channel_wise: bool = False, is_weight=False,threshold = None):
        delta = torch.tensor(0.0)  # can be torch.tensor(0) for an integer zero or torch.tensor(0.0) for a floating-point zero
        zero_point = torch.tensor(0.0)
        if channel_wise:
            x_clone = x.clone().detach()
            n_channels = x_clone.shape[-1] if (len(x.shape) == 3 or len(x.shape) == 2) else x_clone.shape[0]
            if len(x.shape) == 4:
                x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
            elif len(x.shape) == 2:
                x_max = x_clone.abs().max(dim=0)[0]
            elif len(x.shape) == 3:
                x_max = x_clone.abs().max(dim=0)[0].max(dim=0)[0]
            else:
                raise NotImplementedError

            delta = x_max.clone()
            zero_point = x_max.clone()
            # determine the scale and zero point channel-by-channel
            for c in range(n_channels):
                if len(x.shape) == 3:
                    delta[c], zero_point[c] = self.init_quantization_scale(x_clone[:,:,c], channel_wise=False,is_weight = is_weight)
                else:
                    delta[c], zero_point[c] = self.init_quantization_scale(x_clone[:,c], channel_wise=False,is_weight = is_weight)
            if len(x.shape) == 4:
                delta = delta.view(-1, 1, 1, 1)
                zero_point = zero_point.view(-1, 1, 1, 1)
            elif len(x.shape) == 2:
                delta = delta.view(1, -1)
                zero_point = zero_point.view(1, -1)
            elif len(x.shape) == 3:
                delta = delta.view(1, 1, -1)
                zero_point = zero_point.view(1, 1, -1)
            else:
                raise NotImplementedError
        else:
            x_clone = x.clone().detach()
            if threshold is not None:
                x_max = threshold
                x_min = x_clone.min()
            else:
                x_max = x_clone.max()
                x_min = x_clone.min()
            delta = (x_max - x_min) / (2 ** self.n_bits - 1)  # Compute the quantization step size
            zero_point = (-x_min / delta).round()  # Compute the zero point
        return delta, zero_point


class UniformQuantizer_group(nn.Module):
    """
    PyTorch module for asymmetric quantization with channel-wise quantization and dynamic initialization.
    """
    def __init__(self, n_bits: int = 8, channel_wise: bool = False, i=None):
        super(UniformQuantizer_group, self).__init__()
        assert 2 <= n_bits <= 8, 'Unsupported bit width'
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits
        self.delta = None
        self.zero_point = None
        # self.delta1 = None
        # self.zero_point1 = None
        self.inited = False
        self.channel_wise = channel_wise
        self.i = None

        # Initialize the device
        # self.device = 'cuda'
        self.device = torch.device('cpu')
        self.other_activations_zs = torch.zeros(64, device=self.device)
        self.other_activations_ss = torch.zeros(64, device=self.device)
        self.other_activations_zs_high = torch.zeros(64, device=self.device)
        self.other_activations_ss_high = torch.zeros(64, device=self.device)
        self.other_weights_zs = torch.zeros(64, device=self.device)
        self.other_weights_ss = torch.zeros(64, device=self.device)
        self.other_weights_zs_high = torch.zeros(64, device=self.device)
        self.other_weights_ss_high = torch.zeros(64, device=self.device)

    def forward(self, x: torch.Tensor, i=None, step=None, calib5=False, is_weight=False, adjustment = False,group_name=None,threshold = None):
        self.i = i
        device = x.device  # Get the input tensor device
        index = step
        if is_weight:  # Process weights
                if calib5:
                    self.delta, self.zero_point = self.init_quantization_scale(x, self.channel_wise,is_weight = is_weight)
                    self.other_weights_ss[index], self.other_weights_zs[index] = self.delta, self.zero_point
                else:
                    self.zero_point = self.other_weights_zs[index]
                    self.delta = self.other_weights_ss[index]
        else:  # Process activations
                if calib5:
                        if group_name=="high":
                            self.delta, self.zero_point = self.init_quantization_scale(x, self.channel_wise, is_weight,group_name,threshold)
                            self.other_activations_ss_high[index], self.other_activations_zs_high[index] = self.delta, self.zero_point
                        else:
                            self.delta, self.zero_point = self.init_quantization_scale(x, self.channel_wise,False,group_name,threshold)
                            self.other_activations_ss[index], self.other_activations_zs[index] = self.delta, self.zero_point
                else:
                    if group_name=="high":
                        self.zero_point = self.other_activations_zs_high[index]
                        self.delta = self.other_activations_ss_high[index]
                    else:
                        self.zero_point = self.other_activations_zs[index]
                        self.delta = self.other_activations_ss[index]

        self.zero_point = _use_device(self.zero_point, x.device)
        self.delta = _use_device(self.delta, x.device)
        if (isinstance(self.delta, torch.Tensor) and torch.all(self.delta == 0)) or (not isinstance(self.delta, torch.Tensor) and self.delta == 0):
            x_int = x + self.zero_point
        else:
            x_int = torch.round(x / self.delta) + self.zero_point
        x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        x_dequant = (x_quant - self.zero_point) * self.delta

        return x_dequant

# Original
    def init_quantization_scale(self, x: torch.Tensor, channel_wise: bool = False, is_weight=False, group_name=None,threshold = None):
        delta = torch.tensor(0.0)  # can be torch.tensor(0) for an integer zero or torch.tensor(0.0) for a floating-point zero
        zero_point = torch.tensor(0.0)
        if channel_wise:
            x_clone = x.clone().detach()
            n_channels = x_clone.shape[-1] if (len(x.shape) == 3 or len(x.shape) == 2) else x_clone.shape[0]
            if len(x.shape) == 4:
                x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
            elif len(x.shape) == 2:
                x_max = x_clone.abs().max(dim=0)[0]
            elif len(x.shape) == 3:
                x_max = x_clone.abs().max(dim=0)[0].max(dim=0)[0]
            else:
                raise NotImplementedError

            delta = x_max.clone()
            zero_point = x_max.clone()
            # determine the scale and zero point channel-by-channel
            for c in range(n_channels):
                if len(x.shape) == 3:
                    delta[c], zero_point[c] = self.init_quantization_scale(x_clone[:,:,c], channel_wise=False,is_weight = is_weight)
                else:
                    delta[c], zero_point[c] = self.init_quantization_scale(x_clone[:,c], channel_wise=False,is_weight = is_weight)
            if len(x.shape) == 4:
                delta = delta.view(-1, 1, 1, 1)
                zero_point = zero_point.view(-1, 1, 1, 1)
            elif len(x.shape) == 2:
                delta = delta.view(1, -1)
                zero_point = zero_point.view(1, -1)
            elif len(x.shape) == 3:
                delta = delta.view(1, 1, -1)
                zero_point = zero_point.view(1, 1, -1)
            else:
                raise NotImplementedError
        else:
            x_clone = x.clone().detach()
            if threshold is not None:
                if group_name == "low":
                    x_max = threshold
                    x_min = x_clone.min()
                else:
                    x_min = x_clone.min()
                    x_max = x_clone.max()
            else:
                x_min = x_clone.min()
                x_max = x_clone.max()
            delta = (x_max - x_min) / (2 ** self.n_bits - 1)
            zero_point = (- x_min / delta).round()

        return delta, zero_point

    def quantize(self, x, max, min):
        delta = (max - min) / (2 ** self.n_bits - 1)
        zero_point = (- min / delta).round()
        # we assume weight quantization is always signed
        x_int = torch.round(x / delta)
        x_quant = torch.clamp(x_int + zero_point, 0, self.n_levels - 1)
        x_float_q = (x_quant - zero_point) * delta
        # print(f"x_float_q: {x_float_q}")
        return x_float_q

class UniformQuantizer_group_diff(nn.Module):
    """
    PyTorch module for asymmetric quantization with channel-wise quantization and dynamic initialization.
    """
    def __init__(self, n_bits: int = 8, channel_wise: bool = False, i=None):
        super(UniformQuantizer_group_diff, self).__init__()
        assert 2 <= n_bits <= 8, 'Unsupported bit width'
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits
        self.delta = None
        self.zero_point = None
        self.inited = False
        self.channel_wise = channel_wise
        self.i = None

        self.device = torch.device('cpu')

        self.diffloss_zs = torch.zeros(6400, device=self.device)
        self.diffloss_ss = torch.zeros(6400, device=self.device)
        self.diffloss_zs_high = torch.zeros(6400, device=self.device)
        self.diffloss_ss_high = torch.zeros(6400, device=self.device)
        # Initialize weight parameters
        self.diffloss_weights_zs = torch.zeros(6400, device=self.device)
        self.diffloss_weights_ss = torch.zeros(6400, device=self.device)
        self.diffloss_weights_zs_high = torch.zeros(6400, device=self.device)
        self.diffloss_weights_ss_high = torch.zeros(6400, device=self.device)

    def forward(self, x: torch.Tensor, i=None, step=None, calib5=False, is_weight=False, adjustment = False,group_name=None,threshold = None):
        self.i = i
        device = x.device  # Get the input tensor device

        if is_weight:  # Process weights
            if i is not None and i >= 0:  # diffloss weights
                index = step * 100 + i
                if calib5:
                        # Initialize
                        if group_name=="high":
                            self.delta, self.zero_point = self.init_quantization_scale(x, self.channel_wise,is_weight = is_weight,group_name = group_name,threshold = threshold)
                            self.diffloss_weights_ss_high[index], self.diffloss_weights_zs_high[index] = self.delta, self.zero_point
                        else:
                            self.delta, self.zero_point = self.init_quantization_scale(x, self.channel_wise,is_weight = is_weight,group_name = group_name,threshold = threshold)
                            self.diffloss_weights_ss[index], self.diffloss_weights_zs[index] = self.delta, self.zero_point
                else:
                    if group_name=="high":
                        self.zero_point = self.diffloss_weights_zs_high[index]
                        self.delta = self.diffloss_weights_ss_high[index]
                    else:
                        self.zero_point = self.diffloss_weights_zs[index]
                        self.delta = self.diffloss_weights_ss[index]
            elif i == -1:  # other weights
                index = step
                # Process weights in other components
                if calib5:
                        if group_name=="high":
                            self.delta, self.zero_point = self.init_quantization_scale(x, self.channel_wise,is_weight = is_weight)
                            self.other_weights_ss_high[index], self.other_weights_zs_high[index] = self.delta, self.zero_point
                        else:
                            self.delta, self.zero_point = self.init_quantization_scale(x, self.channel_wise,is_weight = is_weight)
                            self.other_weights_ss[index], self.other_weights_zs[index] = self.delta, self.zero_point
                else:
                    if group_name=="high":
                        self.zero_point = self.other_weights_zs_high[index]
                        self.delta = self.other_weights_ss_high[index]
                    else:
                        self.zero_point = self.other_weights_zs[index]
                        self.delta = self.other_weights_ss[index]
        else:  # Process activations
            if i is not None and i >= 0:  # diffloss activations
                index = step * 100 + i
                if calib5:
                    if group_name=="high":
                        self.delta, self.zero_point = self.init_quantization_scale(x, self.channel_wise, is_weight,group_name,threshold)
                        self.diffloss_ss_high[index], self.diffloss_zs_high[index] = self.delta, self.zero_point
                    else:
                        self.delta, self.zero_point = self.init_quantization_scale(x, self.channel_wise, is_weight,group_name,threshold)
                        self.diffloss_ss[index], self.diffloss_zs[index] = self.delta, self.zero_point
                else:
                    if group_name=="high":
                        self.zero_point = self.diffloss_zs_high[index]
                        self.delta = self.diffloss_ss_high[index]
                    else:
                        self.zero_point = self.diffloss_zs[index]
                        self.delta = self.diffloss_ss[index]
            elif i == -1:  # other activations
                index = step
                # Process activations in other components
                if calib5:
                        if group_name=="high":
                            self.delta, self.zero_point = self.init_quantization_scale(x, self.channel_wise)
                            self.other_activations_ss_high[index], self.other_activations_zs_high[index] = self.delta, self.zero_point
                        else:
                            self.delta, self.zero_point = self.init_quantization_scale(x, self.channel_wise)
                            self.other_activations_ss[index], self.other_activations_zs[index] = self.delta, self.zero_point
                else:
                    if group_name=="high":
                        self.zero_point = self.other_activations_zs_high[index]
                        self.delta = self.other_activations_ss_high[index]
                    else:
                        self.zero_point = self.other_activations_zs[index]
                        self.delta = self.other_activations_ss[index]
        self.zero_point = _use_device(self.zero_point, x.device)
        self.delta = _use_device(self.delta, x.device)
        if (isinstance(self.delta, torch.Tensor) and torch.all(self.delta == 0)) or (not isinstance(self.delta, torch.Tensor) and self.delta == 0):
            x_int = x + self.zero_point
        else:
            x_int = torch.round(x / self.delta) + self.zero_point
        x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        x_dequant = (x_quant - self.zero_point) * self.delta
        return x_dequant

    def init_quantization_scale(self, x: torch.Tensor, channel_wise: bool = False, is_weight=False,group_name=None,threshold = None):
        delta = torch.tensor(0.0)  # can be torch.tensor(0) for an integer zero or torch.tensor(0.0) for a floating-point zero
        zero_point = torch.tensor(0.0)
        if channel_wise:
            x_clone = x.clone().detach()
            n_channels = x_clone.shape[-1] if (len(x.shape) == 3 or len(x.shape) == 2) else x_clone.shape[0]
            if len(x.shape) == 4:
                x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
            elif len(x.shape) == 2:
                x_max = x_clone.abs().max(dim=0)[0]
            elif len(x.shape) == 3:
                x_max = x_clone.abs().max(dim=0)[0].max(dim=0)[0]
            else:
                raise NotImplementedError

            delta = x_max.clone()
            zero_point = x_max.clone()
            # determine the scale and zero point channel-by-channel
            for c in range(n_channels):
                if len(x.shape) == 3:
                    delta[c], zero_point[c] = self.init_quantization_scale(x_clone[:,:,c], channel_wise=False,is_weight = is_weight)
                else:
                    delta[c], zero_point[c] = self.init_quantization_scale(x_clone[:,c], channel_wise=False,is_weight = is_weight)
            if len(x.shape) == 4:
                delta = delta.view(-1, 1, 1, 1)
                zero_point = zero_point.view(-1, 1, 1, 1)
            elif len(x.shape) == 2:
                delta = delta.view(1, -1)
                zero_point = zero_point.view(1, -1)
            elif len(x.shape) == 3:
                delta = delta.view(1, 1, -1)
                zero_point = zero_point.view(1, 1, -1)
            else:
                raise NotImplementedError
        else:
            x_clone = x.clone().detach()
            if threshold is not None:
                if group_name == "low":
                    x_max = threshold
                    x_min = x_clone.min()
                else:
                    x_min = x_clone.min()
                    x_max = x_clone.max()
            else:
                x_max = x_clone.max()
                x_min = x_clone.min()
            delta = (x_max - x_min) / (2 ** self.n_bits - 1)  # Compute the quantization step size
            zero_point = (-x_min / delta).round()  # Compute the zero point
            # # x_q = self.quantize(x_clone, x_max, x_min)

        return delta, zero_point

    def quantize(self, x, max, min):
        delta = (max - min) / (2 ** self.n_bits - 1)
        zero_point = (- min / delta).round()
        # we assume weight quantization is always signed
        x_int = torch.round(x / delta)
        x_quant = torch.clamp(x_int + zero_point, 0, self.n_levels - 1)
        x_float_q = (x_quant - zero_point) * delta
        # print(f"x_float_q: {x_float_q}")
        return x_float_q


class UniformQuantizer_group_scaling(nn.Module):
    """
    PyTorch module for asymmetric quantization with channel-wise quantization and dynamic initialization.
    """
    def __init__(self, n_bits: int = 8, channel_wise: bool = False, i=None):
        super(UniformQuantizer_group_scaling, self).__init__()
        assert 2 <= n_bits <= 8, 'Unsupported bit width'
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits
        self.delta = None
        self.zero_point = None
        self.inited = False
        self.channel_wise = channel_wise
        self.i = None

        # Initialize the device
        # self.device = 'cuda'
        self.device = torch.device('cpu')
        self.diffloss_zs = torch.full((6400,), 128, device=self.device)
        self.diffloss_ss = torch.full((6400,), 0.04347, device=self.device)
        self.scaling_or_not = torch.full((6400,), False, dtype=torch.bool, device=self.device)  # All values are F

    def forward(self, x: torch.Tensor, i=None, step=None, calib5=False, is_weight=False, adjustment = False, group_name=None,layer_name = None,scale_quant = None,shift_quant = None):
                self.i = i
                device = x.device  # Get the input tensor device
                index = step * 100 + i

                self.delta, self.zero_point = self.init_quantization_scale(x, self.channel_wise)
                self.zero_point = _use_device(self.zero_point, x.device)
                self.delta = _use_device(self.delta, x.device)
                x_int = torch.round(x  / self.delta) + self.zero_point
                x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
                x_dequant = (x_quant - self.zero_point) * self.delta

                return x_dequant

# Original
    def init_quantization_scale(self, x: torch.Tensor, channel_wise: bool = False, is_weight=False):
        delta = torch.tensor(0.0)  # can be torch.tensor(0) for an integer zero or torch.tensor(0.0) for a floating-point zero
        zero_point = torch.tensor(0.0)
        if channel_wise:
            x_clone = x.clone().detach()
            n_channels = x_clone.shape[-1] if (len(x.shape) == 3 or len(x.shape) == 2) else x_clone.shape[0]
            if len(x.shape) == 4:
                x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
            elif len(x.shape) == 2:
                x_max = x_clone.abs().max(dim=0)[0]
            elif len(x.shape) == 3:
                x_max = x_clone.abs().max(dim=0)[0].max(dim=0)[0]
            else:
                raise NotImplementedError

            delta = x_max.clone()
            zero_point = x_max.clone()
            # determine the scale and zero point channel-by-channel
            for c in range(n_channels):
                if len(x.shape) == 3:
                    delta[c], zero_point[c] = self.init_quantization_scale(x_clone[:,:,c], channel_wise=False,is_weight = is_weight)
                else:
                    delta[c], zero_point[c] = self.init_quantization_scale(x_clone[:,c], channel_wise=False,is_weight = is_weight)
            if len(x.shape) == 4:
                delta = delta.view(-1, 1, 1, 1)
                zero_point = zero_point.view(-1, 1, 1, 1)
            elif len(x.shape) == 2:
                delta = delta.view(1, -1)
                zero_point = zero_point.view(1, -1)
            elif len(x.shape) == 3:
                delta = delta.view(1, 1, -1)
                zero_point = zero_point.view(1, 1, -1)
            else:
                raise NotImplementedError
        else:
            x_clone = x.clone().detach()
            x_max = x_clone.max()
            x_min = x_clone.min()
            delta = (x_max - x_min) / (2 ** self.n_bits - 1)
            zero_point = (- x_min / delta).round()
        return delta, zero_point

