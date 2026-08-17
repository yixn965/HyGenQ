import torch.nn as nn

from .build_model import MatMul
from .quant_modules import (
    QuantLinear_ar,
    QuantLinear_ar_outlier,
    QuantLinear_diff,
    QuantLinear_scaling,
    QuantMatMul,
)


def quant_model(model, input_quant_params={}, weight_quant_params={}):
    module_dict = {}

    for name, module in list(model.named_modules()):
        module_dict[name] = module
        separator = name.rfind(".")
        parent_name = name[:separator] if separator >= 0 else ""
        if parent_name not in module_dict:
            raise RuntimeError(f"Parent module {parent_name} not found")
        parent_module = module_dict[parent_name]
        attribute_name = name[separator + 1:] if separator >= 0 else name

        if isinstance(module, nn.Linear):
            if "input_proj" in name or "diffloss.net.final_layer.linear" in name:
                quantized_module = QuantLinear_scaling(
                    module.in_features,
                    module.out_features,
                    input_quant_params,
                    weight_quant_params,
                )
            elif ".0.mlp.fc2" in name:
                quantized_module = QuantLinear_ar_outlier(
                    module.in_features,
                    module.out_features,
                    input_quant_params,
                    weight_quant_params,
                )
            elif "diffloss" in name:
                quantized_module = QuantLinear_diff(
                    module.in_features,
                    module.out_features,
                    input_quant_params,
                    weight_quant_params,
                )
            else:
                quantized_module = QuantLinear_ar(
                    module.in_features,
                    module.out_features,
                    input_quant_params,
                    weight_quant_params,
                )
            quantized_module.weight.data = module.weight.data
            quantized_module.bias = module.bias
            setattr(parent_module, attribute_name, quantized_module)

        elif isinstance(module, MatMul) and "matmul" in name:
            setattr(parent_module, attribute_name, QuantMatMul(input_quant_params))

    return model


def set_quant_state(model, input_quant=False, weight_quant=False,
                    include_layers=None, exclude_layers=None):
    quantized_types = (
        QuantLinear_ar,
        QuantLinear_ar_outlier,
        QuantLinear_diff,
        QuantLinear_scaling,
        QuantMatMul,
    )
    for name, module in model.named_modules():
        if not isinstance(module, quantized_types):
            continue
        if include_layers and name not in include_layers:
            continue
        if exclude_layers and name in exclude_layers:
            continue
        module.set_quant_state(input_quant, weight_quant)
