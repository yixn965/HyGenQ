from types import MethodType

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers.mlp import Mlp
from timm.models.vision_transformer import Attention, Block


def mlp_forward(self, x, i=None, calib5=False, step=None, d=False, num=None,
                block_name=None, num_bsz=-1, adjustment=False):
    x = self.fc1(
        x, i=i, calib5=calib5, step=step, d=d,
        layer_name=f"{block_name}_block{num}_mlp_fc1", num=num_bsz,
        num_bsz=num_bsz, adjustment=adjustment,
    )
    x = self.act(x)
    x = self.drop1(x)
    x = self.norm(x)
    x = self.fc2(
        x, i=i, calib5=calib5, step=step, d=d,
        layer_name=f"{block_name}_block{num}_mlp_fc2", num=num_bsz,
        num_bsz=num_bsz, params={"n_bits": 8, "channel_wise": False},
    )
    return self.drop2(x)


def block_forward(self, x: torch.Tensor, i=None, calib5=False, step=None,
                  d=False, num=None, block_name=None, num_bsz=None,
                  adjustment=False) -> torch.Tensor:
    attn_output = self.attn(
        self.norm1(x), i=i, calib5=calib5, step=step, d=d, num=num,
        num_bsz=num_bsz, block_name=block_name, adjustment=adjustment,
    )
    x = x + self.drop_path1(self.ls1(attn_output))
    mlp_output = self.mlp(
        self.norm2(x), i=i, calib5=calib5, step=step, d=d, num=num,
        block_name=block_name, num_bsz=num_bsz, adjustment=adjustment,
    )
    return x + self.drop_path2(self.ls2(mlp_output))


def linear_forward(self, input: torch.Tensor, i=None, step=None, calib5=False,
                   d=False, layer_name=None, params=None, adjustment=False,
                   num=None, a=False, num_bsz=-1) -> torch.Tensor:
    return F.linear(input, self.weight, self.bias)


def attention_forward(self, x: torch.Tensor, calib=False, i=None, calib5=False,
                      step=None, d=False, num=None, num_bsz=None,
                      block_name=None, adjustment=False):
    batch_size, token_count, channels = x.shape
    qkv = self.qkv(
        x, i=i, calib5=calib5, step=step, d=d,
        layer_name=f"{block_name}_block{num}_qkv", adjustment=adjustment,
    ).reshape(batch_size, token_count, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = self.q_norm(q), self.k_norm(k)
    attention = self.matmul1(
        q * self.scale, k.transpose(-2, -1), i=i, calib5=calib5,
        step=step, d=d, layer_name=f"{block_name}_block{num}_matmul1",
    )
    x = self.matmul2(
        self.attn_drop(attention.softmax(dim=-1)), v, i=i, calib5=calib5,
        step=step, d=d, layer_name=f"{block_name}_block{num}_matmul2",
    )
    x = x.transpose(1, 2).reshape(batch_size, token_count, channels)
    x = self.proj(
        x, i=i, calib5=calib5, step=step, d=d,
        layer_name=f"{block_name}_block{num}_Projection",
    )
    return self.proj_drop(x)


class MatMul(nn.Module):
    def forward(self, left, right, i=None, calib5=False, step=None, d=False,
                layer_name=None):
        return left @ right


def build_model(model):
    for module in model.modules():
        if isinstance(module, Attention):
            module.matmul1 = MatMul()
            module.matmul2 = MatMul()
            module.forward = MethodType(attention_forward, module)
        elif isinstance(module, nn.Linear):
            module.forward = MethodType(linear_forward, module)
        elif isinstance(module, Block):
            module.forward = MethodType(block_forward, module)
        elif isinstance(module, Mlp):
            module.forward = MethodType(mlp_forward, module)
    return model
