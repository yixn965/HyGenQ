import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import math
from diffusion import create_diffusion

class DiffLoss(nn.Module):
    """Diffusion Loss"""
    def __init__(self, target_channels, z_channels, depth, width, num_sampling_steps, grad_checkpointing=False):
        super(DiffLoss, self).__init__()
        self.in_channels = target_channels
        self.net = SimpleMLPAdaLN(
            in_channels=target_channels,
            model_channels=width,
            out_channels=target_channels * 2,  # for vlb loss
            z_channels=z_channels,
            num_res_blocks=depth,
            grad_checkpointing=grad_checkpointing
        )

        self.train_diffusion = create_diffusion(timestep_respacing="", noise_schedule="cosine")
        self.gen_diffusion = create_diffusion(timestep_respacing=num_sampling_steps, noise_schedule="cosine")

    def forward(self, target, z, mask=None):
        t = torch.randint(0, self.train_diffusion.num_timesteps, (target.shape[0],), device=target.device)
        model_kwargs = dict(c=z)
        loss_dict = self.train_diffusion.training_losses(self.net, target, t, model_kwargs)
        loss = loss_dict["loss"]
        if mask is not None:
            loss = (loss * mask).sum() / mask.sum()
        return loss.mean()

    def sample(self, z, temperature=1.0, cfg=1.0,calib1 = False,calib2 = False,timesteps_fp = [], a = None,calib3=False, step = None,calib5 = False,num_bsz = None, adjustment = False,scaling_or_not_1 = None, scaling_or_not_2 = None, diffusion_callback=None):
        # diffusion loss sampling
        if not cfg == 1.0:
            noise = torch.randn(z.shape[0] // 2, self.in_channels).cuda()
            noise = torch.cat([noise, noise], dim=0)
            model_kwargs = dict(c=z, cfg_scale=cfg)
            sample_fn = self.net.forward_with_cfg
        else:
            noise = torch.randn(z.shape[0], self.in_channels).cuda()
            model_kwargs = dict(c=z)
            sample_fn = self.net.forward
        if calib1:
            sampled_token_latent,activations_timestep = self.gen_diffusion.p_sample_loop(
                sample_fn, noise.shape, noise, clip_denoised=False, model_kwargs=model_kwargs, progress=False,
                temperature=temperature,calib1 = calib1, sample_callback=diffusion_callback
            )
            return sampled_token_latent,activations_timestep
        else:
            sampled_token_latent = self.gen_diffusion.p_sample_loop(
                sample_fn, noise.shape, noise, clip_denoised=False, model_kwargs=model_kwargs, progress=False,
                temperature=temperature, timesteps_fp = timesteps_fp, calib2 = calib2,a=a,calib3=calib3,
                step = step, calib5 = calib5,num_bsz = num_bsz, adjustment = adjustment,scaling_or_not_1 = scaling_or_not_1, scaling_or_not_2 = scaling_or_not_2,
                sample_callback=diffusion_callback,
            )
            return sampled_token_latent


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t,i = None,step = None,calib5 = False):
        # Convert timestep t to a vector
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        # t_emb = self.mlp(t_freq)
        # Use the MLP to project the vector to the required dimension

        # Pass through the first Linear layer
        x = self.mlp[0](t_freq,i = i,step =step,calib5 = calib5, layer_name = "TimestepEmbedder_mlp[0]")
        # x = self.mlp[0](t_freq)

        x = self.mlp[1](x)

        # Pass through the second Linear layer
        x = self.mlp[2](x,i = i,step =step,calib5 = calib5, layer_name = "TimestepEmbedder_mlp[2]")
        # x = self.mlp[2](x)
        # Final output
        t_emb = x
        return t_emb


class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    """

    def __init__(
        self,
        channels
    ):
        super().__init__()
        self.channels = channels

        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 3 * channels, bias=True)
        )

    def forward(self, x, y ,i =None, step =None, calib5 = False, d = False,num = None):
        # shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
        # 1. Apply the SiLU activation function
        # y contains embedded timestep t and z information
        y_activated = self.adaLN_modulation[0](y)  # after SiLU activation

        # 2. Apply the linear layer nn.Linear(channels, 3 * channels, bias=True)
        y_modulated = self.adaLN_modulation[1](y_activated,i = i,step =step,calib5 = calib5,d = d, layer_name = f"res_block{num}_adaLN_modulation[1]")  # After the linear transformation, the output dimension is 3 * channels
        # y_modulated = self.adaLN_modulation[1](y_activated)

        # 3. Split the output into shift_mlp, scale_mlp, and gate_mlp
        shift_mlp, scale_mlp, gate_mlp = y_modulated.chunk(3, dim=-1)  # Split into three parts along the last dimension


        normalized_x = self.in_ln(x)
        h = modulate(normalized_x, shift_mlp, scale_mlp)
        # h = self.mlp(h)
        h = self.mlp[0](h,i = i,step =step,calib5 = calib5,d = d, layer_name = f"res_block{num}_mlp[0]")  # Apply the first nn.Linear(channels, channels, bias=True)
        # h = self.mlp[0](h)
        h = self.mlp[1](h)  # Apply the SiLU activation function

        if num == 11:
            specific_params = {'n_bits': 8, 'channel_wise': True}
        else:
            specific_params = {'n_bits': 8, 'channel_wise': False}
        h = self.mlp[2](h,i = i,step =step,calib5 = calib5,d = d, layer_name = f"res_block{num}_mlp[2]", params=specific_params)  # Apply the second nn.Linear(channels, channels, bias=True)
        # h = self.mlp[2](h)
        m = gate_mlp * h
        return x + m


class FinalLayer(nn.Module):
    """
    The final layer adopted from DiT.
    """
    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_channels, 2 * model_channels, bias=True)
        )

    def forward(self, x, c,i = None,step = None,calib5 = False,d = False,adjustment = False,scale_quant_final = None,shift_quant_final = None,scaling_or_not_1 = None, scaling_or_not_2 = None):
        # shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        # 1. Apply the SiLU activation function
        c_activated = self.adaLN_modulation[0](c)  # after SiLU activation

        # 2. Apply the linear layer nn.Linear(model_channels, 2 * model_channels, bias=True)
        specific_params = {'n_bits': 8, 'channel_wise': True}
        # , params=specific_params
        c_modulated = self.adaLN_modulation[1](c_activated,i = i,step =step,calib5 = calib5, layer_name = "final_adaLN_modulation[1]", params=specific_params)  # After the linear transformation, the output dimension is 2 * model_channels
        # c_modulated = self.adaLN_modulation[1](c_activated)
        # print(c_modulated.shape)

        # 3. Split the output into shift and scale
        shift, scale = c_modulated.chunk(2, dim=-1)  # Split into two parts along the last dimension

        normalized_x = self.norm_final(x)

        # if scale_quant_final is not None:
        #     x = modulate(normalized_x, shift, scale) / scale_quant_final + shift_quant_final
        # else:
        x = modulate(normalized_x, shift, scale)    # Condition x features according to c_modulated
        if i>=0:
            min_val, max_val = torch.aminmax(x)
            scale_quant_final = (max_val - min_val) / 11.09
            shift_quant_final = - (min_val / scale_quant_final ) - 5.545
        # x = x/ scale_quant_final + shift_quant_final

        specific_params = {'n_bits': 8, 'channel_wise': True}
        x = self.linear(x,i = i,step =step,calib5 = calib5, layer_name = "final_linear", params=specific_params, adjustment = adjustment,scale_quant = scale_quant_final, shift_quant = shift_quant_final)
        # x = self.linear(x)
        return x


class SimpleMLPAdaLN(nn.Module):
    """
    The MLP for Diffusion Loss.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param z_channels: channels in the condition.
    :param num_res_blocks: number of residual blocks per downsample.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        z_channels,
        num_res_blocks,
        grad_checkpointing=False
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.grad_checkpointing = grad_checkpointing

        self.time_embed = TimestepEmbedder(model_channels)
        self.cond_embed = nn.Linear(z_channels, model_channels)

        self.input_proj = nn.Linear(in_channels, model_channels)

        res_blocks = []
        for i in range(num_res_blocks):
            res_blocks.append(ResBlock(
                model_channels,
            ))

        self.res_blocks = nn.ModuleList(res_blocks)
        self.final_layer = FinalLayer(model_channels, out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers
        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    # def forward(self, x, t, c):
    #     """
    #     Apply the model to an input batch.
    #     :param x: an [N x C] Tensor of inputs.
    #     :param t: a 1-D batch of timesteps.
    #     :param c: conditioning from AR transformer.
    #     :return: an [N x C] Tensor of outputs.
    #     """
    #     x = self.input_proj(x)
    #     t = self.time_embed(t)
    #     c = self.cond_embed(c)

    #     y = t + c

    #     if self.grad_checkpointing and not torch.jit.is_scripting():
    #         for block in self.res_blocks:
    #             x = checkpoint(block, x, y)
    #     else:
    #         for block in self.res_blocks:
    #             x = block(x, y)

    #     return self.final_layer(x, y)

    def forward(self, x, t, c,calib1=False, timesteps_fp = [], calib2 = False,a = None,i = None,step = None,calib5 = False,d = False,num_bsz = None,adjustment = False,scale_quant = None,shift_quant = None,scaling_or_not_1 = None, scaling_or_not_2 = None):
        """
        Apply the model to an input batch.
        :param x: an [N x C] Tensor of inputs.
        :param t: a 1-D batch of timesteps.
        :param c: conditioning from AR transformer.
        :return: an [N x C] Tensor of outputs.
        """
        # Apply input projection
        if calib1:
            activations = x.clone()
        # if calib2:
        #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #     x = timesteps_fp[a].to(device)
        #     a+=1
        # At the first timestep, x is a pure noise image; subsequent timesteps receive the previous img output
        specific_params = {'n_bits': 8, 'channel_wise': True}
        x = self.input_proj(x,i = i,step = step,calib5 = calib5, layer_name = "input_proj", adjustment = adjustment, params=specific_params,num_bsz = num_bsz,scale_quant = scale_quant,shift_quant = shift_quant)
        # x = self.input_proj(x)

        # Apply time embedding
        t = self.time_embed(t,i = i,step = step,calib5 = calib5)
        # t = self.time_embed(t)

        # Apply conditioning embedding

        # c is z processed earlier by the autoregressive component
        c = self.cond_embed(c,i = i,step = step,calib5 = calib5, layer_name = "cond_embed")
        # c = self.cond_embed(c)

        # Combine time and conditioning embeddings
        y = t + c

        # Apply residual blocks
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.res_blocks:
                x = checkpoint(block, x, y)
        else:
            p = 0
            for block in self.res_blocks:
                x = block(x, y,i = i,step = step,calib5 = calib5,num = p)
                p+=1
                # x = block(x, y)

        # Apply final layer
        output = self.final_layer(x, y,i = i,step = step,calib5 = calib5,adjustment = adjustment,scaling_or_not_1 = scaling_or_not_1, scaling_or_not_2 = scaling_or_not_2)
        # output = self.final_layer(x,y)
        if calib1:
            return output,activations
        else:
            return output

    def forward_with_cfg(self, x, t, c, cfg_scale,calib1=False, timesteps_fp = [], calib2 = None,a = None,i = None,step = None,calib5 = False,num_bsz = None, adjustment = False,scale_quant = None,shift_quant = None,scaling_or_not_1 = None, scaling_or_not_2 = None):
        half = x[: len(x) // 2]
        # if calib5:
        if i>=0:
            min_val, max_val = torch.aminmax(half)
            scale_quant = (max_val - min_val) / 11.09
            shift_quant = - min_val / scale_quant - 5.545
        # half = half / scale_quant + shift_quant

        combined = torch.cat([half, half], dim=0)

        if calib1:
            model_out, activations = self.forward(combined, t, c,calib1=calib1)
        else:
            model_out = self.forward(combined, t, c, timesteps_fp = timesteps_fp, calib2 = calib2,a=a,i = i,step = step,calib5 = calib5,num_bsz = num_bsz, adjustment = adjustment,scale_quant = scale_quant,shift_quant = shift_quant,scaling_or_not_1 = scaling_or_not_1, scaling_or_not_2 = scaling_or_not_2)
            # model_out = self.forward(combined, t, c)
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        if calib1:
            return torch.cat([eps, rest], dim=1), activations
        else:
            return torch.cat([eps, rest], dim=1)
