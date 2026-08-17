import argparse
import os
import random

import numpy as np
import torch

from models import mar
from quant.build_model import build_model
from quant.quant_model import quant_model, set_quant_state
from util.logger_utils import logger


def get_args_parser():
    parser = argparse.ArgumentParser(
        "HyGenQ calibration for MAR with Diffusion Loss"
    )

    parser.add_argument("--model", default="mar_large")
    parser.add_argument("--img_size", default=256, type=int)
    parser.add_argument("--vae_embed_dim", default=16, type=int)
    parser.add_argument("--vae_stride", default=16, type=int)
    parser.add_argument("--patch_size", default=1, type=int)
    parser.add_argument("--mask_ratio_min", default=0.7, type=float)
    parser.add_argument("--label_drop_prob", default=0.1, type=float)
    parser.add_argument("--class_num", default=1000, type=int)
    parser.add_argument("--attn_dropout", default=0.1, type=float)
    parser.add_argument("--proj_dropout", default=0.1, type=float)
    parser.add_argument("--buffer_size", default=64, type=int)

    parser.add_argument("--diffloss_d", default=12, type=int)
    parser.add_argument("--diffloss_w", default=1536, type=int)
    parser.add_argument("--num_sampling_steps", default="100")
    parser.add_argument("--diffusion_batch_mul", default=1, type=int)
    parser.add_argument("--grad_checkpointing", action="store_true")

    parser.add_argument("--num_iter", default=64, type=int)
    parser.add_argument("--cfg", default=1.0, type=float)
    parser.add_argument("--cfg_schedule", default="linear")
    parser.add_argument("--temperature", default=1.0, type=float)

    parser.add_argument("--w_bits", default=8, type=int)
    parser.add_argument("--a_bits", default=8, type=int)
    parser.add_argument("--input_quant", action="store_true")
    parser.add_argument("--weight_quant", action="store_true")
    parser.add_argument("--calib5", action="store_true")
    parser.add_argument("--adjustment", action="store_true")
    parser.add_argument("--include_layers", default="")
    parser.add_argument("--exclude_layers", default="")

    parser.add_argument("--calibration_samples", default=32, type=int)
    parser.add_argument("--calibration_batch_size", default=32, type=int)
    parser.add_argument("--calib_label_seed", default=None, type=int)
    parser.add_argument("--first_calib_only", action="store_true")

    parser.add_argument("--resume", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=1, type=int)
    return parser


def seed_torch(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def calibrate(model, args, calib5, adjustment):
    label_rng = (
        random.Random(args.calib_label_seed)
        if args.calib_label_seed is not None
        else random
    )
    labels = label_rng.sample(range(args.class_num), args.calibration_samples)
    device = next(model.parameters()).device
    labels = torch.tensor(labels, dtype=torch.long, device=device)

    with torch.no_grad():
        for start in range(0, len(labels), args.calibration_batch_size):
            current_labels = labels[start:start + args.calibration_batch_size]
            logger.info(
                "Calibrating labels %d-%d of %d",
                start,
                start + len(current_labels),
                len(labels),
            )
            model.sample_tokens(
                bsz=current_labels.size(0),
                num_iter=args.num_iter,
                cfg=args.cfg,
                cfg_schedule=args.cfg_schedule,
                labels=current_labels,
                temperature=args.temperature,
                calib1=False,
                calib2=False,
                calib3=False,
                calib5=calib5,
                adjustment=adjustment,
            )


def main(args):
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    seed_torch(args.seed)
    logger.info("Using device: %s", device)
    logger.info("Calibration seed: %d", args.seed)

    model = mar.__dict__[args.model](
        img_size=args.img_size,
        vae_stride=args.vae_stride,
        patch_size=args.patch_size,
        vae_embed_dim=args.vae_embed_dim,
        mask_ratio_min=args.mask_ratio_min,
        label_drop_prob=args.label_drop_prob,
        class_num=args.class_num,
        attn_dropout=args.attn_dropout,
        proj_dropout=args.proj_dropout,
        buffer_size=args.buffer_size,
        diffloss_d=args.diffloss_d,
        diffloss_w=args.diffloss_w,
        num_sampling_steps=args.num_sampling_steps,
        diffusion_batch_mul=args.diffusion_batch_mul,
        grad_checkpointing=args.grad_checkpointing,
    )
    model = build_model(model).to(device).eval()

    checkpoint_path = os.path.join(args.resume, "checkpoint-last.pth")
    if args.resume and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        model.load_state_dict(checkpoint["model_ema"], strict=False)
        logger.info("Loaded checkpoint from %s", checkpoint_path)
    else:
        logger.info("No checkpoint loaded.")

    input_quant_params = {"n_bits": args.a_bits, "channel_wise": False}
    weight_quant_params = {"n_bits": args.w_bits, "channel_wise": False}
    q_model = quant_model(
        model,
        input_quant_params=input_quant_params,
        weight_quant_params=weight_quant_params,
    ).to(device).eval()

    set_quant_state(
        q_model,
        input_quant=args.input_quant,
        weight_quant=args.weight_quant,
        include_layers=args.include_layers.split(",") if args.include_layers else None,
        exclude_layers=args.exclude_layers.split(",") if args.exclude_layers else None,
    )
    logger.info(
        "Quantization state: input=%s, weight=%s",
        args.input_quant,
        args.weight_quant,
    )

    calibrate(q_model, args, calib5=args.calib5, adjustment=False)
    if not args.first_calib_only:
        logger.info("Running quantization adjustment calibration.")
        calibrate(q_model, args, calib5=False, adjustment=args.adjustment)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Calibration completed.")


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
