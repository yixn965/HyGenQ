import torch
import torch_fidelity
# import cv2
import numpy as np
import time

from util.logger_utils import logger, outpath


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)

def evaluate(model_without_ddp, vae, ema_params, args, epoch, batch_size=16, cfg=1.0, use_ema=True,scaling_or_not_1 = None, scaling_or_not_2 = None):

    model_without_ddp.eval()

    num_steps = args.num_images // batch_size + 1

    import os
    from datetime import datetime
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_folder_evaluation = os.path.join(outpath,
                            f"ariter{args.num_iter}-diffsteps{args.num_sampling_steps}-"
                            f"temp{args.temperature}-{args.cfg_schedule}cfg{cfg}-"
                            f"image{args.num_images}-{current_time}-evaluate-{batch_size}")
    logger.info(f"Save folder: {save_folder_evaluation}")

    if not os.path.exists(save_folder_evaluation):
        os.makedirs(save_folder_evaluation)
        logger.info(f"Created save folder at {save_folder_evaluation}")

    class_num = args.class_num
    assert args.num_images % class_num == 0  # number of images per class must be the same
    class_label_gen = np.arange(0, class_num).repeat(args.num_images // class_num)
    class_label_gen = np.hstack([class_label_gen, np.zeros(50000)])

    used_time = 0
    gen_img_cnt = 0
    import cv2

    for i in range(num_steps):
        logger.info(f"Generation step {i+1}/{num_steps}")

        batch_start = i * batch_size
        batch_end = (i + 1) * batch_size
        labels_gen_batch = class_label_gen[batch_start:batch_end]
        labels_gen_batch = torch.Tensor(labels_gen_batch).long().cuda()

        torch.cuda.synchronize()
        start_time = time.time()

        # Generate images
        with torch.no_grad():
            # with torch.cuda.amp.autocast():
                sampled_tokens = model_without_ddp.sample_tokens(
                    bsz=batch_size, num_iter=args.num_iter, cfg=cfg,
                    cfg_schedule=args.cfg_schedule, labels=labels_gen_batch,
                    temperature=args.temperature,num_bsz = i,adjustment = False,scaling_or_not_1 = scaling_or_not_1, scaling_or_not_2 = scaling_or_not_2
                )
                # logger.info(f"Sampled tokens shape: {sampled_tokens.shape}")
                sampled_images = vae.decode(sampled_tokens / 0.2325)
                # logger.info(f"Sampled images shape: {sampled_images.shape}")

        if i >= 1:
            torch.cuda.synchronize()
            used_time += time.time() - start_time
            gen_img_cnt += batch_size
            logger.info(f"Generated {gen_img_cnt} images in {used_time:.5f} seconds, "
                        f"{used_time / gen_img_cnt:.5f} sec per image")

        sampled_images = sampled_images.detach().cpu()
        sampled_images = (sampled_images + 1) / 2

        # Save generated images
        for b_id in range(sampled_images.size(0)):
            img_id = i * sampled_images.size(0) + b_id
            if img_id >= args.num_images:
                break
            gen_img = np.round(np.clip(sampled_images[b_id].numpy().transpose([1, 2, 0]) * 255, 0, 255))
            gen_img = gen_img.astype(np.uint8)[:, :, ::-1]

            cv2.imwrite(os.path.join(save_folder_evaluation, '{}.png'.format(str(img_id).zfill(5))), gen_img)

    # Compute FID and Inception Score
    if args.img_size == 256:
        input2 = None
        fid_statistics_file = os.environ.get("HYGENQ_FID_STATISTICS")
        if not fid_statistics_file:
            raise ValueError("Set HYGENQ_FID_STATISTICS to the FID statistics file before evaluation.")
    else:
        raise NotImplementedError

    weights_path = os.environ.get("TORCH_FIDELITY_INCEPTION_WEIGHTS")
    if weights_path:
        logger.info(f"Using local torch-fidelity Inception weights: {weights_path}")

    metrics_dict = torch_fidelity.calculate_metrics(
        input1=save_folder_evaluation,
        input2=input2,
        fid_statistics_file=fid_statistics_file,
        cuda=True,
        isc=True,
        fid=True,
        kid=False,
        prc=False,
        verbose=False,
        feature_extractor_weights_path=weights_path,
    )
    fid = metrics_dict['frechet_inception_distance']
    inception_score = metrics_dict['inception_score_mean']
    postfix = ""
    if use_ema:
        postfix = postfix + "_ema"
    if not cfg == 1.0:
        postfix = postfix + "_cfg{}".format(cfg)

    logger.info("FID: {:.4f}, Inception Score: {:.4f}".format(fid, inception_score))

    # # Remove the temporary folder
    # shutil.rmtree(save_folder)
    # logger.info(f"Removed temporary folder {save_folder}")
