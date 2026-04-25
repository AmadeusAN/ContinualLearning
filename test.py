import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import argparse
import time
import SimpleITK as sitk
from rich import print

from monai.losses import DiceCELoss
from monai.data import load_decathlon_datalist, decollate_batch
from monai.transforms import AsDiscrete, Compose, Invertd, SaveImaged
from monai.metrics import DiceMetric
from monai.inferers import SlidingWindowInferer
# from monai.networks.nets import SwinUNETR

from model.swinunetr import SwinUNETR
from model.swinunetr_partial_v3 import SwinUNETR as SwinUNETR_partial_v3
from model.swinunetr_partial_onehot import SwinUNETR as SwinUNETR_partial_onehot
from dataset.dataloader_continue import get_loader
from utils import loss
from utils.utils import (
    dice_score,
    threshold_organ,
    visualize_label,
    merge_label,
    get_key,
)
from utils.utils import TEMPLATE, ORGAN_NAME, NUM_CLASS
from utils.load_config import load_config, config_to_args

torch.multiprocessing.set_sharing_strategy("file_system")

NUM_CLASS = 38


def detransform_save(tensor_dict, input_transform, save_dir):
    post_transforms = Compose(
        [
            Invertd(
                keys=["label", "one_channel_pred"],
                transform=input_transform,
                orig_keys="image",
                nearest_interp=True,
                to_tensor=True,
            ),
            SaveImaged(
                keys=["label"],
                meta_keys="label_meta_dict",
                output_dir=save_dir,
                output_postfix="gt",
                resample=False,
            ),
            SaveImaged(
                keys=["one_channel_pred"],
                meta_keys="label_meta_dict",
                output_dir=save_dir,
                output_postfix="pred",
                resample=False,
            ),
        ]
    )
    return post_transforms(tensor_dict)


def validation(model, ValLoader, val_transforms, args):
    model.eval()

    dice_list = {}
    for key in TEMPLATE.keys():
        dice_list[key] = np.zeros((2, NUM_CLASS))  # 1st row for dice, 2nd row for count
    percase_result_str = ""

    # create inferer
    infer = SlidingWindowInferer(
        roi_size=(args.roi_x, args.roi_y, args.roi_z),
        sw_batch_size=args.sw_batch_size,
        overlap=args.overlap,
        mode=args.mode,
        sw_device=args.device,
        buffer_steps=4,
        device="cpu",  # 结果拼接在 cpu 中，防止爆显存
    )

    for index, batch in enumerate(tqdm(ValLoader)):
        image, label, name = batch["image"].cuda(), batch["label"], batch["name"]
        label = F.one_hot(
            label.squeeze(dim=1).long(), num_classes=int(label.max()) + 1
        ).permute(0, 4, 1, 2, 3)

        with torch.no_grad():
            pred = infer(image, model)

            if args.out_nonlinear == "sigmoid":
                pred_sigmoid = F.sigmoid(pred)
                pred_hard = pred_sigmoid > 0.5
            elif args.out_nonlinear == "softmax":
                pred_hard = torch.argmax(pred, dim=1)
                pred_hard = F.one_hot(pred_hard, num_classes=args.out_channels).permute(
                    0, 4, 1, 2, 3
                )
                pred_hard = pred_hard[:, 1:]
            # pred_hard = pred_hard.cpu().numpy()

        B, C, D, H, W = pred_hard.shape
        for b in range(B):
            case_name = name[b].split("/")[-1]
            content = f"case {case_name} | "
            template_key = get_key(name[b])
            pred_hard_post = pred_hard

            for organ in args.organ_list:
                if torch.sum(label[b, organ - 1, :, :, :].cuda()) != 0:
                    dice_organ, recall, precision = dice_score(
                        pred_hard_post[b, organ - 1, :, :, :].cuda(),
                        label[b, organ - 1, :, :, :].cuda(),
                    )
                    dice_list[template_key][0][organ - 1] += dice_organ.item()
                    dice_list[template_key][1][organ - 1] += 1
                    content += "%s: %.4f, " % (ORGAN_NAME[organ - 1], dice_organ.item())
                    print(
                        "%s: dice %.4f, recall %.4f, precision %.4f."
                        % (
                            ORGAN_NAME[organ - 1],
                            dice_organ.item(),
                            recall.item(),
                            precision.item(),
                        )
                    )
            print(content)
            percase_result_str += content + "\n"

        # if args.store_result:
        #     # pred_sigmoid_store = (pred_sigmoid.cpu().numpy() * 255).astype(np.uint8)
        #     # label_store = (label.numpy()).astype(np.uint8)

        #     one_channel_pred = label.new_zeros((D, H, W))
        #     for icls in args.organ_list:
        #         one_channel_pred[pred_hard_post[0, icls - 1] == 1] = icls

        #     batch["one_channel_pred"] = one_channel_pred.cpu()[None, None]
        #     de_batch = decollate_batch(batch)

        #     detransform_save(
        #         de_batch[0],
        #         val_transforms,
        #         os.path.join(save_dir, "predict", name[0].split("/")[0]),
        #     )

        # del pred, pred_hard, pred_hard_post
        # torch.cuda.empty_cache()

    ave_organ_dice = np.zeros((2, NUM_CLASS))

    with open(os.path.join(args.log_name, "result.txt"), "w") as f:
        for key in TEMPLATE.keys():
            content = "Task%s| " % (key)
            for organ in args.organ_list:
                dice = dice_list[key][0][organ - 1] / dice_list[key][1][organ - 1]
                content += "%s: %.4f, " % (ORGAN_NAME[organ - 1], dice)
                ave_organ_dice[0][organ - 1] += dice_list[key][0][organ - 1]
                ave_organ_dice[1][organ - 1] += dice_list[key][1][organ - 1]
            print(content)
            f.write(content)
            f.write("\n")
        content = "Average | "
        for i in args.organ_list:
            content += "%s: %.4f, " % (
                ORGAN_NAME[i - 1],
                ave_organ_dice[0][i - 1] / ave_organ_dice[1][i - 1],
            )
        print(content)
        f.write(content)
        f.write("\n")
        print(np.mean(ave_organ_dice[0] / ave_organ_dice[1]))
        f.write(
            "%s: %.4f, " % ("average", np.mean(ave_organ_dice[0] / ave_organ_dice[1]))
        )
        f.write("\n")
        f.write(percase_result_str)


def main(log_name: str = None):
    args = config_to_args(config=load_config("config_test.yaml"))

    args.log_name = log_name if log_name else args.log_name
    # prepare the 3D model
    if args.model == "swinunetr_partial":
        model = SwinUNETR_partial_v3(
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            in_channels=1,
            out_channels=args.out_channels,
            feature_size=48,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
            use_checkpoint=False,
            encoding=args.trans_encoding,
        )
    elif args.model == "swinunetr":
        model = SwinUNETR(
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            in_channels=1,
            out_channels=args.out_channels,
            feature_size=48,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
            use_checkpoint=False,
        )
    elif args.model == "our_onehot":
        model = SwinUNETR_partial_onehot(
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            in_channels=1,
            out_channels=args.out_channels,
            feature_size=48,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
            use_checkpoint=False,
            encoding=args.trans_encoding,
        )

    # Load pre-trained weights
    checkpoint = torch.load(args.resume)["net"]
    torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
        checkpoint, "module."
    )

    model.load_state_dict(checkpoint)
    print("Use pretrained weights")

    torch.cuda.set_device(args.device)
    model.cuda()
    torch.backends.cudnn.benchmark = True
    _, val_loader, val_transforms = get_loader(args)
    validation(model, val_loader, val_transforms, args)


if __name__ == "__main__":
    main()
