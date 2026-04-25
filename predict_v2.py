import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
import wandb
from monai.data import DataLoader, Dataset, list_data_collate, decollate_batch
from utils.load_config import load_config, config_to_args
from rich import print
import numpy as np

# from monai.networks.nets import SwinUNETR
from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
    Invertd,
    SaveImaged,
)
from monai.inferers import SlidingWindowInferer


from model.swinunetr import SwinUNETR
from model.swinunetr_partial_v3 import SwinUNETR as SwinUNETR_partial_v3
from model.swinunetr_partial_onehot import SwinUNETR as SwinUNETR_partial_onehot
from utils.utils import (
    organ_post_process,
)

torch.multiprocessing.set_sharing_strategy("file_system")


def get_loader(args):
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(
                    args.dataset["space_x"],
                    args.dataset["space_y"],
                    args.dataset["space_z"],
                ),
                mode=("bilinear", "nearest"),
            ),  # process h5 to here
            ScaleIntensityRanged(
                keys=["image"],
                a_min=args.dataset["a_min"],
                a_max=args.dataset["a_max"],
                b_min=args.dataset["b_min"],
                b_max=args.dataset["b_max"],
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
        ]
    )
    pred_img = []
    pred_lbl = []
    pred_name = []
    for line in open(args.dataset["predict_data_txt_path"]):
        name = line.strip().split()[1].split(".")[0]
        pred_img.append(args.dataset["data_root_path"] + line.strip().split()[0])
        pred_lbl.append(args.dataset["data_root_path"] + line.strip().split()[1])
        pred_name.append(name)
    data_dicts = [
        {"image": image, "label": label, "name": name}
        for image, label, name in zip(pred_img, pred_lbl, pred_name)
    ]
    print("predict len {}".format(len(data_dicts)))

    pred_dataset = Dataset(data=data_dicts, transform=val_transforms)
    pred_loader = DataLoader(
        pred_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=list_data_collate,
    )
    return pred_loader, val_transforms


def detransform_save(tensor_dict, input_transform, save_dir):
    # post_transforms = Compose(
    #     [
    #         # 逆转变换
    #         Invertd(
    #             keys=["one_channel_pred", "pred_logits"],
    #             transform=input_transform,
    #             orig_keys="image",
    #             nearest_interp=[True, False],  # 标签用最近邻，Logits用线性
    #             to_tensor=True,
    #         ),
    #         # 保存预测标签 (0-13)
    #         SaveImaged(
    #             keys="one_channel_pred",
    #             meta_keys="label_meta_dict",
    #             output_dir=save_dir,
    #             output_postfix="pred",  # 这里传字符串
    #             resample=False,
    #             output_dtype=np.uint8,  # 标签保存为整数
    #         ),
    #         # 保存原始 Logits
    #         SaveImaged(
    #             keys="pred_logits",
    #             meta_keys="image_meta_dict",
    #             output_dir=save_dir,
    #             output_postfix="logits",  # 这里传字符串
    #             resample=False,
    #             output_dtype=np.float32,  # Logits 保存为浮点数
    #         ),
    #     ]
    # )
    
    post_transforms = Compose(
        [
            # 逆转变换
            Invertd(
                keys=["one_channel_pred"],
                transform=input_transform,
                orig_keys="image",
                nearest_interp=[True],  # 标签用最近邻，Logits用线性
                to_tensor=True,
            ),
            # 保存预测标签 (0-13)
            SaveImaged(
                keys="one_channel_pred",
                meta_keys="label_meta_dict",
                output_dir=save_dir,
                output_postfix="pred",  # 这里传字符串
                resample=False,
                output_dtype=np.uint8,  # 标签保存为整数
            ),
        ]
    )    
    return post_transforms(tensor_dict)


def predict(model, pred_loader, val_transforms, args):
    save_dir = os.path.join(
        args.log["log_dir"],
        args.log["log_name"],
    )
    os.makedirs(save_dir, exist_ok=True)

    model.eval()

    infer = SlidingWindowInferer(
        roi_size=(args.dataset["roi_x"], args.dataset["roi_y"], args.dataset["roi_z"]),
        sw_batch_size=args.sw_batch_size,
        overlap=args.overlap,
        mode=args.mode,
        sw_device=args.device,
        buffer_steps=4,
        device="cpu",  # 结果拼接在 cpu 中，防止爆显存
    )

    for index, batch in enumerate(tqdm(pred_loader)):
        image, label, name = batch["image"].cuda(), batch["label"], batch["name"]
        
        if os.path.isdir(os.path.join(save_dir, name[0].split("/")[-1].split(".")[0])):
            continue
        with torch.no_grad():
            pred = infer(image, model)

            if args.model["out_nonlinear"] == "sigmoid":
                pred_sigmoid = F.sigmoid(pred)
                pred_hard = (pred_sigmoid > 0.5).cpu().numpy()
            elif args.model["out_nonlinear"] == "softmax":
                c = pred.size(1)
                pred_hard = torch.argmax(pred, dim=1).cpu()
                pred_hard = F.one_hot(pred_hard, num_classes=c).permute(0, 4, 1, 2, 3)
                pred_hard = pred_hard[:, 1:]
                pred_hard = pred_hard.numpy()

        pred_hard_post = organ_post_process(pred_hard, args.dataset["organ_list"])
        pred_hard_post = torch.tensor(pred_hard_post)
        torch.cuda.empty_cache()

        B, C, D, H, W = pred_hard_post.shape

        # 这里使用模型的预测类别，替换标签中对应的旧类别的数据，同时保持新的类别不变
        # 同时添加一个 logits 引导，防止伪标签失效
        # one_channel_pred = label.new_zeros((D, H, W))
        one_channel_pred = label.clone().squeeze()
        for icls in args.dataset["organ_list"]:
            one_channel_pred[one_channel_pred == icls] = 0  #
            one_channel_pred[pred_hard_post[0, icls - 1] == 1] = icls

        batch["one_channel_pred"] = one_channel_pred.cpu()[None, None]
        batch["pred_logits"] = pred.cpu().squeeze(dim=1)
        
        if args.save_npz:
            np.save(
                os.path.join(save_dir, name[0].split("/")[-1].split(".")[0] + "_logits.npy"),
                arr = pred_sigmoid.as_tensor().squeeze()[:len(args.dataset["organ_list"])].cpu().numpy()
            )
        
        de_batch = decollate_batch(batch)

        detransform_save(de_batch[0], val_transforms, os.path.join(save_dir))
        torch.cuda.empty_cache()
        del pred


def main(log_name: str = None):
    args = config_to_args(config=load_config("config_pred_stage_1.yaml"))

    args.log["log_name"] = log_name if log_name else args.log["log_name"]

    with wandb.init(
        project="continue_learning", config=args, name=args.log["log_name"]
    ) as run:
        if args.model["type"] == "swinunetr_partial":
            model = SwinUNETR_partial_v3(
                img_size=(
                    args.dataset["roi_x"],
                    args.dataset["roi_y"],
                    args.dataset["roi_z"],
                ),
                in_channels=1,
                out_channels=args.model["out_channels"],
                feature_size=48,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                dropout_path_rate=0.0,
                use_checkpoint=False,
                encoding=args.model["trans_encoding"],
            )
        elif args.model["type"] == "swinunetr":
            model = SwinUNETR(
                img_size=(
                    args.dataset[".roi_x"],
                    args.dataset["roi_y"],
                    args.dataset["roi_z"],
                ),
                in_channels=1,
                out_channels=args.model["out_channels"],
                feature_size=48,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                dropout_path_rate=0.0,
                use_checkpoint=False,
            )
        elif args.model["type"] == "our_onehot":
            model = SwinUNETR_partial_onehot(
                img_size=(
                    args.dataset["roi_x"],
                    args.dataset["roi_y"],
                    args.dataset["roi_z"],
                ),
                in_channels=1,
                out_channels=args.model["out_channels"],
                feature_size=48,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                dropout_path_rate=0.0,
                use_checkpoint=False,
                encoding=args.model["trans_encoding"],
            )

        # Load pre-trained weights
        checkpoint = torch.load(args.model["resume"])
        load_dict = checkpoint if "net" not in checkpoint else checkpoint["net"]
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
            load_dict, "module."
        )

        model.load_state_dict(load_dict)
        print("Use pretrained weights")

        torch.cuda.set_device(args.device)
        model.cuda()

        torch.backends.cudnn.benchmark = True

        pred_loader, pred_transforms = get_loader(args)

        predict(model, pred_loader, pred_transforms, args)


if __name__ == "__main__":
    main()
