import torch
from torch import nn
from tqdm import tqdm
import os
import wandb
from wandb.sdk.wandb_run import Run
import warnings
from pathlib import Path
from monai.inferers import SlidingWindowInferer
from model.swinunetr import SwinUNETR  # noqa: E402
from model.swinunetr_partial_onehot import SwinUNETR as SwinUNETR_onehot
from model.swinunetr_partial_v3 import SwinUNETR as SwinUNETR_partial_v3
from model.swinunetr_partial_v3_selfdistill import SwinUNETR as SwinUNETR_partial_v3_ds
from model.swinunetr_partial_v3_de import SwinUNETR as SwinUNETR_partial_v3_de
from dataset.dataloader_continue import get_loader
from utils.loss import DiceLoss, Multi_BCELoss, SelfDistillationLoss, Multi_MSELoss
from utils.metrics_ import SegmentationMetrics
from utils.load_config import load_config, config_to_args
from utils.best_model_saver import model_param_saver
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import logging
from torch.amp import autocast, GradScaler

scaler = GradScaler()

log = logging.getLogger(__name__)


warnings.filterwarnings("ignore")
torch.multiprocessing.set_sharing_strategy("file_system")


def train(
    args,
    train_loader,
    model,
    optimizer,
    loss_func_dice,
    loss_func_bce,
    loss_func_mse,
    loss_func_ce,
    loss_func_ds,
    wbrun: Run = None,
):
    model.train()
    loss_bce_ave = 0
    loss_mse_ave = 0
    loss_dice_ave = 0
    loss_ds_ave = 0
    epoch_iterator = tqdm(
        train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
    )
    for step, batch in enumerate(epoch_iterator):
        x, y = (
            batch["image"].to(args.device),
            batch["label"]
            .float()
            .to(
                args.device
            ),  # 一直都是使用 post_label 做训练，某种程度上不需要 Label 了。或者直接将 post_label 当做 label 算了。
        )

        if args.enable_logits_aux:
            # 之前获取 logits 的时候忘了做 sigmoid 的
            logits = batch["logits"].to(args.device)
            logits_list = []
            for i in range(len(batch["bbox"])):
                b = batch["bbox"][i]
                target_logits_patch = logits[
                    i, :, b[0, 0] : b[0, 1], b[1, 0] : b[1, 1], b[2, 0] : b[2, 1]
                ]
                logits_list.append(target_logits_patch)
            logits = torch.stack(logits_list)

        optimizer.zero_grad()
        with autocast(device_type="cuda", enabled=True):
            log.debug("computing with cross entropy loss...")
            if args.enable_ds:
                log.debug("computing with self distillation loss...")
                output_dict = model(x, return_for_distill=True)
                logit_map = output_dict["logits"]
            else:
                logit_map = model(x)[-1]

            if args.out_nonlinear == "sigmoid":
                # 这里把 organ_list 传进去，大概也就在这几个器官上做损失。
                term_seg_Dice = loss_func_dice.forward(logit_map, y, args.organ_list)
                term_seg_BCE = loss_func_bce.forward(logit_map, y, args.organ_list)

                term_seg_MSE = (
                    loss_func_mse.forward(
                        logit_map, logits, args.organ_list, args.last_stage_organ_list
                    )
                    if args.enable_logits_aux
                    else torch.tensor(0.0, device=args.device)
                )

                dist_loss = (
                    loss_func_ds(output_dict)
                    if args.enable_ds
                    else torch.tensor(0.0, device=args.device)
                )

                loss = (
                    args.bce_weight * term_seg_BCE
                    + term_seg_Dice
                    + dist_loss
                    + term_seg_MSE
                )

            loss_bce_ave += term_seg_BCE.item()
            loss_mse_ave += term_seg_MSE.item()
            loss_dice_ave += term_seg_Dice.item()
            loss_ds_ave += dist_loss.item()
            epoch_iterator.set_description(
                "Epoch=%d: Training (%d / %d Steps) (dice_loss=%2.5f, bce_loss=%2.5f, mse_loss=%2.5f, ds_loss=%2.5f)"
                % (
                    args.epoch,
                    step,
                    len(train_loader),
                    term_seg_Dice.item(),
                    term_seg_BCE.item(),
                    term_seg_MSE.item(),
                    dist_loss.item(),
                )
            )

            # elif args.out_nonlinear == "softmax":
            #     b, c, d, h, w = y.shape
            #     label = y.new_zeros((b, d, h, w), dtype=torch.long)
            #     for icls in args.organ_list:
            #         label[y[:, icls - 1] == 1] = icls
            #     term_seg_Dice = loss_func_dice.forward(logit_map[:, 1:], y, args.organ_list)
            #     term_seg_CE = loss_func_ce(logit_map, label)
            #     loss = term_seg_Dice + term_seg_CE
            #     loss_mse_ave += term_seg_MSE.item()
            #     loss_ce_ave += term_seg_CE.item()
            #     loss_dice_ave += term_seg_Dice.item()
            #     epoch_iterator.set_description(
            #         "Epoch=%d: Training (%d / %d Steps) (dice_loss=%2.5f, ce_loss=%2.5f, mse_loss=%2.5f)"
            #         % (
            #             args.epoch,
            #             step,
            #             len(train_loader),
            #             term_seg_Dice.item(),
            #             term_seg_MSE.item(),
            #             term_seg_CE.item(),
            #         )
            #     )

        # loss.backward()
        # optimizer.step()
        # torch.cuda.empty_cache()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.empty_cache()
    print(
        "Epoch=%d: ave_dice_loss=%2.5f, ave_bce_loss=%2.5f, ave_mse_loss=%2.5f, ave_ds_loss=%2.5f"
        % (
            args.epoch,
            loss_dice_ave / len(epoch_iterator),
            loss_bce_ave / len(epoch_iterator),
            loss_mse_ave / len(epoch_iterator),
            loss_ds_ave / len(epoch_iterator),
        )
    )

    return (
        loss_dice_ave / len(epoch_iterator),
        loss_bce_ave / len(epoch_iterator),
        loss_mse_ave / len(epoch_iterator),
        loss_ds_ave / len(epoch_iterator),
    )


def evaluate(
    args,
    val_loader,
    infer,
    model,
    metrics,
    wbrun: Run = None,
    saver: model_param_saver = None,
):
    assert args.out_nonlinear == "sigmoid", "Only support sigmoid"
    model.eval()
    metrics.reset()

    with torch.no_grad():
        for step, batch in tqdm(enumerate(val_loader), desc="evaluating"):
            logits = infer(batch["image"].to(args.device), model)

            spacing = [args.space_x, args.space_y, args.space_z]
            metrics.update(
                logits, batch["label"].float().cpu(), args.organ_list, spacing
            )
    summary = metrics.compute()

    # check and replace bast model
    saver.check(
        model=model,
        metrics=summary,
    )

    return summary


def process(args, wbrun: Run = None):

    torch.cuda.set_device(args.device)

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
        model = SwinUNETR_onehot(
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
    elif args.model == "SwinUNETR_partial_v3_ds":
        model = SwinUNETR_partial_v3_ds(
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            in_channels=1,
            out_channels=args.out_channels,
            feature_size=48,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
            encoding=args.trans_encoding,
            use_checkpoint=True,
        )
        log.debug("load distiall model")
    elif args.model == "SwinUNETR_partial_v3_de":
        model = SwinUNETR_partial_v3_de(
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            in_channels=1,
            out_channels=args.out_channels,
            feature_size=48,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
            encoding=args.trans_encoding,
            use_checkpoint=True,
            use_bypass=True,
            freq_alpha=0.015,
        )
        log.debug("load de model")

    # Load pre-trained weights
    store_dict = model.state_dict()
    if args.pretrain:
        pretrain_checkpoint = torch.load(args.pretrain)

        if "state_dict" in pretrain_checkpoint:
            model_dict = pretrain_checkpoint["state_dict"]
        else:
            model_dict = pretrain_checkpoint["net"]
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
            model_dict, "module."
        )

        for key in model_dict.keys():
            if "out" not in key:
                store_dict[key] = model_dict[key]
            else:
                print(f"{key} is not in model state dict")

        model.load_state_dict(store_dict)
        print("Use pretrained weights")

    # 加载预训练的词向量
    if args.model == "swinunetr_partial" and args.trans_encoding == "word_embedding":
        word_embedding = torch.load(args.word_embedding)
        model.organ_embedding.data = word_embedding.float()
        print("load word embedding")
    model.to(args.device)
    model.train()

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

    # criterion and optimizer
    # loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    loss_func_dice = DiceLoss().to(args.device)
    loss_func_bce = Multi_BCELoss().to(args.device)
    loss_func_mse = Multi_MSELoss().to(args.device)
    loss_func_ce = nn.CrossEntropyLoss()
    loss_func_ds = SelfDistillationLoss(model, args, lambda_distill=args.distill_weight)
    metrics = SegmentationMetrics(num_classes=args.out_channels, metrics=["dc"])
    saver = model_param_saver(
        save_dir=Path(args.log_dir) / args.log_name, monitor_metric=args.monitor_metric
    )

    if args.model in [
        "swinunetr_partial",
        "our_onehot",
        "SwinUNETR_partial_v3_ds",
        "SwinUNETR_partial_v3_de",
    ]:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    elif args.model == "swinunetr":
        model_params = [v for k, v in model.named_parameters() if "out" not in k]
        class_params = model.module.out.parameters()
        optimizer = torch.optim.AdamW(
            [{"params": class_params, "lr": 100 * args.lr}, {"params": model_params}],
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer, warmup_epochs=args.warmup_epoch, max_epochs=args.max_epoch
    )

    # 断点训练功能
    if args.resume:
        checkpoint = torch.load(args.resume)
        if args.dist:
            model.load_state_dict(checkpoint["net"])
        else:
            store_dict = model.state_dict()
            model_dict = checkpoint["net"]
            # for key in model_dict.keys():
            #     store_dict[".".join(key.split(".")[1:])] = model_dict[key]
            model.load_state_dict(store_dict)

        optimizer.load_state_dict(checkpoint["optimizer"])
        args.epoch = checkpoint["epoch"]
        scheduler.load_state_dict(checkpoint["scheduler"])

        print("success resume from ", args.resume)

    # training should use both train_loader and val_laoder
    train_loader, val_loader, _ = get_loader(args)

    if not os.path.isdir(os.path.join(args.log_dir, args.log_name)):
        os.mkdir(os.path.join(args.log_dir, args.log_name))

    while args.epoch < args.max_epoch:
        scheduler.step()

        loss_dice, loss_bce, loss_mse, loss_ds = train(
            args,
            train_loader,
            model,
            optimizer,
            loss_func_dice,
            loss_func_bce,
            loss_func_mse,
            loss_func_ce,
            loss_func_ds,
        )

        wbrun.log(
            {
                "train_dice_loss": loss_dice,
                "train_bce_loss": loss_bce,
                "train_mse_loss": loss_mse,
                "train_ds_loss": loss_ds,
            },
            step=args.epoch,
        )

        if args.epoch % args.store_num == 0 and args.epoch != 0:
            checkpoint = {
                "net": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": args.epoch,
            }
            torch.save(
                checkpoint,
                os.path.join(args.log_dir, args.log_name, f"epoch_{args.epoch}.pth"),
            )
            print("save model success")

        if args.epoch % args.eval_interval == 0:
            eval_metrics = evaluate(
                args,
                val_loader,
                infer,
                model,
                metrics,
                wbrun,
                saver,
            )

            wbrun.log(eval_metrics, step=args.epoch)

        args.epoch += 1


def main(log_name: str = None):
    args = config_to_args(config=load_config("config_stage_2.yaml"))

    args.log_name = log_name if log_name else args.log_name

    with wandb.init(
        project="continue_learning", config=args, name=args.log_name
    ) as run:
        process(args=args, wbrun=run)

    return args


if __name__ == "__main__":
    main()
