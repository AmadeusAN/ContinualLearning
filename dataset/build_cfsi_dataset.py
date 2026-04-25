import os
import torch
import numpy as np
import random
import glob
import shutil
from monai.transforms import LoadImage, SaveImage, Spacing, CropForeground, SpatialPad
from pathlib import Path
import sys

sys.path.append("/public/cjh/workspace/ContinualLearning")

from utils.load_config import load_config
from tqdm import tqdm

import os
import torch
import numpy as np
import random
import glob
import shutil
from pathlib import Path
from tqdm import tqdm

from monai.transforms import (
    LoadImage,
    SaveImage,
    Spacingd,
    SpatialPadd,
    Compose,
    ToTensord,
    EnsureChannelFirstd,
)

import os
import torch
import numpy as np
import random
import glob
import shutil
from pathlib import Path
from tqdm import tqdm
import os
import torch
import numpy as np
import random
import glob
import shutil
from pathlib import Path
from tqdm import tqdm

from monai.transforms import (
    LoadImage,
    SaveImage,
    Spacingd,
    SpatialPadd,
    EnsureChannelFirstd,
    ToTensord,
    Compose,
)
import os
import torch
import numpy as np
import random
import glob
import shutil
from pathlib import Path
from tqdm import tqdm

from monai.transforms import (
    LoadImage,
    SaveImage,
    Spacingd,
    Resized,
    EnsureChannelFirstd,
    ToTensord,
    Compose,
)


def cfs_interpolate(
    x: torch.Tensor, ref: torch.Tensor, alpha: float = 0.01, lambda_val: float = None
) -> torch.Tensor:
    """核心 CFSI 函数"""
    if lambda_val is None:
        lambda_val = random.uniform(0.0, 1.0)

    x_freq = torch.fft.fftn(x, dim=(-3, -2, -1))
    x_freq_shift = torch.fft.fftshift(x_freq, dim=(-3, -2, -1))
    amp = torch.abs(x_freq_shift)
    phase = torch.angle(x_freq_shift)

    ref_freq = torch.fft.fftn(ref, dim=(-3, -2, -1))
    ref_freq_shift = torch.fft.fftshift(ref_freq, dim=(-3, -2, -1))
    ref_amp = torch.abs(ref_freq_shift)

    D, H, W = amp.shape[-3], amp.shape[-2], amp.shape[-1]
    mask = torch.zeros((D, H, W), device=amp.device, dtype=torch.float32)
    cz, cy, cx = D // 2, H // 2, W // 2
    r_d = int(alpha * D / 2)
    r_h = int(alpha * H / 2)
    r_w = int(alpha * W / 2)
    mask[
        max(0, cz - r_d) : min(D, cz + r_d),
        max(0, cy - r_h) : min(H, cy + r_h),
        max(0, cx - r_w) : min(W, cx + r_w),
    ] = 1.0

    new_amp = (1 - lambda_val) * amp + lambda_val * ref_amp
    new_amp = new_amp * mask + amp * (1 - mask)

    new_freq_shift = new_amp * torch.exp(1j * phase)
    new_freq = torch.fft.ifftshift(new_freq_shift, dim=(-3, -2, -1))
    x_aug = torch.fft.ifftn(new_freq, dim=(-3, -2, -1)).real
    return x_aug


def apply_cfs_i_augmentation(
    data_root: str,
    alpha: float = 0.01,
    lambda_min: float = 0.0,
    lambda_max: float = 1.0,
    random_seed: int = 42,
    output_suffix: str = "_CFSI",
    overwrite: bool = False,
    output_path: str or Path = None,
    target_spacing: tuple = (1.5, 1.5, 1.5),
):
    if isinstance(output_path, str):
        output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "img").mkdir(exist_ok=True)
    (output_path / "label").mkdir(exist_ok=True)

    random.seed(random_seed)
    torch.manual_seed(random_seed)

    img_dir = os.path.join(data_root, "img")
    label_dir = os.path.join(data_root, "label")

    img_paths = sorted(glob.glob(os.path.join(img_dir, "img*.nii.gz")))
    if not img_paths:
        raise ValueError(f"在 {img_dir} 未找到任何 img*.nii.gz 文件！")

    print(
        f"共找到 {len(img_paths)} 个样本，开始【Spacing + 计算平均尺寸 + Resize + CFSI】..."
    )

    loader = LoadImage(image_only=False, reader="ITKReader")
    saver = SaveImage(
        output_postfix=output_suffix,
        output_ext=".nii.gz",
        separate_folder=False,
        writer="ITKWriter",
    )

    # ===================== Pass 1: 计算 Spacing 后的平均尺寸 =====================
    print("正在计算所有样本 Spacing 后的平均尺寸...")
    spacing_d = Compose(
        [
            EnsureChannelFirstd(keys=["image"]),
            Spacingd(keys=["image"], pixdim=target_spacing, mode="bilinear"),
            ToTensord(keys=["image"]),
        ]
    )

    shapes_after_spacing = []
    for p in img_paths:
        data, meta = loader(p)
        d = {"image": data, "image_meta_dict": meta}
        d = spacing_d(d)
        shapes_after_spacing.append(d["image"].shape[1:])  # (D, H, W)

    # 计算平均尺寸并四舍五入（保证是整数）
    avg_d = round(sum(s[0] for s in shapes_after_spacing) / len(shapes_after_spacing))
    avg_h = round(sum(s[1] for s in shapes_after_spacing) / len(shapes_after_spacing))
    avg_w = round(sum(s[2] for s in shapes_after_spacing) / len(shapes_after_spacing))
    target_size = (avg_d, avg_h, avg_w)
    print(f"✅ 计算得到平均尺寸: {target_size}")

    # ===================== Pass 2: 完整预处理 pipeline（使用 Resize） =====================
    preproc_d = Compose(
        [
            EnsureChannelFirstd(keys=["image"]),
            Spacingd(keys=["image"], pixdim=target_spacing, mode="bilinear"),
            Resized(
                keys=["image"], spatial_size=target_size, mode="bilinear"
            ),  # ← 改为 Resize
            ToTensord(keys=["image"]),
        ]
    )

    label_preproc_d = Compose(
        [
            EnsureChannelFirstd(keys=["label"]),
            Spacingd(keys=["label"], pixdim=target_spacing, mode="nearest"),
            Resized(
                keys=["label"], spatial_size=target_size, mode="nearest"
            ),  # ← 改为 Resize
            ToTensord(keys=["label"]),
        ]
    )

    # ===================== 正式处理每个样本 =====================
    for idx, img_path in tqdm(enumerate(img_paths), total=len(img_paths)):
        basename = os.path.basename(img_path)
        print(f"[{idx + 1}/{len(img_paths)}] 处理 {basename}")

        # 1. 当前图像
        x_data, x_meta = loader(img_path)
        x_dict = {"image": x_data, "image_meta_dict": x_meta}
        x_dict = preproc_d(x_dict)
        x_preproc = x_dict["image"]
        x_meta = x_dict["image_meta_dict"]
        x = x_preproc

        # 2. 参考图像
        other_paths = [p for p in img_paths if p != img_path]
        ref_path = random.choice(other_paths)
        ref_data, ref_meta = loader(ref_path)
        ref_dict = {"image": ref_data, "image_meta_dict": ref_meta}
        ref_dict = preproc_d(ref_dict)
        ref = ref_dict["image"]

        # 3. 执行 CFSI
        lambda_val = random.uniform(lambda_min, lambda_max)
        x_aug = cfs_interpolate(x, ref, alpha=alpha, lambda_val=lambda_val)

        # 4. 保存增强图像
        aug_filename = Path(img_path).name.replace(".nii.gz", f"{output_suffix}.nii.gz")
        aug_img_path = output_path / "img" / aug_filename
        # 5. 保存原始预处理图像
        orig_filename = Path(img_path).name
        orig_img_path = output_path / "img" / orig_filename
        saver(
            x_preproc[0].cpu().numpy(),
            meta_data=x_meta,
            filename=str(orig_img_path).split(".")[0],
        )

        # if aug_img_path.exists() and not overwrite:
        #     continue
        # saver(
        #     x_aug[0].cpu().numpy(),
        #     meta_data=x_meta,
        #     filename=str(aug_img_path).split(".")[0],
        # )

        # 6. 处理 label
        label_filename = basename.replace("img", "label")
        label_path = os.path.join(label_dir, label_filename)
        label_data, label_meta = loader(label_path)
        label_dict = {"label": label_data, "label_meta_dict": label_meta}
        label_dict = label_preproc_d(label_dict)
        label_preproc = label_dict["label"]

        new_label_path = output_path / "label" / Path(label_path).name
        label_aug_path = (
            output_path
            / "label"
            / Path(label_filename).name.replace(".nii.gz", f"{output_suffix}.nii.gz")
        )

        if not new_label_path.exists() or overwrite:
            saver(
                label_preproc[0].cpu().numpy(),
                meta_data=x_meta,
                filename=str(new_label_path).split(".")[0],
            )
        if not label_aug_path.exists() or overwrite:
            saver(
                label_preproc[0].cpu().numpy(),
                meta_data=x_meta,
                filename=str(label_aug_path).split(".")[0],
            )

    print("✅ 预处理 + CFSI 增强全部完成！")
    print(f"   输出目录：{output_path}")
    print(f"   所有样本已 resize 到平均尺寸: {target_size}")


# ===================== 使用示例 =====================
if __name__ == "__main__":
    data_root = "data/01_Multi-Atlas_Labeling"

    config = load_config()
    apply_cfs_i_augmentation(
        data_root=data_root,
        alpha=0.01,  # 可改
        lambda_min=0.0,
        lambda_max=1.0,
        random_seed=config["seed"],
        overwrite=False,
        output_path="data/01_Multi-Atlas_Labeling_CFSI",
        target_spacing=(1.5, 1.5, 1.5),
    )
