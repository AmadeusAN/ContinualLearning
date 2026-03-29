import SimpleITK as sitk
import numpy as np
import os

# 读取数据文件
with open("./dataset/dataset_list/btcv_train_new.txt", "r") as f:
    lines = f.readlines()

# 检查前几个文件的维度
for i, line in enumerate(lines[:3]):
    img_path, label_path = line.strip().split()
    img_full_path = os.path.join("./data/", img_path)
    label_full_path = os.path.join("./data/", label_path)

    print(f"检查文件 {i + 1}:")
    print(f"图像路径: {img_full_path}")
    print(f"标签路径: {label_full_path}")

    # 读取图像
    if os.path.exists(img_full_path):
        img = sitk.ReadImage(img_full_path)
        img_np = sitk.GetArrayFromImage(img)
        print(f"图像维度: {img_np.shape}")
    else:
        print(f"图像文件不存在: {img_full_path}")
    
    # 读取标签
    if os.path.exists(label_full_path):
        label = sitk.ReadImage(label_full_path)
        label_np = sitk.GetArrayFromImage(label)
        print(f"标签维度: {label_np.shape}")
    else:
        print(f"标签文件不存在: {label_full_path}")
    
    print("=" * 50)
