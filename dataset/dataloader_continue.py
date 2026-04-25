from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    ToTensord,
    SpatialPadd,
    apply_transform,
    RandZoomd,
    RandCropByLabelClassesd,
    Identityd,
)
from utils.utils import get_key
from utils.load_config import load_config, config_to_args

from monai.data import (
    DataLoader,
    Dataset,
    list_data_collate,
    CacheDataset,
)
from monai.utils.enums import PostFix
from monai.transforms import SpatialCrop
from monai.utils import fall_back_tuple, ensure_tuple

import sys
import numpy as np
import torch
from typing import Dict, List, Mapping, Hashable
from copy import deepcopy

sys.path.append("/public/cjh/workspace/ContinualLearning")
del sys.path[0]


DEFAULT_POST_FIX = PostFix.meta()


class RandCropWithBBoxd(RandCropByPosNegLabeld):
    def __call__(
        self, data: Mapping[Hashable, torch.Tensor], lazy: bool and None = None
    ) -> List[Dict[Hashable, torch.Tensor]]:
        d = dict(data)
        fg_indices = d.pop(self.fg_indices_key, None)
        bg_indices = d.pop(self.bg_indices_key, None)

        # 1. 触发内部随机化，生成中心点
        self.randomize(
            d.get(self.label_key), fg_indices, bg_indices, d.get(self.image_key)
        )

        num_samples = self.cropper.num_samples
        ret: List = [dict(d) for _ in range(num_samples)]

        # 深拷贝非 key 字段
        for i in range(num_samples):
            for key in set(d.keys()).difference(set(self.keys)):
                ret[i][key] = deepcopy(d[key])

        lazy_ = self.lazy if lazy is None else lazy

        # 2. 执行裁剪
        for key in self.key_iterator(d):
            img_data = d[key]
            img_spatial_shape = img_data.shape[1:]
            roi_size = fall_back_tuple(
                self.cropper.spatial_size, default=img_spatial_shape
            )

            for i, cropped_img in enumerate(
                self.cropper(img_data, randomize=False, lazy=lazy_)
            ):
                ret[i][key] = cropped_img

                # 3. 核心：提取 bbox
                if key == self.keys[0]:
                    center = self.cropper.centers[i]

                    # 根据你提供的源码，SpatialCrop 初始化时会调用 compute_slices 并存入 self.slices
                    temp_cropper = SpatialCrop(roi_center=center, roi_size=roi_size)

                    # 这里的 self.slices 是一个 tuple of slice objects
                    # 例如 (slice(None, None, None), slice(10, 74), slice(20, 84), slice(5, 69))
                    # 第一个是 Channel 维，我们要的是后面的空间维
                    full_slices = ensure_tuple(temp_cropper.slices)

                    # 转换为 tensor: [[z_start, z_end], [y_start, y_end], [x_start, x_end]]
                    # 这里的 s.start 和 s.stop 可能为 None，需要处理（虽然在 ROI 模式下通常有值）
                    bbox = []
                    for s in full_slices:
                        if isinstance(s, slice) and s.start is not None:
                            bbox.append([s.start, s.stop])

                    # 存入结果字典
                    ret[i]["bbox"] = torch.tensor(bbox)

        return ret


class UniformDataset(Dataset):
    def __init__(self, data, transform, datasetkey):
        super().__init__(data=data, transform=transform)
        self.dataset_split(data, datasetkey)
        self.datasetkey = datasetkey

    def dataset_split(self, data, datasetkey):
        self.data_dic = {}
        for key in datasetkey:
            self.data_dic[key] = []
        for img in data:
            key = get_key(img["name"])
            self.data_dic[key].append(img)

        self.datasetnum = []
        for key, item in self.data_dic.items():
            assert len(item) != 0, f"the dataset {key} has no data"
            self.datasetnum.append(len(item))
        self.datasetlen = len(datasetkey)

    def _transform(self, set_key, data_index):
        data_i = self.data_dic[set_key][data_index]
        return (
            apply_transform(self.transform, data_i)
            if self.transform is not None
            else data_i
        )

    def __getitem__(self, index):
        ## the index generated outside is only used to select the dataset
        ## the corresponding data in each dataset is selelcted by the np.random.randint function
        set_index = index % self.datasetlen
        set_key = self.datasetkey[set_index]
        # data_index = int(index / self.__len__() * self.datasetnum[set_index])
        data_index = np.random.randint(self.datasetnum[set_index], size=1)[0]
        return self._transform(set_key, data_index)


class UniformCacheDataset(CacheDataset):
    def __init__(self, data, transform, cache_rate, datasetkey):
        super().__init__(data=data, transform=transform, cache_rate=cache_rate)
        self.datasetkey = datasetkey
        self.data_statis()

    def data_statis(self):
        data_num_dic = {}
        for key in self.datasetkey:
            data_num_dic[key] = 0

        for img in self.data:
            key = get_key(img["name"])
            data_num_dic[key] += 1

        self.data_num = []
        for key, item in data_num_dic.items():
            assert item != 0, f"the dataset {key} has no data"
            self.data_num.append(item)

        self.datasetlen = len(self.datasetkey)

    def index_uniform(self, index):
        ## the index generated outside is only used to select the dataset
        ## the corresponding data in each dataset is selelcted by the np.random.randint function
        set_index = index % self.datasetlen
        data_index = np.random.randint(self.data_num[set_index], size=1)[0]
        post_index = int(sum(self.data_num[:set_index]) + data_index)
        return post_index

    def __getitem__(self, index):
        post_index = self.index_uniform(index)
        # print(post_index, self.__len__())
        return self._transform(post_index)


# class LoadImageh5d(MapTransform):
#     def __init__(
#         self,
#         keys: KeysCollection,
#         reader: Optional[Union[ImageReader, str]] = None,
#         dtype: DtypeLike = np.float32,
#         meta_keys: Optional[KeysCollection] = None,
#         meta_key_postfix: str = DEFAULT_POST_FIX,
#         overwriting: bool = False,
#         image_only: bool = False,
#         ensure_channel_first: bool = False,
#         simple_keys: bool = False,
#         allow_missing_keys: bool = False,
#         *args,
#         **kwargs,
#     ) -> None:
#         super().__init__(keys, allow_missing_keys)
#         self._loader = LoadImage(
#             reader,
#             image_only,
#             dtype,
#             ensure_channel_first,
#             simple_keys,
#             *args,
#             **kwargs,
#         )
#         if not isinstance(meta_key_postfix, str):
#             raise TypeError(
#                 f"meta_key_postfix must be a str but is {type(meta_key_postfix).__name__}."
#             )
#         self.meta_keys = (
#             ensure_tuple_rep(None, len(self.keys))
#             if meta_keys is None
#             else ensure_tuple(meta_keys)
#         )
#         if len(self.keys) != len(self.meta_keys):
#             raise ValueError("meta_keys should have the same length as keys.")
#         self.meta_key_postfix = ensure_tuple_rep(meta_key_postfix, len(self.keys))
#         self.overwriting = overwriting

#     def register(self, reader: ImageReader):
#         self._loader.register(reader)

#     def __call__(self, data, reader: Optional[ImageReader] = None):
#         d = dict(data)
#         for key, meta_key, meta_key_postfix in self.key_iterator(
#             d, self.meta_keys, self.meta_key_postfix
#         ):
#             data = self._loader(d[key], reader)
#             if self._loader.image_only:
#                 d[key] = data
#             else:
#                 if not isinstance(data, (tuple, list)):
#                     raise ValueError(
#                         "loader must return a tuple or list (because image_only=False was used)."
#                     )
#                 d[key] = data[0]
#                 if not isinstance(data[1], dict):
#                     raise ValueError("metadata must be a dict.")
#                 meta_key = meta_key or f"{key}_{meta_key_postfix}"
#                 if meta_key in d and not self.overwriting:
#                     raise KeyError(
#                         f"Metadata with key {meta_key} already exists and overwriting=False."
#                     )
#                 d[meta_key] = data[1]

#         # post_label_pth = d["post_label"]
#         # with h5py.File(post_label_pth, "r") as hf:
#         #     data = hf["post_label"][()]
#         # d["post_label"] = data[0]
#         # 对于训练阶段，直接使用原始标签作为 post_label
#         if "post_label" in d:
#             # 检查是否存在 post_label 文件
#             if os.path.exists(d["post_label"]):
#                 with h5py.File(d["post_label"], "r") as hf:
#                     data = hf["post_label"][()]
#                 d["post_label"] = data[0]
#                 # 确保 post_label 有通道维度
#                 if d["post_label"].ndim == 3:
#                     # 添加通道维度
#                     d["post_label"] = np.expand_dims(d["post_label"], axis=0)
#             else:
#                 print(
#                     "初始阶段，不存在 post_label，也不需要 post_label，因此使用 label 代替"
#                 )
#                 # 如果不存在，使用原始标签
#                 d["post_label"] = d["label"]
#         return d


class RandZoomd_select(RandZoomd):
    def __call__(self, data):
        d = dict(data)
        name = d["name"]
        key = get_key(name)
        if key not in ["10_03", "10_06", "10_07", "10_08", "10_09", "10_10"]:
            return d
        d = super().__call__(d)
        return d


class RandCropByPosNegLabeld_select(RandCropByPosNegLabeld):
    def __call__(self, data):
        d = dict(data)
        name = d["name"]
        key = get_key(name)
        if key in ["10_03", "10_07", "10_08", "04"]:
            return d
        d = super().__call__(d)
        return d


class RandCropByLabelClassesd_select(RandCropByLabelClassesd):
    def __call__(self, data):
        d = dict(data)
        name = d["name"]
        key = get_key(name)
        if key not in ["10_03", "10_07", "10_08", "04"]:
            return d
        d = super().__call__(d)
        return d


class Compose_Select(Compose):
    def __call__(self, input_):
        name = input_["name"]
        key = get_key(name)
        for index, _transform in enumerate(self.transforms):
            # for RandCropByPosNegLabeld and RandCropByLabelClassesd case
            if (key in ["10_03", "10_07", "10_08", "04"]) and (index == 8):
                continue
            elif (key not in ["10_03", "10_07", "10_08", "04"]) and (index == 9):
                continue
            # for RandZoomd case
            if (key not in ["10_03", "10_06", "10_07", "10_08", "10_09", "10_10"]) and (
                index == 7
            ):
                continue
            input_ = apply_transform(
                _transform, input_, self.map_items, self.unpack_items, self.log_stats
            )
        return input_


def get_loader(args):
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),  # 0
            LoadImaged(keys=["logits"])
            if args.enable_logits_aux
            else Identityd(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(args.space_x, args.space_y, args.space_z),
                mode=("bilinear", "nearest"),
            ),  # process h5 to here
            ScaleIntensityRanged(
                keys=["image"],
                a_min=args.a_min,
                a_max=args.a_max,
                b_min=args.b_min,
                b_max=args.b_max,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            # SpatialPadd(
            #     keys=["image", "label"],
            #     spatial_size=(args.roi_x, args.roi_y, args.roi_z),
            #     mode="constant",
            # ),
            RandCropWithBBoxd(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                pos=2,
                neg=1,
                num_samples=args.num_samples,
                image_key="image",
                image_threshold=0,
            ),  # 8
            RandRotate90d(
                keys=["image", "label"],
                prob=0.10,
                max_k=3,
            ),
            RandShiftIntensityd(
                keys=["image"],
                offsets=0.10,
                prob=0.20,
            ),
            ToTensord(keys=["image", "label"]),
        ]
    )

    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(args.space_x, args.space_y, args.space_z),
                mode=("bilinear", "nearest"),
            ),  # process h5 to here
            ScaleIntensityRanged(
                keys=["image"],
                a_min=args.a_min,
                a_max=args.a_max,
                b_min=args.b_min,
                b_max=args.b_max,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            # ToTensord(keys=["image", "label"]),   # 现在最新版的 monai 不需要这个了
        ]
    )

    # training dict part
    train_img = []
    train_lbl = []
    train_logits = []
    train_name = []

    for line in open(args.train_data_txt_path):
        if line.startswith("#"):
            continue
        name = line.strip().split()[1].split(".")[0]
        train_img.append(args.data_root_path + line.strip().split()[0])
        train_lbl.append(args.data_root_path + line.strip().split()[1])
        if args.enable_logits_aux:
            train_logits.append(args.data_root_path + line.strip().split()[2])
        train_name.append(name)
    data_dicts_train = (
        [
            {"image": image, "label": label, "name": name}
            for image, label, name in zip(train_img, train_lbl, train_name)
        ]
        if not args.enable_logits_aux
        else [
            {"image": image, "label": label, "logits": logits, "name": name}
            for image, label, logits, name in zip(
                train_img, train_lbl, train_logits, train_name
            )
        ]
    )
    print("train len {}".format(len(data_dicts_train)))

    ## validation dict part
    val_img = []
    val_lbl = []
    val_name = []
    for line in open(args.val_data_txt_path):
        if line.startswith("#"):
            continue
        name = line.strip().split()[1].split(".")[0]
        val_img.append(args.data_root_path + line.strip().split()[0])
        val_lbl.append(args.data_root_path + line.strip().split()[1])
        val_name.append(name)
    data_dicts_val = [
        {"image": image, "label": label, "name": name}
        for image, label, name in zip(val_img, val_lbl, val_name)
    ]
    print("val len {}".format(len(data_dicts_val)))

    print("暂时不需要测试")

    print("没有必要专门搞一个 continue data path")

    if args.cache_dataset:
        if args.uniform_sample:
            train_dataset = UniformCacheDataset(
                data=data_dicts_train,
                transform=train_transforms,
                cache_rate=args.cache_rate,
                datasetkey=args.datasetkey,
            )
        else:
            train_dataset = CacheDataset(
                data=data_dicts_train,
                transform=train_transforms,
                cache_rate=args.cache_rate,
            )
    else:
        if args.uniform_sample:
            train_dataset = UniformDataset(
                data=data_dicts_train,
                transform=train_transforms,
                datasetkey=args.datasetkey,
            )
        else:
            train_dataset = Dataset(data=data_dicts_train, transform=train_transforms)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=list_data_collate,
    )

    if args.cache_dataset:
        val_dataset = CacheDataset(
            data=data_dicts_val,
            transform=val_transforms,
            cache_rate=args.cache_rate,
        )
    else:
        val_dataset = Dataset(data=data_dicts_val, transform=val_transforms)

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=list_data_collate,
    )
    return train_loader, val_loader, val_transforms


if __name__ == "__main__":
    args = config_to_args(
        load_config("/public/cjh/workspace/ContinualLearning/config.yaml")
    )
    train_loader, val_loader = get_loader(args)
    for index, item in enumerate(train_loader):
        print(item["image"].shape, item["label"].shape)
        input()
