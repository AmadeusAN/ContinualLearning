# %%
import numpy as np
import os
import SimpleITK as sitk

# %%
felix2paot_label_mapping = {
    1: 8,    # Aorta
    2: 12,    # Adrenal Gland
    4: 33,    # CBD
    5: 25,    # Celiac abdominal aorta
    6: 18,    # Colon
    7: 14,    # Duodenum
    8: 4,    # Gall Bladder
    9: 9,    # IVC / postcava
    10: 3,    # L Kidney
    11: 2,    # R Kidney
    12: 6,    # Liver
    13: 11,    # Pancreas
    14: 34,    # pancreatic duct
    15: 35,    # SMA
    16: 19,    # small bowel / intestine
    18: 1,    # spleen
    19: 7,    # stomach
    20: 10,    # veins
    21: 36,    # Kidney_LtRV
    22: 37,    # Kidney_RtRV
    24: 38,    # CBD stent
}

paot2felix_label_mapping = {v: k for k, v in felix2paot_label_mapping.items()}
paot2felix_label_mapping[13] = 2

# %%
fnames = open('dataset/dataset_list/felix_small_step2_train.txt').readlines()
fnames = [x.split()[1] for x in fnames]
fnames = [x.split('/')[-1] for x in fnames]
fnames[0]

# %%
save_dir = 'output/felix_onehot_step1/step2_pseudo_label'
os.makedirs(save_dir, exist_ok=True)
for fname in fnames:
    label = sitk.ReadImage(os.path.join('Input Image Dir', fname))
    pred = sitk.ReadImage(os.path.join('output/felix_onehot_step1/test_epoch_100/predict', fname.split('.')[0], fname.split('.')[0] + '_pred.nii.gz'))
    label_np = sitk.GetArrayFromImage(label)
    pred_np = sitk.GetArrayFromImage(pred)
    pseudo = np.zeros_like(label_np)    # felix style
    label_cls = list(np.unique(label_np))    # felix style
    label_cls.remove(0)
    for paot_cls in [1, 2, 3, 4, 6, 9, 11, 12, 13]:    # pretrained classes
        felix_cls = paot2felix_label_mapping[paot_cls]
        pseudo[pred_np == paot_cls] = felix_cls
        if felix_cls in label_cls:
            label_cls.remove(felix_cls)
    for icls in label_cls:
        pseudo[label_np == icls] = icls
    pseudo = sitk.GetImageFromArray(pseudo)
    pseudo.SetOrigin(label.GetOrigin())
    pseudo.SetSpacing(label.GetSpacing())
    pseudo.SetDirection(label.GetDirection())
    sitk.WriteImage(pseudo, os.path.join(save_dir, fname))

# %%
fnames = open('dataset/dataset_list/felix_small_step3_train.txt').readlines()
fnames = [x.split()[1] for x in fnames]
fnames = [x.split('/')[-1] for x in fnames]
fnames[0]

# %%
save_dir = 'output/felix_v3_step2_run2/step3_pseudo_label'
os.makedirs(save_dir, exist_ok=True)
for fname in fnames:
    label = sitk.ReadImage(os.path.join('Input Image Dir', fname))
    pred = sitk.ReadImage(os.path.join('output/felix_v3_step2_run2/test_epoch_100/predict', fname.split('.')[0], fname.split('.')[0] + '_pred.nii.gz'))
    label_np = sitk.GetArrayFromImage(label)
    pred_np = sitk.GetArrayFromImage(pred)
    pseudo = np.zeros_like(label_np)    # felix style
    label_cls = list(np.unique(label_np))    # felix style
    label_cls.remove(0)
    for paot_cls in [1, 2, 3, 4, 6, 9, 11, 12, 13, 7, 14, 18, 19]:    # pretrained classes
        felix_cls = paot2felix_label_mapping[paot_cls]
        pseudo[pred_np == paot_cls] = felix_cls
        if felix_cls in label_cls:
            label_cls.remove(felix_cls)
    for icls in label_cls:
        pseudo[label_np == icls] = icls
    pseudo = sitk.GetImageFromArray(pseudo)
    pseudo.SetOrigin(label.GetOrigin())
    pseudo.SetSpacing(label.GetSpacing())
    pseudo.SetDirection(label.GetDirection())
    sitk.WriteImage(pseudo, os.path.join(save_dir, fname))

# %% [markdown]
# # BTCV -> LiTS

# %%
lits2paot_label_mapping = {
    1: 6,    # Liver
    2: 27,    # Liver tumor
}

paot2lits_label_mapping = {v: k for k, v in lits2paot_label_mapping.items()}

# %%
fnames = open('./dataset/dataset_list/lits_train.txt').readlines()
fnames = [fname.split('/')[-1].split('.')[0] for fname in fnames]
os.makedirs('output/btcv_v3_cls-all/pseudo_label_lits', exist_ok=True)
for fname in fnames:
    print(fname)
    label = sitk.ReadImage(os.path.join('LiTS label dir', fname))
    pred = sitk.ReadImage(os.path.join('output/btcv_v3_cls-all/test_epoch_100/predict', fname, fname + '_pred.nii.gz'))
    label_np = sitk.GetArrayFromImage(label)
    pred_np = sitk.GetArrayFromImage(pred)
    pseudo = np.zeros_like(label_np)
    label_cls = list(np.unique(label_np))    # lits style
    label_cls.remove(0)
    for icls in [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14]:    # no liver
        pseudo[pred_np == icls] = icls
        # if icls in label_cls:
        #     label_cls.remove(icls)
    for lits_cls in label_cls:
        pseudo[label_np == lits_cls] = lits2paot_label_mapping[lits_cls]
    pseudo = sitk.GetImageFromArray(pseudo)
    pseudo.CopyInformation(label)
    sitk.WriteImage(pseudo, os.path.join('output/btcv_v3_cls-all/pseudo_label_lits/', fname + '.nii.gz'))
    

# %%
fnames = open('./dataset/dataset_list/lits_train.txt').readlines()
fnames = [fname.split('/')[-1].split('.')[0] for fname in fnames]
os.makedirs('output/btcv_swin-ce_cls-all/pseudo_label_lits', exist_ok=True)
for fname in fnames:
    label = sitk.ReadImage(os.path.join('LiTS label dir', fname))
    pred = sitk.ReadImage(os.path.join('output/btcv_swin-ce_cls-all/test_epoch_100/predict', fname, fname + '_pred.nii.gz'))
    label_np = sitk.GetArrayFromImage(label)
    pred_np = sitk.GetArrayFromImage(pred)
    pseudo = np.zeros_like(label_np)
    label_cls = list(np.unique(label_np))    # lits style
    label_cls.remove(0)
    for icls in [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14]:
        pseudo[pred_np == icls] = icls
        # if icls in label_cls:
        #     label_cls.remove(icls)
    for lits_cls in label_cls:
        pseudo[label_np == lits_cls] = lits2paot_label_mapping[lits_cls]
    pseudo = sitk.GetImageFromArray(pseudo)
    pseudo.CopyInformation(label)
    sitk.WriteImage(pseudo, os.path.join('output/btcv_swin-ce_cls-all/pseudo_label_lits/', fname + '.nii.gz'))
    

# %%
fnames = open('./dataset/dataset_list/lits_train.txt').readlines()
fnames = [fname.split('/')[-1].split('.')[0] for fname in fnames]
os.makedirs('output/btcv_onehot_cls-all/pseudo_label_lits', exist_ok=True)
for fname in fnames:
    print(fname)
    label = sitk.ReadImage(os.path.join('LiTS label dir', fname))
    pred = sitk.ReadImage(os.path.join('output/btcv_onehot_cls-all/test_epoch_100/predict', fname, fname + '_pred.nii.gz'))
    label_np = sitk.GetArrayFromImage(label)
    pred_np = sitk.GetArrayFromImage(pred)
    pseudo = np.zeros_like(label_np)
    label_cls = list(np.unique(label_np))    # lits style
    label_cls.remove(0)
    for icls in [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14]:    # no liver
        pseudo[pred_np == icls] = icls
        # if icls in label_cls:
        #     label_cls.remove(icls)
    for lits_cls in label_cls:
        pseudo[label_np == lits_cls] = lits2paot_label_mapping[lits_cls]
    pseudo = sitk.GetImageFromArray(pseudo)
    pseudo.CopyInformation(label)
    sitk.WriteImage(pseudo, os.path.join('output/btcv_onehot_cls-all/pseudo_label_lits/', fname + '.nii.gz'))

# %%



