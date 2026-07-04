from light_training.dataloading.dataset import get_test_loader_from_test_list
import torch
from monai.inferers import SlidingWindowInferer
from light_training.evaluation.metric import dice
from light_training.trainer import Trainer
from monai.utils import set_determinism
from light_training.evaluation.metric import dice
set_determinism(123)
import os
from light_training.prediction import Predictor
import SimpleITK as sitk
from medpy import metric
import os, numpy as np
import matplotlib
import torch.nn as nn
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
data_dir = "./data/train_fullres_process"
env = "pytorch"
max_epoch = 1000
batch_size = 2
val_every = 2
num_gpus = 1
device = "cuda:0"
patch_size = [128, 128, 128]
class BraTSTrainer(Trainer):
    def __init__(self, env_type, max_epochs, batch_size, device="cpu", val_every=1, num_gpus=1, logdir="./logs/",
                 master_ip='localhost', master_port=17750, training_script="3_train.py"):
        super().__init__(env_type, max_epochs, batch_size, device, val_every, num_gpus, logdir, master_ip, master_port,
                         training_script)
        self.patch_size = patch_size
        self.augmentation = False
    def convert_labels(self, labels):
        return labels.float()
    def get_input(self, batch):
        image = batch["data"]
        label = batch["seg"]
        properties = batch["properties"]
        raw_label = label
        label = self.convert_labels(label)
        return image, label, properties, raw_label
    def define_model_segmambav2(self):
        from models.HCR.HCR_netwok import hcr
        model = hcr(
            res_ratio=1,
            in_channels=2,
            out_channels=2,
            embed_dims=(16, 32, 64, 128, 128),
            dropout_rate=0.2,
            deep_supervision=True
        )
        model_path = "./data/3D_parameter_ISLES2022/HCR_ISLES_2022.pth"
        new_sd = self.filte_state_dict(torch.load(model_path, map_location="cpu"))
        model.load_state_dict(new_sd, strict=False)
        model.eval()
        window_infer = SlidingWindowInferer(roi_size=patch_size,
                                            sw_batch_size=2,
                                            overlap=0.5,
                                            progress=True,
                                            mode="gaussian")
        predictor = Predictor(window_infer=window_infer,
                              mirror_axes=[0, 1, 2])
        file_name = os.path.splitext(os.path.basename(model_path))[0]
        save_path = "./data/3D_results/isles22" + "/" + file_name
        os.makedirs(save_path, exist_ok=True)
        save_visual_path = "./data/3D_results/isles22" + "/" + file_name
        os.makedirs(save_visual_path, exist_ok=True)
        return model, predictor, save_path, save_visual_path
    def convert_labels_unsqueeze(self, labels):
        labels = labels.unsqueeze(dim=0)
        result = [labels == 0, labels == 1]
        return torch.cat(result, dim=0).float()
    def validation_step(self, batch):
        image, label, properties, raw_label = self.get_input(batch)
        model, predictor, save_path, save_visual_path = self.define_model_segmambav2()
        model_output = predictor.maybe_mirror_and_predict(image, model, device=device)
        model_output = predictor.predict_raw_probability(model_output, properties=properties)
        model_output = model_output.argmax(dim=0)
        model_output = predictor.predict_noncrop_probability(model_output, properties)
        predictor.save_to_nii(model_output,
                              raw_spacing=[2, 2, 2],
                              case_name=properties['name'][0],
                              save_dir=save_path)
        voxel_spacing = [2, 2, 2]
        case_name = properties['name'][0]
        raw_data_dir = "./data/ISLES_Handle/"
        adc_itk = os.path.join(raw_data_dir, case_name, "adc.nii.gz")
        adc_itk = sitk.ReadImage(adc_itk)
        adc_array = sitk.GetArrayFromImage(adc_itk).astype(np.int32)
        gt_itk = os.path.join(raw_data_dir, case_name, "mask.nii.gz")
        gt_itk = sitk.ReadImage(gt_itk)
        gt_array = sitk.GetArrayFromImage(gt_itk).astype(np.int32)
        gt_array = torch.from_numpy(gt_array)
        gt_array = self.convert_labels_unsqueeze(gt_array)
        model_final_output = torch.from_numpy(model_output)
        model_output = self.convert_labels_unsqueeze(model_final_output)
        print(f"Processing {case_name}")
        print(f"Model output shape: {model_output.shape}")
        print(f"GT shape: {gt_array.shape}")
        save_topk_slices_isles(model_output, gt_array, case_name, adc_array,
                               save_dir=save_visual_path, top_k=4, step=10, dpi=400)
        label = gt_array
        c = 2
        dices = []
        hd95s = []
        for i in range(1, c):
            output_i = model_output[i].cpu().numpy()
            label_i = label[i].cpu().numpy()
            dice_val, hd96 = cal_metric(output_i, label_i, voxel_spacing)
            dices.append(dice_val)
            hd95s.append(hd96)
        print(f"Dice: {dices}")
        print(f"HD95: {hd95s}")
        return dices, hd95s
    def filte_state_dict(self, sd):
        if "module" in sd:
            sd = sd["module"]
        new_sd = {}
        for k, v in sd.items():
            k = str(k)
            new_k = k[7:] if k.startswith("module") else k
            new_sd[new_k] = v
        del sd
        return new_sd
def cal_metric(pred, gt, voxel_spacing=[2, 2, 2]):
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt, voxelspacing=voxel_spacing)
        return np.array([dice, hd95])
    else:
        return np.array([0.0, 50])
def save_topk_slices_isles(pred, label, case_name, ADC_bg, top_k=4, step=10,
                           save_dir="/home/Videos", dpi=400):
    os.makedirs(save_dir, exist_ok=True)
    def to_np(x):
        if hasattr(x, "cpu"):
            x = x.cpu()
        return np.asarray(x)
    pred, label, ADC_bg = to_np(pred), to_np(label), to_np(ADC_bg)
    C, Z, H, W = pred.shape
    def collapse_labels(multi_hot):
        bg, lesion = multi_hot
        final = np.zeros_like(lesion, dtype=np.int32)
        final[lesion > 0.5] = 1
        return final
    pred_labels = collapse_labels(pred)
    gt_labels = collapse_labels(label)
    lesion_areas = np.array([np.sum(gt_labels[z] > 0) for z in range(Z)])
    sorted_idx = np.argsort(-lesion_areas)
    sorted_idx = [idx for idx in sorted_idx if lesion_areas[idx] > 0]
    if len(sorted_idx) == 0:
        print(f"Warning: No lesion found in {case_name}")
        return []
    idx = sorted_idx[::step][:top_k]
    my_color = 'purple'
    alpha = 0.45
    def overlay(image2d, mask2d, color='purple', alpha=0.45):
        bg = (image2d - image2d.min()) / (image2d.max() - image2d.min() + 1e-8)
        img = np.stack([bg] * 3, axis=-1)
        m = (mask2d > 0).astype(np.uint8)
        if m.sum() > 0:
            img = img * (1 - m[..., None] * alpha) + m[..., None] * alpha * np.array(mcolors.to_rgb(color))
        return img
    def get_bbox(image2d, pad=2):
        coords = np.argwhere(image2d > 0)
        if coords.shape[0] == 0:
            return (0, image2d.shape[0], 0, image2d.shape[1])
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        y0 = max(y0 - pad, 0)
        y1 = min(y1 + pad, image2d.shape[0] - 1)
        x0 = max(x0 - pad, 0)
        x1 = min(x1 + pad, image2d.shape[1] - 1)
        return (y0, y1, x0, x1)
    out_files = []
    for rank, z in enumerate(idx):
        slice_img = ADC_bg[z]
        img_pred = overlay(slice_img, pred_labels[z], color=my_color, alpha=alpha)
        y0, y1, x0, x1 = get_bbox(slice_img, pad=2)
        img_pred = img_pred[y0:y1 + 1, x0:x1 + 1, :]
        dice_slice = 2 * np.sum(pred_labels[z] * gt_labels[z]) / (
                np.sum(pred_labels[z]) + np.sum(gt_labels[z]) + 1e-8)
        f_pred = os.path.join(save_dir, f"{case_name}_z{z:03d}_rank{rank + 1}_pred.png")
        plt.figure(figsize=(4, 4), dpi=dpi)
        plt.imshow(img_pred)
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(f_pred, bbox_inches="tight", pad_inches=0)
        plt.close()
        out_files.append(f_pred)
        print(f"Slice {z} (rank {rank + 1}): Dice: {dice_slice:.3f}")
    return out_files
if __name__ == "__main__":
    trainer = BraTSTrainer(env_type=env,
                           max_epochs=max_epoch,
                           batch_size=batch_size,
                           device=device,
                           logdir="",
                           val_every=val_every,
                           num_gpus=num_gpus,
                           master_port=16667,
                           training_script=__file__)
    from data.test_list import test_list
    test_ds = get_test_loader_from_test_list(data_dir=data_dir, test_list=test_list)
    trainer.validation_single_gpu_two_metrics(test_ds)
