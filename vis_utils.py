import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt


def compute_raw_cam(clip_model, processor, image, text_label, device):
    """
    Computes raw activation-weighted CAM before relative scaling.
    """
    prompt = f"a photo of {text_label}"
    inputs = processor(text=[prompt], images=image, return_tensors="pt", padding=True).to(device)

    target_layer = clip_model.vision_model.encoder.layers[-1].layer_norm1
    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    f_handle = target_layer.register_forward_hook(forward_hook)
    b_handle = target_layer.register_full_backward_hook(backward_hook)

    clip_model.zero_grad()
    outputs = clip_model(**inputs)
    
    # Backprop theo logit similarity
    score = outputs.logits_per_image[0, 0]
    score.backward(retain_graph=False)

    f_handle.remove()
    b_handle.remove()

    act = activations[0]
    grad = gradients[0]

    spatial_act = act[:, 1:, :]    
    spatial_grad = grad[:, 1:, :]  

    weights = spatial_grad.mean(dim=1, keepdim=True)
    cam = (spatial_act * weights).sum(dim=-1)
    cam = F.relu(cam)

    grid_size = int(np.sqrt(cam.size(-1)))
    cam = cam.reshape(1, 1, grid_size, grid_size)
    cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
    cam_np = cam.squeeze().detach().cpu().numpy()

    clip_model.zero_grad()
    return cam_np, score.item()


def apply_heatmap_on_image(orig_img, cam_map):
    img_resized = orig_img.resize((224, 224))
    img_np = np.array(img_resized)
    if len(img_np.shape) == 2:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)

    cam_clipped = np.clip(cam_map, 0.0, 1.0)
    heatmap = np.uint8(255 * cam_clipped)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(img_np, 0.55, heatmap_colored, 0.45, 0)
    return overlay


def generate_and_save_gradcam(model, dataset, settings, num_samples=2, epoch=None):
    device = torch.device(settings["misc"]["device"])

    if "checkpoint" in settings["train"]:
        ckpt_dir = settings["train"]["checkpoint"]
        timestamp = os.path.basename(os.path.normpath(ckpt_dir))
        save_dir = os.path.join("expmgr", "heatmaps", timestamp)
    else:
        save_dir = os.path.join("expmgr", "heatmaps")

    os.makedirs(save_dir, exist_ok=True)

    from transformers import CLIPModel
    orig_clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

    for p in orig_clip.parameters():
        p.requires_grad = True
    for p in model.clip_model.parameters():
        p.requires_grad = True

    processor, curr_clip = model.processor, model.clip_model
    actual_dataset = dataset.dataset if hasattr(dataset, "dataset") else dataset
    indices = dataset.indices if hasattr(dataset, "indices") else list(range(len(dataset)))

    target_configs = {
        "apple": {"label_id": 0,  "status": "Forget", "indices": []},
        "fox":   {"label_id": 34, "status": "Retain", "indices": []},
        "rose":  {"label_id": 70, "status": "Retain", "indices": []}
    }

    for idx in indices:
        item = actual_dataset[idx]
        lbl = item.get("label", None)
        txt = str(item.get("text", "")).lower()
        for name, cfg in target_configs.items():
            if len(cfg["indices"]) < num_samples:
                if (lbl is not None and lbl == cfg["label_id"]) or (name in txt):
                    cfg["indices"].append(idx)
        if all(len(cfg["indices"]) >= num_samples for cfg in target_configs.values()):
            break

    for name, cfg in target_configs.items():
        status_label = cfg["status"]
        for i, idx in enumerate(cfg["indices"]):
            item = actual_dataset[idx]
            img = item["image"]

            # Lay CAM va score goc
            cam_before, score_before = compute_raw_cam(orig_clip, processor, img, name, device)
            cam_after, score_after = compute_raw_cam(curr_clip, processor, img, name, device)

            # Dung max cua mo hinh goc lam scale chuan de so sanh tuyet doi
            ref_max = max(cam_before.max(), 1e-6)
            cam_before_norm = cam_before / ref_max
            cam_after_norm = cam_after / ref_max

            overlay_before = apply_heatmap_on_image(img, cam_before_norm)
            overlay_after = apply_heatmap_on_image(img, cam_after_norm)

            fig, axes = plt.subplots(1, 3, figsize=(11, 4))
            plt.subplots_adjust(top=0.82)
            axes[0].imshow(img.resize((224, 224)))
            axes[0].set_title(f"(a) {name.capitalize()} [{status_label}]", fontsize=11, pad=12)
            axes[0].axis("off")

            axes[1].imshow(overlay_before)
            axes[1].set_title(f"(b) Before unlearning\nScore: {score_before:.2f}", fontsize=10, pad=12)
            axes[1].axis("off")

            axes[2].imshow(overlay_after)
            axes[2].set_title(f"(c) After unlearning\nScore: {score_after:.2f}", fontsize=10, pad=12)
            axes[2].axis("off")

            epoch_prefix = f"epoch_{epoch}_" if epoch is not None else ""
            filename = f"{epoch_prefix}{name}_{status_label.lower()}_sample_{i+1}_cam.png"
            plt.savefig(os.path.join(save_dir, filename), dpi=200, bbox_inches="tight")
            plt.close()

    print(f"[*] Da xuat heatmap chuan hoa tuyet doi vao: {save_dir}")