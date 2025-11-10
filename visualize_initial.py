# import torch
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# import numpy as np
# from dataset import load_data
# import os
# import random

# Get one batch

# images, labels = next(iter(loader))
# # images = images.to('cuda')  # or CPU
# with torch.no_grad():
#     outputs = model(images)
#     concepts = outputs['concepts']
#     class_logits = outputs['class']
#     pred_classes = class_logits.argmax(1)

# def visualize_concepts(model, dataloader, device, n_concepts_to_show=5):
#     model.eval().to(device)

#     # store conv features
#     feats = []
#     def hook_fn(module, inp, out):
#         feats.append(out.detach())
#     h = model.first_model.resnet.layer4.register_forward_hook(hook_fn)

#     batch = next(iter(dataloader))
#     # Depending on your dataset settings:
#     if len(batch) == 3:
#         imgs, labels, attrs = batch
#     else:
#         imgs, labels = batch
#     imgs = imgs.to(device)

#     with torch.no_grad():
#         out = model(imgs)
#         # handle tuple or dict
#         if isinstance(out, tuple):
#             concepts, preds = out  # End2EndModel
#         elif isinstance(out, dict):
#             concepts, preds = out["concepts"], out["class"]
#         else:
#             raise TypeError(f"Unexpected output type {type(out)}")

#         preds = preds.softmax(dim=1)
#         pred_classes = preds.argmax(dim=1)

#     h.remove()
#     fmap = feats[0].cpu()        # [B,C,H,W]
#     w = model.first_model.concept_head.weight.data.cpu()  # [n_attr, C]

#     for i in range(min(len(imgs), 8)):   # show up to 8 samples
#         fig, axes = plt.subplots(2, n_concepts_to_show, figsize=(3*n_concepts_to_show,6))
#         img_np = imgs[i].cpu().permute(1,2,0).numpy()
#         img_np = (img_np - img_np.min())/(img_np.max()-img_np.min())

#         # handle multi-channel case
#         if img_np.shape[2] > 3:
#             img_vis = img_np[:, :, :3]  # visualize only RGB channels
#         else:
#             img_vis = img_np

#         # axes[0, j].imshow(img_vis)


#         for j in range(n_concepts_to_show):
#             heat = torch.einsum("c,chw->hw", w[j], fmap[i])
#             heat = F.relu(heat)
#             heat = heat / (heat.max() + 1e-6)
#             heat = F.interpolate(
#                 heat.unsqueeze(0).unsqueeze(0),
#                 size=img_np.shape[:2], mode='bilinear'
#             )[0,0].numpy()

#             # img_vis = img_np[:, :, :3]
#             axes[0,j].imshow(img_vis)
#             axes[0,j].axis("off")
#             if j == 0:
#                 axes[0,j].set_title(f"Input\nPred: {pred_classes[i].item()}")

#             axes[1,j].imshow(img_vis)
#             axes[1,j].imshow(heat, cmap="jet", alpha=0.45)
#             axes[1,j].axis("off")
#             axes[1,j].set_title(f"Concept {j}")

#         plt.tight_layout()
#         plt.show()

"""
Visualization script for Concept Bottleneck Models (CBMs)
Supports fine-tuned models and the CUB dataset
"""

import os
import torch
import yaml
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from dataset import CUBDataset  # Your provided dataset loader
from template_model import End2EndModel
from PIL import Image
import seaborn as sns
from project_utils import to_numpy, to_numpy_img
from interpretability.utils import grad_to_img, explanation_mode
from sklearn.metrics import accuracy_score, f1_score
plt.rcParams['text.usetex'] = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_cub_class_names(class_file):
    idx_to_class = {}
    with open(class_file, "r") as f:
        for line in f:
            idx, name = line.strip().split(' ')
            idx_to_class[int(idx) - 1] = name  # shift to 0-based index
    return idx_to_class

def load_cub_concept_names(attributes_file):
    concept_names = []
    with open(attributes_file, 'r') as f:
        for line in f:
            idx, name = line.strip().split(' ', 1)
            concept_names.append(name)
    return concept_names


# -------------------------------------------------------------------------
# 1. Load Config
# -------------------------------------------------------------------------
def load_config(yaml_path):
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    return config

# -------------------------------------------------------------------------
# 2. Load fine-tuned model
# -------------------------------------------------------------------------
def load_fine_tuned_model(config):
    model_path = '/h/sneharao/CBM/logs/best_model_399epochs.pth'

    print("Loading fine-tuned model...")
    checkpoint = torch.load(model_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Handle case where torch.load returns the model directly
    if isinstance(checkpoint, torch.nn.Module):
        model = checkpoint
    elif isinstance(checkpoint, dict):
        model = End2EndModel(config["model"]["params"])
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=False)
    else:
        raise ValueError(f"Unexpected checkpoint type: {type(checkpoint)}")

    model.to(device)
    model.eval()
    print("Model loaded.")
    return model

# -------------------------------------------------------------------------
# 3. Load CUB Dataset
# -------------------------------------------------------------------------
def load_test_data(pkl_path, image_dir='images', resol=299):
    transform = transforms.Compose([
        transforms.CenterCrop(resol),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[2, 2, 2])
    ])

    dataset = CUBDataset(
        pkl_file_paths=[pkl_path],
        use_attr=True,
        no_img=False,
        uncertain_label=False,
        image_dir=image_dir,
        n_class_attr=2,
        transform=transform
    )
    return dataset
    
# -------------------------------------------------------------------------
# 4. Saliency / Concept visualization helper
# -------------------------------------------------------------------------
def compute_saliency_map(model, image_tensor, class_idx=None):
    """Compute saliency map for an image and given class"""
    image_tensor = image_tensor.unsqueeze(0).to(device)
    image_tensor.requires_grad = True

    output = model(image_tensor)

    # Handle tuple or dict outputs
    if isinstance(output, tuple):
        concepts, class_logits = output
    elif isinstance(output, dict):
        concepts, class_logits = output["concepts"], output["class"]
    else:
        raise TypeError(f"Unexpected model output type: {type(output)}")

    if class_idx is None:
        class_idx = class_logits.argmax(dim=1).item()

    loss = class_logits[0, class_idx]
    loss.backward()

    saliency = image_tensor.grad.data.abs().squeeze().cpu()
    saliency, _ = torch.max(saliency, dim=0)
    return saliency


def overlay_saliency(image_tensor, saliency):
    """Overlay saliency map over RGB image"""
    import matplotlib.cm as cm

    img = image_tensor[:3].permute(1, 2, 0).cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min())

    sal = saliency.numpy()
    sal = (sal - sal.min()) / (sal.max() - sal.min())
    sal_color = cm.jet(sal)[:, :, :3]

    overlay = 0.6 * img + 0.4 * sal_color
    return overlay

# -------------------------------------------------------------------------
# 5. Visualization Grid
# -------------------------------------------------------------------------
def visualize_grid(images, saliencies, n_cols=4, save_path=None):
    n = len(images)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows))

    for i, ax in enumerate(axes.flat):
        if i < n:
            ax.imshow(images[i])
            ax.set_title(f"Sample {i+1}")
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

def to_numpy_img_display(img):
    img = img - img.min()
    img = img / (img.max() + 1e-8)
    return np.clip(img, 0, 1)

# -------------------------------------------------------------------------
# 6. Main
# -------------------------------------------------------------------------
if __name__ == "__main__":
    yaml_file = os.path.join(os.path.dirname(__file__), "bcos.yaml")
    with open(yaml_file) as f:
        config = yaml.safe_load(f)

    model = load_fine_tuned_model(config)

    print("Loading test CUB dataset...")
    test_dataset = load_test_data(
        config["test_dir"]
    )

    print(f"Loaded {len(test_dataset)} samples.")

    # Choose some samples to visualize
    num_samples = 8
    images, saliencies = [], []
    class_names = load_cub_class_names('/h/sneharao/CBM/CUB_200_2011/CUB_200_2011/classes.txt')
    concepts = load_cub_concept_names('/h/sneharao/CBM/CUB_200_2011/attributes.txt')
    c_acc, acc = 0, 0 

    for i in range(num_samples):
        img_6ch, label, attr = test_dataset[i]
        img_rgb = img_6ch[:3]  # extract RGB part
        image_tensor = img_6ch.unsqueeze(0).to(device)
        image_tensor.requires_grad = True

        # Top 2 predictions
        # top2, c_idcs = model(image_tensor)[0].topk(2)
        concept_output = model.first_model(image_tensor)
        concept_logits = concept_output[1]
        print(f'concept output : {concept_output}')
        print(f'concept_logits: {concept_logits}')
        top2, c_idcs = concept_logits.topk(2)

        print(top2.shape)
        top2 = top2.squeeze()
        c_idcs = c_idcs.squeeze()

        # pred_concepts = torch.sigmoid(concept_output["concepts"]).detach().cpu().numpy()
        # pred_binary = (pred_concepts > 0.5).astype(int)

        # concept_acc = accuracy_score(np.array(attr), pred_binary)
        # concept_f1 = f1_score(np.array(attr), pred_binary)
        # print(f"Concept accuracy: {concept_acc:.3f}, F1: {concept_f1:.3f}")

        # Convert attr to numpy safely
        if isinstance(attr, torch.Tensor):
            attr_np = attr.cpu().numpy()
        elif isinstance(attr, list):
            attr_np = np.array(attr)
        else:
            raise TypeError(f"Unexpected attribute type: {type(attr)}")

        print(f"Concept output : {concept_output}")
        # Convert model predictions to binary (0/1)
        pred_concepts = torch.sigmoid(concept_output[1]).detach().cpu().numpy()
        pred_binary = (pred_concepts > 0.5).astype(int)

        # Flatten arrays for comparison
        attr_np = attr_np.flatten()
        pred_binary = pred_binary.flatten()

        # ---- Restrict to the first 112 concepts ----
        # MAX_CONCEPTS = 112

        # # Convert model predictions to binary (0/1)
        # pred_concepts = torch.sigmoid(concept_output["concepts"]).detach().cpu().numpy()
        # pred_binary = (pred_concepts > 0.5).astype(int)

        # # Convert attr to numpy safely
        # if isinstance(attr, torch.Tensor):
        #     attr_np = attr.cpu().numpy()
        # elif isinstance(attr, list):
        #     attr_np = np.array(attr)
        # else:
        #     raise TypeError(f"Unexpected attribute type: {type(attr)}")

        # # Ensure both arrays have at least 112 concepts
        # min_len = min(MAX_CONCEPTS, attr_np.shape[-1], pred_binary.shape[-1])
        # attr_np = attr_np[..., :min_len].flatten()
        # pred_binary = pred_binary[..., :min_len].flatten()

        # # Restrict concept names to the same range
        # concepts = concepts[:min_len]

        # # Compare ground truth and predicted concepts
        # correct = np.sum(attr_np == pred_binary)
        # concept_acc = correct / len(attr_np)
        # print(f"→ Concept prediction accuracy (first {min_len} concepts): {concept_acc*100:.2f}%")


        # Make sure both are same length
        min_len = min(len(attr_np), len(pred_binary))
        attr_np = attr_np[:min_len]
        pred_binary = pred_binary[:min_len]

        # Compare ground truth and predicted concepts
        correct = 0
        for idx, (gt, pred) in enumerate(zip(attr_np, pred_binary)):
            gt_val = int(gt)
            pred_val = int(pred)
            if gt_val == pred_val:
                correct += 1
                # print(f"[✓] Correct concept: {concepts[idx]}")
            # else:
                # print(f"[✗] Incorrect concept: {concepts[idx]} (GT={gt_val}, Pred={pred_val})")

        concept_acc = correct / len(attr_np)
        c_acc += concept_acc
        print(f"→ Concept prediction accuracy: {concept_acc*100:.2f}% for this sample")

        # -----------------------------
        # Get predicted class from model2 (classifier)
        # -----------------------------
        with torch.no_grad():
            class_output = model.sec_model(concept_logits)
            pred_class_idx = class_output.argmax(dim=1).item()
            pred_class_name = class_names[pred_class_idx]
            true_class_name = class_names[label]

        if pred_class_name == true_class_name:
            acc += 1
        # top_2, c_idcs = model(image_tensor)[1].squeeze(0).topk(2)
        # Compute w_1:
        image_tensor.grad = None
        top2[0].backward(retain_graph=True)
        w1 = image_tensor.grad.clone()

        image_tensor.grad = None
        top2[1].backward()
        w2 = image_tensor.grad.clone()

        sns.set_style("white")
        for logit, c_idx, w in zip(top2, c_idcs, [w1[0], w2[0]]):
            print('++++', c_idx)
            print(c_idx)
            extracted_concept = (
                concepts[int(c_idx)] if int(c_idx) < len(concepts) else f"Concept_{c_idx}"
            )
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))
            # The 'positive evidence' is derived from those locations that have a net positive contribution
            # to the respective output, according to the linear transformation matrix W_{1->L}
            # fig.suptitle(f"Positive evidence for {name_map[c_idx.item()].split(',')[0]}", fontsize=18)
            ax1.imshow(to_numpy_img_display(to_numpy_img(img_rgb)))
            ax1.set_title(f"True: {true_class_name}\nPred: {pred_class_name}")
            ax1.axis("off")
            # Decoding the linear transform w based on the angles of the color channels.
            # Only showing positively contributing pixels.
            ax2.imshow(grad_to_img(img_6ch, w))
            ax2.set_title(f"Concept acc: {concept_acc}")
            ax2.axis("off")
            # for ax in [ax1, ax2]:
            #     ax.set_xticks([])
            #     ax.set_yticks([])
        
        # fig.suptitle(f'Sample {i} | Logit {logit.item():.3f} | Concepts {extracted_concepts}')
        # # fig.savefig(f"sample_{i}_class_{int(c_idx)}.png")
        # save_path = f"sample_{i}_class_{int(c_idx)}.png"
        # plt.tight_layout()
        # # Limit max figure width before saving
        # fig.set_size_inches(min(fig.get_size_inches()[0], 40), fig.get_size_inches()[1])
        # plt.savefig(save_path, dpi=100, bbox_inches='tight')
        # plt.close(fig)

        plt.suptitle(f"Sample {i} | Logit {logit.item():.3f}", fontsize=10)
        plt.tight_layout()

        save_path = f"sample_{i}_concept_{c_idx}_pred_{pred_class_name}.png"
        fig.set_size_inches(min(fig.get_size_inches()[0], 12), fig.get_size_inches()[1])
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        # sal = compute_saliency_map(model, img_6ch)
        # overlay = overlay_saliency(img_rgb, sal)

        # images.append(overlay)
        # saliencies.append(sal)

    print(acc)
    print(f'Concept prediction accuracy for {num_samples} samples is {(c_acc/num_samples) * 100}')
    print(f'Class prediction accuracy for {num_samples} samples is {(acc/num_samples) * 100}')

    # visualize_grid(images, saliencies, n_cols=4, save_path="visualization_grid.png")
    print("Visualization saved as visualization_grid.png")



# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# data_dir = os.path.join('/Users/sneha/Downloads/CBMs', 'test' + '.pkl')
# dataloader = load_data([data_dir], True, False, 16, 'images',
#                        n_class_attr=2)
# # Assume you already loaded your trained CBM:
# # Load the entire model directly
# model = torch.load('/Users/sneha/Downloads/CBMs/logs/best_model_1.pth', weights_only=False)
# visualize_cub_concepts_from_loader(model, dataloader)