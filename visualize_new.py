"""
Visualization script for Concept Bottleneck Models (CBMs) using BcosResNet18 backbone.
Displays predicted and true class labels along with concept weights and heatmaps.
"""

import os
import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, f1_score
from dataset import CUBDataset
from template_model import End2EndModel
from project_utils import to_numpy, to_numpy_img
from interpretability.utils import grad_to_img
from PIL import Image

plt.rcParams['text.usetex'] = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------------------------------------------------------
# 1. Utility Loaders
# -------------------------------------------------------------------------
def load_cub_class_names(class_file):
    idx_to_class = {}
    with open(class_file, "r") as f:
        for line in f:
            idx, name = line.strip().split(' ')
            idx_to_class[int(idx) - 1] = name
    return idx_to_class


def load_cub_concept_names(attributes_file):
    concept_names = []
    with open(attributes_file, 'r') as f:
        for line in f:
            idx, name = line.strip().split(' ', 1)
            concept_names.append(name)
    return concept_names


def load_config(yaml_path):
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


# -------------------------------------------------------------------------
# 2. Load Fine-tuned Model
# -------------------------------------------------------------------------
def load_fine_tuned_model(config, model_path):
    print(f"Loading fine-tuned model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, torch.nn.Module):
        model = checkpoint
    elif isinstance(checkpoint, dict):
        model = End2EndModel(config["model"]["params"])
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=False)
    else:
        raise ValueError(f"Unexpected checkpoint type: {type(checkpoint)}")

    model.to(device).eval()
    print("Model loaded successfully.")
    return model


# -------------------------------------------------------------------------
# 3. Load Dataset
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

def to_numpy_img_display(img):
    img = img - img.min()
    img = img / (img.max() + 1e-8)
    return np.clip(img, 0, 1)

# -------------------------------------------------------------------------
# 4. Visualization Loop
# -------------------------------------------------------------------------
def visualize_model(model, test_dataset, class_names, concepts, num_samples=8):
    c_acc, acc = 0, 0

    for i in range(num_samples):
        img_6ch, label, attr = test_dataset[i]
        img_rgb = img_6ch[:3]
        image_tensor = img_6ch.unsqueeze(0).to(device)
        image_tensor.requires_grad = True

        # Forward pass through model
        concept_output = model.first_model(image_tensor)
        concept_logits = concept_output[1:] if isinstance(concept_output, (list, tuple)) else concept_output["concepts"]

        # Compute concept accuracy
        if isinstance(attr, torch.Tensor):
            attr_np = attr.cpu().numpy()
        elif isinstance(attr, list):
            attr_np = np.array(attr)
        else:
            raise TypeError(f"Unexpected attribute type: {type(attr)}")

        # Get outputs from first_model (list)
        first_outputs = model.first_model(image_tensor)

        # Handle both class and concept outputs
        if isinstance(first_outputs, list):
            class_logits = first_outputs[0]
            concept_logits = first_outputs
            concept_logits = torch.cat(concept_logits, dim=1)
        else:
            # fallback for dict-based outputs (old style)
            class_logits = first_outputs["class"]
            concept_logits = first_outputs["concepts"]

        pred_concepts = torch.sigmoid(concept_logits).detach().cpu().numpy()

        pred_binary = (pred_concepts > 0.5).astype(int)

        min_len = min(len(attr_np), len(pred_binary))
        attr_np = attr_np[:min_len]
        pred_binary = pred_binary[:min_len]

        correct = np.sum(attr_np == pred_binary)
        concept_acc = correct / len(attr_np)
        c_acc += concept_acc

        print(f"\nSample {i}: Concept prediction accuracy = {concept_acc*100:.2f}%")

        # Predict class
        with torch.no_grad():
            print("concept_logits shape:", concept_logits.shape)
            print("expected sec_model input:", model.sec_model.linear1.in_features)

            class_output = model.sec_model(concept_logits)
            pred_class_idx = class_output.argmax(dim=1).item()
            pred_class_name = class_names[pred_class_idx]
            true_class_name = class_names[label]

        if pred_class_name == true_class_name:
            acc += 1

        print(f"True class: {true_class_name} | Predicted class: {pred_class_name}")

        # -----------------------------------------------------------------
        # Extract and visualize top concept weights for predicted class
        # -----------------------------------------------------------------
        if hasattr(model.sec_model, "linear"):
            w_cls = model.sec_model.linear.weight[pred_class_idx]  # [n_attributes]
        else:
            # Fallback: locate the first Linear layer
            w_cls = None
            for m in model.sec_model.modules():
                if isinstance(m, torch.nn.Linear):
                    w_cls = m.weight[pred_class_idx]
                    break

        if w_cls is not None:
            w_sorted, idx_sorted = torch.sort(w_cls.abs(), descending=True)
            topk = 5
            print(f"\nTop {topk} concept weights for class '{pred_class_name}':")
            for j in range(topk):
                c_idx = idx_sorted[j].item()
                w_val = w_cls[c_idx].item()
                c_name = concepts[c_idx] if c_idx < len(concepts) else f"Concept_{c_idx}"
                print(f"  [{j+1}] {c_name:45s} → weight={w_val:+.4f}")

            # --- Plot top-k weights ---
            fig_w, ax_w = plt.subplots(figsize=(6, 3))
            sns.barplot(
                x=w_sorted[:topk].detach().cpu().numpy(),
                y=[concepts[idx] for idx in idx_sorted[:topk].cpu().numpy()],
                ax=ax_w
            )
            ax_w.set_title(f"Top {topk} Concept Weights\nPred: {pred_class_name} | True: {true_class_name}")
            plt.tight_layout()
            plt.savefig(f"weights_sample_{i}_class_{pred_class_name}.png", dpi=150)
            plt.close(fig_w)
        else:
            print("Could not locate classifier weight layer for concept visualization.")

        # -----------------------------------------------------------------
        # Gradient visualization for top-2 concept logits
        # -----------------------------------------------------------------
        top2, c_idcs = torch.topk(concept_logits.squeeze(), 2)
        image_tensor.grad = None
        top2[0].backward(retain_graph=True)
        w1 = image_tensor.grad.clone()

        image_tensor.grad = None
        top2[1].backward()
        w2 = image_tensor.grad.clone()

        sns.set_style("white")
        for logit, c_idx, w in zip(top2, c_idcs, [w1[0], w2[0]]):
            extracted_concept = (
                concepts[int(c_idx)] if int(c_idx) < len(concepts) else f"Concept_{c_idx}"
            )
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))
            ax1.imshow(to_numpy_img_display(to_numpy_img(img_rgb)))
            ax1.set_title(f"True: {true_class_name}\nPred: {pred_class_name}")
            ax1.axis("off")

            ax2.imshow(grad_to_img(img_6ch, w))
            ax2.set_title(f"{extracted_concept}\nConcept acc: {concept_acc:.2f}")
            ax2.axis("off")

            plt.suptitle(f"Sample {i} | Logit {logit.item():.3f}", fontsize=10)
            plt.tight_layout()
            save_path = f"sample_{i}_concept_{c_idx}_pred_{pred_class_name}.png"
            fig.set_size_inches(min(fig.get_size_inches()[0], 12), fig.get_size_inches()[1])
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

    print("\n======================================")
    print(f"Average Concept Prediction Accuracy: {(c_acc/num_samples)*100:.2f}%")
    print(f"Average Class Prediction Accuracy: {(acc/num_samples)*100:.2f}%")
    print("======================================\n")


# -------------------------------------------------------------------------
# 5. Main Entry
# -------------------------------------------------------------------------
if __name__ == "__main__":
    yaml_file = os.path.join(os.path.dirname(__file__), "bcos.yaml")
    config = load_config(yaml_file)

    model_path = '/h/sneharao/CBM/Joint0.01Model__Seed1_SG/outputs/best_model_1.pth'
    model = load_fine_tuned_model(config, model_path)

    print("Loading test dataset...")
    test_dataset = load_test_data(config["test_dir"])
    print(f"Loaded {len(test_dataset)} samples.")

    class_names = load_cub_class_names('/h/sneharao/CBM/CUB_200_2011/CUB_200_2011/classes.txt')
    concepts = load_cub_concept_names('/h/sneharao/CBM/CUB_200_2011/attributes.txt')

    visualize_model(model, test_dataset, class_names, concepts, num_samples=8)
