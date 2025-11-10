import os
from collections import Counter

# Path to your CUB dataset folder
cub_root = '/h/sneharao/CBM/CUB_200_2011/CUB_200_2011/'

# Files containing class labels
labels_file = os.path.join(cub_root, 'image_class_labels.txt')

# Load image → class mappings
class_counts = Counter()
with open(labels_file, 'r') as f:
    for line in f:
        img_id, class_id = line.strip().split()
        class_counts[int(class_id)] += 1

# Print summary
print(f"Total classes: {len(class_counts)}")
print(f"Total images: {sum(class_counts.values())}")

# Print a few examples
for cls_id in sorted(class_counts.keys()):
    print(f"Class {cls_id}: {class_counts[cls_id]} images")

# Find extremes
max_class = max(class_counts, key=class_counts.get)
min_class = min(class_counts, key=class_counts.get)
print(f"\nMost images: Class {max_class} → {class_counts[max_class]} images")
print(f"Fewest images: Class {min_class} → {class_counts[min_class]} images")
