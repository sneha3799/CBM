from dataset import load_data  # or the correct import path to your load_data function

# ✅ Use actual paths to your CUB pickled data
pkl_paths = [
    "/Users/sneha/Downloads/CBMs/train.pkl"  # replace with your actual file
]

loader = load_data(
    pkl_paths=pkl_paths,
    use_attr=True,
    no_img=False,
    batch_size=4,
    uncertain_label=False,
    n_class_attr=2,
    image_dir="images",
    resampling=False,
    resol=299
)

# Grab one batch
batch = next(iter(loader))

# Depending on your dataset settings:
if len(batch) == 3:
    images, labels, attrs = batch
else:
    images, labels = batch

print("Image batch shape:", images.shape)
