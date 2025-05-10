from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
import os

# Load the dataset (both 'train' and 'validation' splits)
ds = load_dataset("clane9/imagenet-100")

# Define the root directory to save the images
save_root = os.path.abspath("./imagenet100")

# Iterate through 'train' and 'validation' splits
for split in ['train', 'validation']:
    split_ds = ds[split]
    print(f"\n🔄 Saving images from '{split}' split, total: {len(split_ds)} samples...")

    for i, sample in tqdm(enumerate(split_ds), total=len(split_ds), desc=f"Saving {split}"):
        label = sample['label']
        class_name = split_ds.features['label'].int2str(label)
        img = sample['image']

        # Create the directory: imagenet100/train/class_name/
        save_dir = os.path.join(save_root, split, class_name)
        os.makedirs(save_dir, exist_ok=True)

        # Save the image as a JPEG file
        img_path = os.path.join(save_dir, f"{i}.jpg")
        img.save(img_path)

    print(f"[✔] Saved {len(split_ds)} images from '{split}' split.")

print("\n✅ All images have been saved to: ./imagenet100/")