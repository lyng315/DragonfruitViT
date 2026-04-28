import os
import splitfolders
import torch
import random
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Reproducibility
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

# Environment Configuration
IS_KAGGLE = os.path.exists('/kaggle/input')
if IS_KAGGLE:
    ORIGINAL_DATASET_DIR = '/kaggle/input/datasets/linhngkhanh/dragonfruit/dataset'
    SPLIT_DATASET_DIR = '/kaggle/working/split_dataset'
    NUM_WORKERS = 2
else:
    ORIGINAL_DATASET_DIR = r"D:\DragonfruitViT\dataset"
    SPLIT_DATASET_DIR = r"D:\DragonfruitViT\split_dataset"
    NUM_WORKERS = 0

OUTPUTS_DIR = "outputs"

def get_class_distribution(directory):
    if not os.path.exists(directory):
        return {}
    distribution = {}
    for class_name in os.listdir(directory):
        class_path = os.path.join(directory, class_name)
        if os.path.isdir(class_path):
            count = len([f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            distribution[class_name] = count
    return distribution

def print_original_stats():
    print(f"\n{'='*20} ORIGINAL DATASET STATS {'='*20}")
    orig_dist = get_class_distribution(ORIGINAL_DATASET_DIR)
    if orig_dist:
        total_raw = sum(orig_dist.values())
        print(f"Total images: {total_raw}")
        for cls, count in orig_dist.items():
            print(f"  - {cls:10}: {count}")
    else:
        print(f"Dataset not found at: {ORIGINAL_DATASET_DIR}")

def split_data():
    print(f"\nSplitting data: {ORIGINAL_DATASET_DIR} -> {SPLIT_DATASET_DIR}")
    splitfolders.ratio(ORIGINAL_DATASET_DIR, output=SPLIT_DATASET_DIR,
                       seed=42, ratio=(0.8, 0.1, 0.1), group_prefix=None, move=False)
    print("Data split completed!")

def get_dataloaders(batch_size=32):
    train_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomRotation(degrees=15),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    eval_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dir = os.path.join(SPLIT_DATASET_DIR, 'train')
    val_dir   = os.path.join(SPLIT_DATASET_DIR, 'val')
    test_dir  = os.path.join(SPLIT_DATASET_DIR, 'test')

    print(f"\n{'='*20} SPLIT DATASET STATS {'='*20}")
    for name, path in [("TRAIN", train_dir), ("VAL", val_dir), ("TEST", test_dir)]:
        dist = get_class_distribution(path)
        print(f"  {name}: {sum(dist.values())} images")
        for cls, count in dist.items():
            print(f"    - {cls}: {count}")

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset   = datasets.ImageFolder(val_dir,   transform=eval_transforms)
    test_dataset  = datasets.ImageFolder(test_dir,  transform=eval_transforms)

    class_names = train_dataset.classes

    g = torch.Generator()
    g.manual_seed(42)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=NUM_WORKERS, worker_init_fn=seed_worker, generator=g)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,
                              num_workers=NUM_WORKERS, worker_init_fn=seed_worker, generator=g)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False,
                              num_workers=NUM_WORKERS, worker_init_fn=seed_worker, generator=g)

    return train_loader, val_loader, test_loader, class_names

def inspect_data(loader, class_names):
    """
    Comprehensive preprocessing sanity check.
    Covers: global stats, pixel trace, label mapping, tensor view, and visual comparison.
    """
    indices = [0, 1, 2, 3]
    m_val = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    s_val = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)

    # ------------------------------------------------------------------
    # MODULE 1: Full-dataset statistical scan (raw 0-255 vs normalized)
    # ------------------------------------------------------------------
    print(f"\nScanning {len(loader.dataset)} images for global statistics...")
    raw_0255_min, raw_0255_max, raw_0255_sum = float('inf'), float('-inf'), 0.0
    norm_min, norm_max, norm_sum = float('inf'), float('-inf'), 0.0
    total_pixels = 0

    for imgs, _ in loader:
        # After normalization
        norm_min = min(norm_min, imgs.min().item())
        norm_max = max(norm_max, imgs.max().item())
        norm_sum += imgs.sum().item()

        # Denormalize back to 0-255
        raw_01    = imgs.numpy() * s_val + m_val
        raw_0255  = raw_01 * 255.0
        raw_0255_min = min(raw_0255_min, raw_0255.min())
        raw_0255_max = max(raw_0255_max, raw_0255.max())
        raw_0255_sum += raw_0255.sum()

        total_pixels += imgs.numel()

    # ------------------------------------------------------------------
    # Print full diagnostic report
    # ------------------------------------------------------------------
    print(f"\n{'='*25} PREPROCESSING DIAGNOSTIC REPORT {'='*25}")

    # --- Size comparison ---
    sample_path, _ = loader.dataset.samples[0]
    with Image.open(sample_path) as tmp_img:
        orig_w, orig_h = tmp_img.size
    print("\n1. SIZE COMPARISON:")
    print(f"   [BEFORE] Original on disk : {orig_w}x{orig_h} (WxH)")
    print(f"   [AFTER]  Fed into model   : 224x224 (HxW) — Resize(256) + CenterCrop(224)")

    # --- Global stats ---
    print("\n2. GLOBAL PIXEL STATISTICS:")
    print(f"   [BEFORE] Raw (0-255)   : Min={raw_0255_min:7.2f}, Max={raw_0255_max:7.2f}, Mean={raw_0255_sum/total_pixels:7.2f}")
    print(f"   [AFTER]  Normalized    : Min={norm_min:7.4f}, Max={norm_max:7.4f}, Mean={norm_sum/total_pixels:7.4f}")

    # --- Single pixel trace ---
    print("\n3. SINGLE PIXEL TRACE (pixel at center 112,112 of first image):")
    with Image.open(sample_path).convert('RGB') as img_trace:
        w, h = img_trace.size
        new_w, new_h = (256, int(h * 256 / w)) if w < h else (int(w * 256 / h), 256)
        img_rsz = img_trace.resize((new_w, new_h))
        l, t    = (new_w - 224) // 2, (new_h - 224) // 2
        img_cp  = img_rsz.crop((l, t, l + 224, t + 224))
        p_0255  = np.array(img_cp.getpixel((112, 112)))

    r_orig, g_orig, b_orig = p_0255
    r_01, g_01, b_01       = r_orig / 255.0, g_orig / 255.0, b_orig / 255.0
    rm, gm, bm             = 0.485, 0.456, 0.406
    rs, gs, bs             = 0.229, 0.224, 0.225
    print(f"   Step 1 — Raw  (0-255)  : R={r_orig:3},     G={g_orig:3},     B={b_orig:3}")
    print(f"   Step 2 — ToTensor(0-1) : R={r_01:.4f}, G={g_01:.4f}, B={b_01:.4f}")
    print(f"   Step 3 — Normalized    : R={(r_01-rm)/rs:.4f}, G={(g_01-gm)/gs:.4f}, B={(b_01-bm)/bs:.4f}")

    # --- Label mapping ---
    print("\n4. LABEL MAPPING (class index -> class name):")
    for idx, name in enumerate(class_names):
        print(f"   - Index {idx} => {name}")

    print(f"\n{'='*80}")

    # ------------------------------------------------------------------
    # MODULE 5: Side-by-side visual (Original vs Augmented)
    # ------------------------------------------------------------------
    print("\n5. VISUAL COMPARISON (Original vs Augmented — saving preprocess_check.png)...")
    plt.figure(figsize=(16, 9))

    for i, idx in enumerate(indices):
        img_path, label_idx = loader.dataset.samples[idx]

        # Row 1: Original image from disk
        with Image.open(img_path).convert('RGB') as raw:
            plt.subplot(2, 4, i + 1)
            plt.imshow(raw)
            plt.title(f"Original: {class_names[label_idx]}")
            plt.axis('off')

        # Row 2: After full preprocessing + augmentation pipeline
        img_augmented, _ = loader.dataset[idx]
        proc_img = img_augmented.numpy() * s_val + m_val
        proc_img = proc_img.transpose((1, 2, 0))
        proc_img = np.clip(proc_img, 0, 1)

        plt.subplot(2, 4, i + 5)
        plt.imshow(proc_img)
        plt.title(f"Augmented 224x224: {class_names[label_idx]}")
        plt.axis('off')

    # ------------------------------------------------------------------
    # MODULE 6: Tensor structure visualization (AI view)
    # ------------------------------------------------------------------
    print("\n6. TENSOR STRUCTURE (AI View — 3-channel layout):")
    img_tensor, _ = loader.dataset[0]
    print(f"   Shape: {img_tensor.shape}  (Channels x Height x Width)")
    print("   Inspecting 2x2 patch at center (112, 112):")

    for c, color_name in enumerate(['RED   (Channel 0)', 'GREEN (Channel 1)', 'BLUE  (Channel 2)']):
        matrix_2x2 = img_tensor[c, 112:114, 112:114].numpy()
        print(f"     > {color_name}:")
        for row in matrix_2x2:
            print(f"       [ {row[0]:7.4f}  {row[1]:7.4f} ]")

    print("\n   => Each channel is a separate 2D matrix of real numbers.")
    print("      The model receives all 3 stacked together as a single tensor.")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, 'preprocess_check.png'))
    print("\n--> Saved visual comparison to: preprocess_check.png")
    plt.show()

if __name__ == "__main__":
    # ==========================================================================
    # DATA PREPARATION PIPELINE
    # ==========================================================================
    print_original_stats()
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    if not os.path.exists(SPLIT_DATASET_DIR):
        split_data()
    else:
        print(f"\nDirectory '{SPLIT_DATASET_DIR}' already exists — skipping split step.")

    train_loader, val_loader, test_loader, classes = get_dataloaders()

    print(f"\n{'='*20} INSPECTING TRAIN SET {'='*20}")
    inspect_data(train_loader, classes)

    print("\nData preprocessing and inspection completed. Ready for training.")
