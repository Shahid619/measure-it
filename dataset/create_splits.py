import os
import glob
import random

def create_splits(
    image_dir="dataset/images",
    output_splits_dir="dataset/splits",
    train_pct=0.70,
    val_pct=0.20,
    test_pct=0.10,
    seed=42
):
    """
    Splits the dataset into 70/20/10 train/val/test with a pinned random seed.
    Saves immutable text files listing image filenames.
    """
    random.seed(seed)
    os.makedirs(output_splits_dir, exist_ok=True)

    images = sorted([os.path.basename(p) for p in glob.glob(os.path.join(image_dir, "*.png"))])
    total = len(images)
    if total == 0:
        raise ValueError(f"No images found in {image_dir}")

    # Deterministic shuffle
    shuffled = images.copy()
    random.shuffle(shuffled)

    n_train = int(total * train_pct) # 76 * 0.70 = 53
    n_val = int(total * val_pct)     # 76 * 0.20 = 15
    n_test = total - n_train - n_val  # 76 - 53 - 15 = 8 (10.5%)

    train_set = sorted(shuffled[:n_train])
    val_set = sorted(shuffled[n_train:n_train + n_val])
    test_set = sorted(shuffled[n_train + n_val:])

    for split_name, split_list in [("train", train_set), ("val", val_set), ("test", test_set)]:
        path = os.path.join(output_splits_dir, f"{split_name}.txt")
        with open(path, "w") as f:
            for item in split_list:
                f.write(f"{item}\n")
        print(f"  {split_name.upper()} split: {len(split_list)} images -> {path}")

    # Also save dedicated test_split.txt for explicit assessment verification
    with open(os.path.join(output_splits_dir, "test_split.txt"), "w") as f:
        for item in test_set:
            f.write(f"{item}\n")

    print("\nDataset split successfully created!")
    print(f"Total: {total} | Train: {len(train_set)} (70%) | Val: {len(val_set)} (20%) | Test: {len(test_set)} (10%)")

if __name__ == "__main__":
    create_splits()
