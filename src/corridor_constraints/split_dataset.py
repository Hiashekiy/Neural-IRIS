import numpy as np
import os

def split_and_save_dataset(input_file, out_dir="data/splits", train_ratio=0.8, val_ratio=0.1):
    print(f"Loading full dataset from {input_file}...")
    data = np.load(input_file)
    patches = data['patches']
    labels = data['labels']
    
    total_samples = len(patches)
    print(f"Total samples loaded: {total_samples}")
    
    # 设定固定的随机种子，确保每次切分结果完全一致
    np.random.seed(42)
    indices = np.random.permutation(total_samples)
    
    train_end = int(total_samples * train_ratio)
    val_end = int(total_samples * (train_ratio + val_ratio))
    
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    print("Splitting dataset...")
    train_patches, train_labels = patches[train_idx], labels[train_idx]
    val_patches, val_labels = patches[val_idx], labels[val_idx]
    test_patches, test_labels = patches[test_idx], labels[test_idx]
    
    # 确保输出目录存在
    os.makedirs(out_dir, exist_ok=True)
    
    train_file = os.path.join(out_dir, "train_iris.npz")
    val_file = os.path.join(out_dir, "val_iris.npz")
    test_file = os.path.join(out_dir, "test_iris.npz")
    
    print("Saving splits to disk...")
    np.savez_compressed(train_file, patches=train_patches, labels=train_labels)
    np.savez_compressed(val_file, patches=val_patches, labels=val_labels)
    np.savez_compressed(test_file, patches=test_patches, labels=test_labels)
    
    print("="*40)
    print("Dataset Split Summary:")
    print(f"Train set: {len(train_patches):>8} samples -> {train_file}")
    print(f"Val set:   {len(val_patches):>8} samples -> {val_file}")
    print(f"Test set:  {len(test_patches):>8} samples -> {test_file}")
    print("="*40)

if __name__ == "__main__":
    split_and_save_dataset("data/filtered_iris_dataset.npz")