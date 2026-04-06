import numpy as np
import os
import argparse

def filter_dataset(input_path, output_path):
    print(f"Loading dataset from {input_path}...")
    try:
        data = np.load(input_path)
        patches = data['patches']
        labels = data['labels']
    except Exception as e:
        print(f"Failed to load {input_path}: {e}")
        return

    num_samples = len(patches)
    print(f"Total samples before filtering: {num_samples}")

    # 解包标签: dx, dy, a, b, sin_theta, cos_theta
    # 其中 c_x = 64 + dx, c_y = 64 + dy (假设 patch_size = 128)
    # 图片中心点为 img_c = (64, 64)
    # 所以 img_c 相对于椭圆中心的向量 v = (-dx, -dy)
    dx = labels[:, 0]
    dy = labels[:, 1]
    a = labels[:, 2]
    b = labels[:, 3]
    sin_theta = labels[:, 4]
    cos_theta = labels[:, 5]

    # 将 v = (-dx, -dy) 旋转回椭圆基准轴系
    # R^T * v = [cos_theta, sin_theta; -sin_theta, cos_theta] * [-dx; -dy]
    v_rot_x = -dx * cos_theta - dy * sin_theta
    v_rot_y = dx * sin_theta - dy * cos_theta

    # 计算目标点(图片中心点)在椭圆坐标系里的距离方程：(x/a)^2 + (y/b)^2 <= 1
    # 增加一点点宽容度(1.0 + 1e-4)处理计算精度问题
    inside_ellipse_metric = (v_rot_x / a)**2 + (v_rot_y / b)**2
    valid_mask = inside_ellipse_metric <= 1.0001

    filtered_patches = patches[valid_mask]
    filtered_labels = labels[valid_mask]
    
    num_filtered = len(filtered_patches)
    print(f"Total samples after filtering: {num_filtered}")
    print(f"Removed {num_samples - num_filtered} samples.")

    # 建立 data 目录并保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(output_path, patches=filtered_patches, labels=filtered_labels)
    print(f"Filtered dataset saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='full_iris_dataset.npz', help='Path to the full dataset')
    parser.add_argument('--output', type=str, default='data/filtered_iris_dataset.npz', help='Path to save the filtered dataset')
    args = parser.parse_args()
    
    filter_dataset(args.input, args.output)
