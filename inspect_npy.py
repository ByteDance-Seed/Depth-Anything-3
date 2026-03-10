import os
import numpy as np
import matplotlib.pyplot as plt

def inspect_data():
    dir_path = "/Users/aweise/dev/sunscape2/sunscape2/DepthAnything3MLX/Tests/DepthAnything3MLXTests/"
    input_path = os.path.join(dir_path, "input_images.npy")
    depth_path = os.path.join(dir_path, "expected_depth.npy")

    if not os.path.exists(input_path) or not os.path.exists(depth_path):
        print("Files not found. Please ensure they exist.")
        return

    images = np.load(input_path)
    depths = np.load(depth_path)

    print(f"Loaded input_images with shape: {images.shape}")
    print(f"Loaded expected_depth with shape: {depths.shape}")

    # input_images should be (1, N, 518, 518, 3)
    # expected_depth should be (N, 518, 518)
    if len(images.shape) == 5:
        images = images[0]

    N = images.shape[0]

    # De-normalize image for plotting
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    fig, axes = plt.subplots(N, 2, figsize=(10, 4 * N))
    
    # Handle the case where N=1 so axes behavior is consistent
    if N == 1:
        axes = [axes]

    for i in range(N):
        img = images[i]
        # De-normalize and clip
        img_visual = np.clip(img * std + mean, 0, 1)
        depth = depths[i]

        ax_img = axes[i][0]
        ax_depth = axes[i][1]

        ax_img.imshow(img_visual)
        ax_img.set_title(f"Input Image {i+1}")
        ax_img.axis('off')

        depth_plot = ax_depth.imshow(depth, cmap='inferno')
        ax_depth.set_title(f"Expected Depth {i+1}")
        ax_depth.axis('off')
        fig.colorbar(depth_plot, ax=ax_depth, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    inspect_data()
