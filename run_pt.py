import sys
import os
import torch
import numpy as np
from PIL import Image
from omegaconf import OmegaConf

sys.path.append("/Users/aweise/dev/sunscape2/da3conv/Depth-Anything-3-coreml/src")
from depth_anything_3.cfg import create_object

def load_image(path, size=(518, 518)):
    img = Image.open(path).convert('RGB').resize(size)
    img_np = np.array(img).astype(np.float32) / 255.0
    # normalize using ImageNet mean/std
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_np = (img_np - mean) / std
    return img_np

def run_pytorch():
    config_name = "da3-base"
    weights_path = "/Users/aweise/.cache/huggingface/hub/models--depth-anything--DA3-BASE/snapshots/f4a6c9b3c95e41c82048423d3493a81ec3fa810e/model.safetensors"
    
    cfg_path = f"/Users/aweise/dev/sunscape2/da3conv/Depth-Anything-3-coreml/src/depth_anything_3/configs/{config_name}.yaml"
    cfg = OmegaConf.load(cfg_path)
    model = create_object(cfg)
    model.eval()
    
    from safetensors.torch import load_file
    state_dict = load_file(weights_path)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            new_state_dict[k[6:]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict, strict=False)
    
    dir_path = "/Users/aweise/dev/sunscape2/sunscape2/DepthAnything3MLX/Tests/DepthAnything3MLXTests/"
    img1 = load_image(os.path.join(dir_path, "sv1.png"))
    img2 = load_image(os.path.join(dir_path, "sv2.png"))
    img3 = load_image(os.path.join(dir_path, "sv3.png"))
    
    # images_np: (1, N, H, W, 3)
    images_np = np.stack([img1, img2, img3])[None, ...]
    images_np.astype(np.float32).tofile(os.path.join(dir_path, "input_images.bin"))
    np.save(os.path.join(dir_path, "input_images.npy"), images_np.astype(np.float32))
    print("Saved input_images.bin and input_images.npy")
    
    images_pt = torch.from_numpy(images_np).permute(0, 1, 4, 2, 3).float()
    print("Running pytorch inference with shape", images_pt.shape)
    
    with torch.no_grad():
        output = model(images_pt, extrinsics=None, intrinsics=None, export_feat_layers=[], infer_gs=False, use_ray_pose=False)
        
    depth = output["depth"].numpy()[0] # (N, H, W)
    depth.astype(np.float32).tofile(os.path.join(dir_path, "expected_depth.bin"))
    np.save(os.path.join(dir_path, "expected_depth.npy"), depth.astype(np.float32))
    print("Saved expected_depth.bin and expected_depth.npy")

if __name__ == "__main__":
    run_pytorch()
