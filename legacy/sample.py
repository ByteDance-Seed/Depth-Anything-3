import glob, os, torch
from depth_anything_3.api import DepthAnything3


print("Loading model...")   
device = torch.device("cuda")
print("Device: ", device)

model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE")
model = model.to(device=device)
example_path = "assets/examples/SOH"
images = sorted(glob.glob(os.path.join(example_path, "*.png")))
prediction = model.inference(
    images,
)
# prediction.processed_images : [N, H, W, 3] uint8   array
print(prediction.processed_images.shape)
# prediction.depth            : [N, H, W]    float32 array
print(prediction.depth.shape)  
# prediction.conf             : [N, H, W]    float32 array
print(prediction.conf.shape)  
# prediction.extrinsics       : [N, 3, 4]    float32 array # opencv w2c or colmap format
print(prediction.extrinsics.shape)
# prediction.intrinsics       : [N, 3, 3]    float32 array
print(prediction.intrinsics.shape)