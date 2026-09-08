import numpy as np
import torch
import cv2
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

import config_reference as config

class ModelWrapper(torch.nn.Module):
    def __init__(self, model, metadata):
        super().__init__()
        self.model = model
        self.metadata = metadata
        
    def forward(self, x):
        cat_logits, reg_out, features = self.model(x, self.metadata)
        return cat_logits

def generate_gradcam(model, image, metadata, device, target_layer=None):
    if target_layer is None:
        # Fallback to a generic assumption if target_layer is not provided
        target_layer = [list(model.children())[0][-1]] 
    else:
        target_layer = [target_layer]
        
    image_t = image.unsqueeze(0).to(device)
    meta_t = metadata.unsqueeze(0).to(device)
    
    wrapper = ModelWrapper(model, meta_t)
    
    with GradCAM(model=wrapper, target_layers=target_layer, use_cuda=(device.type == 'cuda')) as cam:
        grayscale_cam = cam(input_tensor=image_t, targets=None)
        heatmap = grayscale_cam[0, :]
        return heatmap

def save_gradcam_image(heatmap, output_path, size=256):
    import matplotlib.pyplot as plt
    # Upscale
    heatmap_resized = cv2.resize(heatmap, (size, size))
    # Apply colormap
    colormap = plt.get_cmap('jet')
    rgba_img = colormap(heatmap_resized) # Output is [0, 1] RGBA
    
    # Set alpha channel proportional to intensity
    rgba_img[:, :, 3] = heatmap_resized
    
    # Convert to 0-255
    rgba_img_255 = (rgba_img * 255).astype(np.uint8)
    
    img = Image.fromarray(rgba_img_255, mode="RGBA")
    img.save(output_path)

def generate_composite_image(image_tensor, output_path, size=256):
    # image_tensor: [5, 64, 64]
    ir1 = image_tensor[0].cpu().numpy()
    wv = image_tensor[1].cpu().numpy()
    pmw = image_tensor[2].cpu().numpy()
    vis = image_tensor[3].cpu().numpy()
    vis_mask = image_tensor[4].cpu().numpy()
    
    # R = Inverted IR1
    r = -1.0 * ir1
    # G = WV
    g = wv
    # B = VIS where mask=1, else PMW
    b = np.where(vis_mask == 1, vis, pmw)
    
    # Normalize channels to 0-255
    def norm(c):
        c_min, c_max = c.min(), c.max()
        if c_max > c_min:
            c = (c - c_min) / (c_max - c_min)
        return (c * 255).astype(np.uint8)
        
    r_norm = norm(r)
    g_norm = norm(g)
    b_norm = norm(b)
    
    rgb = np.stack([r_norm, g_norm, b_norm], axis=-1)
    
    img = Image.fromarray(rgb, mode="RGB")
    img = img.resize((size, size), Image.Resampling.BILINEAR)
    img.save(output_path)
