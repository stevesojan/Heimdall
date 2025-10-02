from vit import annotate_patterns_on_image, NUM_CLASSES
import torch
from torchvision import models
vit = models.vision_transformer.vit_b_16(weights=None)
vit.heads[0] = torch.nn.Linear(vit.heads[0].in_features, NUM_CLASSES)
vit.load_state_dict(torch.load('vit_candlestick.pth'))
annotate_patterns_on_image(vit, "candlestick_images/CDL3OUTSIDE_bearish_938.png", threshold=0.2)
