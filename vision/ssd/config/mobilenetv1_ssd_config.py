import numpy as np
from vision.utils.box_utils import SSDSpec, SSDBoxSizes, generate_ssd_priors

# 1. RESOLUSI DIUBAH KE 512
image_size = 512 
image_mean = np.array([127, 127, 127])  # RGB layout
image_std = 128.0
iou_threshold = 0.45
center_variance = 0.1
size_variance = 0.2

# 2. FEATURE MAPS & ANCHOR BOXES DISESUAIKAN UNTUK 512
# Rasio pembesaran = 512 / 300 = ~1.7
specs = [
    # Format: SSDSpec(feature_map_size, shrinkage/stride, SSDBoxSizes(min, max), aspect_ratios)
    SSDSpec(32, 16, SSDBoxSizes(51, 102), [2, 3, 5]),
    SSDSpec(16, 32, SSDBoxSizes(179, 256), [2, 3, 5]),
    SSDSpec(8, 64, SSDBoxSizes(256, 333), [2, 3, 5]),
    SSDSpec(4, 128, SSDBoxSizes(333, 410), [2, 3]),
    SSDSpec(2, 256, SSDBoxSizes(410, 486), [2, 3]),
    SSDSpec(1, 512, SSDBoxSizes(486, 563), [2, 3])
]

priors = generate_ssd_priors(specs, image_size)