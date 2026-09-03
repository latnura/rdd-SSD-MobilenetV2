import torch
from torchvision import transforms
import cv2
import numpy as np
import types
import random

def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]

def jaccard_numpy(box_a, box_b):
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) * (box_a[:, 3]-box_a[:, 1]))
    area_b = ((box_b[2]-box_b[0]) * (box_b[3]-box_b[1]))
    union = area_a + area_b - inter
    return inter / union

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None):
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
        return img, boxes, labels

class Lambda(object):
    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img, boxes=None, labels=None):
        return self.lambd(img, boxes, labels)

class ConvertFromInts(object):
    def __call__(self, image, boxes=None, labels=None):
        return image.astype(np.float32), boxes, labels

class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), boxes, labels

class ToAbsoluteCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, _ = image.shape
        if boxes is not None and len(boxes) > 0:
            boxes[:, 0] *= width
            boxes[:, 2] *= width
            boxes[:, 1] *= height
            boxes[:, 3] *= height
        return image, boxes, labels

class ToPercentCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, _ = image.shape
        if boxes is not None and len(boxes) > 0:
            boxes[:, 0] /= width
            boxes[:, 2] /= width
            boxes[:, 1] /= height
            boxes[:, 3] /= height
        return image, boxes, labels

class Resize(object):
    def __init__(self, size=300):
        self.size = size

    def __call__(self, image, boxes=None, labels=None):
        image = cv2.resize(image, (self.size, self.size))
        return image, boxes, labels

class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(0, 2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)
        return image, boxes, labels

class RandomHue(object):
    def __init__(self, delta=18.0):
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(0, 2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, boxes, labels

class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1), (1, 0, 2),
                      (1, 2, 0), (2, 0, 1), (2, 1, 0))

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(0, 2):
            # swap = self.perms[random.randint(0, len(self.perms) - 1)]
            swap = random.choice(self.perms)
            image = SwapChannels(swap)(image)
        return image, boxes, labels

class ConvertColor(object):
    def __init__(self, current, transform):
        self.current = current
        self.transform = transform

    def __call__(self, image, boxes=None, labels=None):
        code = {
            ('BGR', 'HSV'): cv2.COLOR_BGR2HSV,
            ('RGB', 'HSV'): cv2.COLOR_RGB2HSV,
            ('BGR', 'RGB'): cv2.COLOR_BGR2RGB,
            ('HSV', 'BGR'): cv2.COLOR_HSV2BGR,
            ('HSV', 'RGB'): cv2.COLOR_HSV2RGB
        }.get((self.current, self.transform), None)
        if code is None:
            raise NotImplementedError
        image = cv2.cvtColor(image, code)
        return image, boxes, labels

class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(0, 2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image, boxes, labels

class RandomBrightness(object):
    def __init__(self, delta=32):
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(0, 2):
            image += random.uniform(-self.delta, self.delta)
        return image, boxes, labels

class ToCV2Image(object):
    def __call__(self, tensor, boxes=None, labels=None):
        return tensor.cpu().numpy().astype(np.float32).transpose((1, 2, 0)), boxes, labels

class ToTensor(object):
    def __call__(self, cvimage, boxes=None, labels=None):
        return torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1), boxes, labels

class RandomSampleCrop(object):
    def __init__(self):
        self.sample_options = [
            None,  # No crop
            (0.1, 0.3),
            (0.3, 0.5),
            (0.7, 0.9),
            (0.9, 1.0)
        ]

    def __call__(self, image, boxes=None, labels=None):
        if boxes is None or len(boxes) == 0:
            return image, boxes, labels
        height, width, _ = image.shape
        while True:
            mode = random.choice(self.sample_options)
            if mode is None:
                return image, boxes, labels
            min_iou, max_iou = mode
            for _ in range(50):
                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)
                if h / w < 0.5 or h / w > 2:
                    continue
                left = random.uniform(0, width - w)
                top = random.uniform(0, height - h)
                rect = np.array([int(left), int(top), int(left+w), int(top+h)])
                overlap = jaccard_numpy(boxes, rect)
                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue
                current_image = image[rect[1]:rect[3], rect[0]:rect[2], :]
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])
                mask = m1 * m2
                if not mask.any():
                    continue
                current_boxes = boxes[mask, :].copy()
                current_labels = labels[mask]
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2], rect[:2]) - rect[:2]
                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:], rect[2:]) - rect[:2]
                return current_image, current_boxes, current_labels

class Expand(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, image, boxes, labels):
        if random.randint(0, 2):
            return image, boxes, labels
        height, width, depth = image.shape
        ratio = random.uniform(1, 4)
        left = random.uniform(0, width*ratio - width)
        top = random.uniform(0, height*ratio - height)
        expand_image = np.zeros((int(height*ratio), int(width*ratio), depth), dtype=image.dtype)
        expand_image[:, :, :] = self.mean
        expand_image[int(top):int(top+height), int(left):int(left+width)] = image
        if boxes is not None and len(boxes) > 0:
            boxes = boxes.copy()
            boxes[:, :2] += (int(left), int(top))
            boxes[:, 2:] += (int(left), int(top))
        return expand_image, boxes, labels

class RandomMirror(object):
    def __call__(self, image, boxes, labels):
        _, width, _ = image.shape
        if random.randint(0, 2):
            image = image[:, ::-1]
            if boxes is not None and len(boxes) > 0:
                boxes = boxes.copy()
                boxes[:, 0::2] = width - boxes[:, 2::-2]
        return image, boxes, labels

class SwapChannels(object):
    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        return image[:, :, self.swaps]

class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(current="RGB", transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='RGB'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, image, boxes, labels):
        im = image.copy()
        im, boxes, labels = self.rand_brightness(im, boxes, labels)
        if random.randint(0, 2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im, boxes, labels = distort(im, boxes, labels)
        return self.rand_light_noise(im, boxes, labels)

# Transformasi untuk training SSD
train_transform = Compose([
    ConvertFromInts(),
    ToAbsoluteCoords(),
    PhotometricDistort(),
    Expand([0, 0, 0]),
    RandomSampleCrop(),
    RandomMirror(),
    ToPercentCoords(),
    Resize(300),
    SubtractMeans([0, 0, 0]),  # sesuaikan kalau kamu pakai mean RGB tertentu
    ToTensor()
])

# Transformasi untuk validasi SSD
val_transform = Compose([
    ConvertFromInts(),
    Resize(300),
    SubtractMeans([0, 0, 0]),  # sesuaikan kalau kamu punya mean RGB
    ToTensor()
])
