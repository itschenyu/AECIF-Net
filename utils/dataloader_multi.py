import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from utils.utils import preprocess_input, cvtColor

class SegmentationDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, train, dataset_path):
        super(SegmentationDataset, self).__init__()
        self.annotation_lines   = annotation_lines
        self.length             = len(annotation_lines)
        self.input_shape        = input_shape
        self.num_classes        = num_classes
        self.train              = train
        self.dataset_path       = dataset_path

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        annotation_line = self.annotation_lines[index]
        name            = annotation_line.split()[0]

        jpg         = Image.open(os.path.join(os.path.join(self.dataset_path, "VOC2007/JPEGImages"), name + ".jpg"))
        png_e         = Image.open(os.path.join(os.path.join(self.dataset_path, "VOC2007/SegmentationClass/element"), name + ".png"))
        png_d         = Image.open(os.path.join(os.path.join(self.dataset_path, "VOC2007/SegmentationClass/defect"), name + ".png"))
        jpg, png_e, png_d    = self.get_random_data(jpg, png_e, png_d, self.input_shape, random = self.train)

        jpg         = np.transpose(preprocess_input(np.array(jpg, np.float64)), [2,0,1])
        png_e         = np.array(png_e)
        png_e[png_e >= self.num_classes[0]] = self.num_classes[0]
        png_d         = np.array(png_d)
        png_d[png_d >= self.num_classes[1]] = self.num_classes[1]
        seg_labels_e  = np.eye(self.num_classes[0] + 1)[png_e.reshape([-1])]
        seg_labels_e  = seg_labels_e.reshape((int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes[0] + 1))
        seg_labels_d  = np.eye(self.num_classes[1] + 1)[png_d.reshape([-1])]
        seg_labels_d  = seg_labels_d.reshape((int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes[1] + 1))        
        return jpg, png_e, png_d, seg_labels_e, seg_labels_d

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, image, label, label_1, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.3, random=True):
        image   = cvtColor(image)
        label   = Image.fromarray(np.array(label))
        label_1 = Image.fromarray(np.array(label_1))
        iw, ih  = image.size
        h, w    = input_shape

        if not random:
            iw, ih  = image.size
            scale   = min(w/iw, h/ih)
            nw      = int(iw*scale)
            nh      = int(ih*scale)

            image       = image.resize((nw,nh), Image.BICUBIC)
            new_image   = Image.new('RGB', [w, h], (128,128,128))
            new_image.paste(image, ((w-nw)//2, (h-nh)//2))

            label       = label.resize((nw,nh), Image.NEAREST)
            new_label   = Image.new('L', [w, h], (0))
            new_label.paste(label, ((w-nw)//2, (h-nh)//2))

            label_1     = label_1.resize((nw,nh), Image.NEAREST)
            new_label_1 = Image.new('L', [w, h], (0))
            new_label_1.paste(label_1, ((w-nw)//2, (h-nh)//2))

            return new_image, new_label, new_label_1

        new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
        scale = self.rand(0.5, 2)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)
        label = label.resize((nw,nh), Image.NEAREST)
        label_1 = label_1.resize((nw,nh), Image.NEAREST)
        
        flip = self.rand()<.5
        if flip: 
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
            label_1 = label_1.transpose(Image.FLIP_LEFT_RIGHT)
        
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_label = Image.new('L', (w,h), (0))
        new_label_1 = Image.new('L', (w,h), (0))        
        new_image.paste(image, (dx, dy))
        new_label.paste(label, (dx, dy))
        new_label_1.paste(label_1, (dx, dy))
        image = new_image
        label = new_label
        label_1 = new_label_1

        image_data      = np.array(image, np.uint8)
        blur = self.rand() < 0.25
        if blur: 
            image_data = cv2.GaussianBlur(image_data, (5, 5), 0)

        rotate = self.rand() < 0.25
        if rotate: 
            center      = (w // 2, h // 2)
            rotation    = np.random.randint(-10, 11)
            M           = cv2.getRotationMatrix2D(center, -rotation, scale=1)
            image_data  = cv2.warpAffine(image_data, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=(128,128,128))
            label       = cv2.warpAffine(np.array(label, np.uint8), M, (w, h), flags=cv2.INTER_NEAREST, borderValue=(0))
            label_1     = cv2.warpAffine(np.array(label_1, np.uint8), M, (w, h), flags=cv2.INTER_NEAREST, borderValue=(0))

        r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        hue, sat, val   = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype           = image_data.dtype
        x       = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)
        
        return image_data, label, label_1

def seg_dataset_collate(batch):
    images      = []
    pngs_e        = []
    pngs_d        = []
    seg_labels_e  = []
    seg_labels_d  = []
    for img, png_e, png_d, labels_e, labels_d in batch:
        images.append(img)
        pngs_e.append(png_e)
        pngs_d.append(png_d)
        seg_labels_e.append(labels_e)
        seg_labels_d.append(labels_d)
    images      = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    pngs_e        = torch.from_numpy(np.array(pngs_e)).long()
    pngs_d        = torch.from_numpy(np.array(pngs_d)).long()
    seg_labels_e  = torch.from_numpy(np.array(seg_labels_e)).type(torch.FloatTensor)
    seg_labels_d  = torch.from_numpy(np.array(seg_labels_d)).type(torch.FloatTensor)
    return images, pngs_e, pngs_d, seg_labels_e, seg_labels_d
