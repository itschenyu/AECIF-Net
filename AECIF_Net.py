import colorsys
import copy
import time
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torch import nn
from collections import Counter

from nets.AECIF_Net import AECIF_Net
from utils.utils import cvtColor, preprocess_input, resize_image, show_config

class HRnet_Segmentation(object):
    _defaults = {
        "model_path": 'logs/best_epoch_weights.pth',    # weight path
        "num_classes"       : [7, 2],   # [element, defect]
        "backbone"          : "hrnetv2_w48",    # backbone name
        "input_shape"       : [520, 520],      # [h, w]
        "mix_type"          : 0,
        "cuda"              : True,
    }

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
        #---------------------------------------------------#
        if self.num_classes[0] <= 21:
            self.colors = [ (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128),
                            (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128),
                            (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128),
                            (128, 64, 12)]

            self.colors_1 = [ (128, 128, 128), (128, 64, 0), (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128),
                            (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128),
                            (64, 128, 128), (192, 128, 128), (0, 64, 0),  (0, 192, 0), (128, 192, 0), (0, 64, 128),
                            (128, 64, 12)]
        else:
            hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.generate()
        
        show_config(**self._defaults)
                    
    def generate(self, onnx=False):
        self.net = AECIF_Net(num_classes=self.num_classes, backbone=self.backbone, pretrained=False)
        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state_dict = torch.load(self.model_path, map_location=device)
        weights_dict = {}
        for k, v in state_dict.items():
            new_k = k.replace('module.', '') if 'module' in k else k
            weights_dict[new_k] = v
        self.net.load_state_dict(weights_dict)
        self.net    = self.net.eval()
        print('{} model, and classes loaded.'.format(self.model_path))
        if not onnx:
            if self.cuda:
                self.net = nn.DataParallel(self.net)
                self.net = self.net.cuda()

    def detect_image(self, image, count=False, name_classes=None):
        image       = cvtColor(image)
        old_img     = copy.deepcopy(image)
        orininal_h  = np.array(image).shape[0]
        orininal_w  = np.array(image).shape[1]
        image_data, nw, nh  = resize_image(image, (self.input_shape[1],self.input_shape[0]))
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
                
            pr_e = self.net(images)[0][0]
            pr_d = self.net(images)[1][0]            
            pr_e = F.softmax(pr_e.permute(1,2,0),dim = -1).cpu().numpy()
            pr_d = F.softmax(pr_d.permute(1,2,0),dim = -1).cpu().numpy()            
            pr_e = pr_e[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                    int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]
            pr_d = pr_d[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                    int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]
            pr_e = cv2.resize(pr_e, (orininal_w, orininal_h), interpolation = cv2.INTER_LINEAR)
            pr_d = cv2.resize(pr_d, (orininal_w, orininal_h), interpolation = cv2.INTER_LINEAR)
            pr_e = pr_e.argmax(axis=-1)
            pr_d = pr_d.argmax(axis=-1)

        if count:
            classes_nums_e        = np.zeros([self.num_classes[0]])
            classes_nums_d        = np.zeros([self.num_classes[1]])
            total_points_num    = orininal_h * orininal_w
            print('-' * 63)
            print("|%25s | %15s | %15s|"%("Key", "Value", "Ratio"))
            print('-' * 63)
            for i in range(self.num_classes[0]):
                num     = np.sum(pr_e == i)
                ratio   = num / total_points_num * 100
                if num > 0:
                    print("|%25s | %15s | %14.2f%%|"%(str(name_classes[0][i]), str(num), ratio))
                    print('-' * 63)
                classes_nums_e[i] = num
            print("classes_nums_e:", classes_nums_e)

            for i in range(self.num_classes[1]):
                num     = np.sum(pr_d == i)
                ratio   = num / total_points_num * 100
                if num > 0:
                    print("|%25s | %15s | %14.2f%%|"%(str(name_classes[1][i]), str(num), ratio))
                    print('-' * 63)
                classes_nums_d[i] = num
            print("classes_nums_d:", classes_nums_d)

        if self.mix_type == 0:
            seg_img_e = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr_e, [-1])], [orininal_h, orininal_w, -1])
            seg_img_d = np.reshape(np.array(self.colors_1, np.uint8)[np.reshape(pr_d, [-1])], [orininal_h, orininal_w, -1])
            image_e   = Image.fromarray(np.uint8(seg_img_e))
            image_d   = Image.fromarray(np.uint8(seg_img_d))
            image_e   = Image.blend(old_img, image_e, 0.7)
            image_d   = Image.blend(old_img, image_d, 0.7)

            contours, hierarchy = cv2.findContours(np.uint8(pr_d), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contour_map = np.array(image_e)
            cv2.drawContours(contour_map, contours, -1, (255, 255, 255), thickness = 5)
            contour_map = Image.fromarray(np.uint8(contour_map))

        elif self.mix_type == 1:
            seg_img_e = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr_e, [-1])], [orininal_h, orininal_w, -1])
            seg_img_d = np.reshape(np.array(self.colors_1, np.uint8)[np.reshape(pr_d, [-1])], [orininal_h, orininal_w, -1])
            image_e   = Image.fromarray(np.uint8(seg_img_e))
            image_d   = Image.fromarray(np.uint8(seg_img_d))

        elif self.mix_type == 2:
            seg_img_e = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr_e, [-1])], [orininal_h, orininal_w, -1])
            seg_img_d = np.reshape(np.array(self.colors_1, np.uint8)[np.reshape(pr_d, [-1])], [orininal_h, orininal_w, -1])
            image_e   = Image.fromarray(np.uint8(seg_img_e))
            image_d   = Image.fromarray(np.uint8(seg_img_d))
        
        return image_e, image_d

    def get_FPS(self, image, test_interval):
        image       = cvtColor(image)
        image_data, nw, nh  = resize_image(image, (self.input_shape[1],self.input_shape[0]))
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
                
            pr_e = self.net(images)[0][0]
            pr_d = self.net(images)[1][0] 
            pr_e = F.softmax(pr_e.permute(1,2,0),dim = -1).cpu().numpy().argmax(axis=-1)
            pr_d = F.softmax(pr_d.permute(1,2,0),dim = -1).cpu().numpy().argmax(axis=-1)
            pr_e = pr_e[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                    int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]
            pr_d = pr_d[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                    int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]

        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                pr_e = self.net(images)[0][0]
                pr_d = self.net(images)[1][0] 
                pr_e = F.softmax(pr_e.permute(1,2,0),dim = -1).cpu().numpy().argmax(axis=-1)
                pr_d = F.softmax(pr_d.permute(1,2,0),dim = -1).cpu().numpy().argmax(axis=-1)
                pr_e = pr_e[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                        int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]
                pr_d = pr_d[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                        int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    def convert_to_onnx(self, simplify, model_path):
        import onnx
        self.generate(onnx=True)

        im                  = torch.zeros(1, 3, *self.input_shape).to('cpu')  # image size(1, 3, 512, 512) BCHW
        input_layer_names   = ["images"]
        output_layer_names  = ["output"]
        
        # Export the model
        print(f'Starting export with onnx {onnx.__version__}.')
        torch.onnx.export(self.net,
                        im,
                        f               = model_path,
                        verbose         = False,
                        opset_version   = 12,
                        training        = torch.onnx.TrainingMode.EVAL,
                        do_constant_folding = True,
                        input_names     = input_layer_names,
                        output_names    = output_layer_names,
                        dynamic_axes    = None)

        # Checks
        model_onnx = onnx.load(model_path)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model

        # Simplify onnx
        if simplify:
            import onnxsim
            print(f'Simplifying with onnx-simplifier {onnxsim.__version__}.')
            model_onnx, check = onnxsim.simplify(
                model_onnx,
                dynamic_input_shape=False,
                input_shapes=None)
            assert check, 'assert check failed'
            onnx.save(model_onnx, model_path)

        print('Onnx model save as {}'.format(model_path))

    def get_miou_png(self, image):
        image       = cvtColor(image)
        orininal_h  = np.array(image).shape[0]
        orininal_w  = np.array(image).shape[1]
        image_data, nw, nh  = resize_image(image, (self.input_shape[1],self.input_shape[0]))
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            pr_e = self.net(images)[0][0]
            pr_d = self.net(images)[1][0]
            pr_e = F.softmax(pr_e.permute(1,2,0),dim = -1).cpu().numpy()
            pr_d = F.softmax(pr_d.permute(1,2,0),dim = -1).cpu().numpy()
            pr_e = pr_e[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                    int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]
            pr_d = pr_d[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                    int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]        
            pr_e = cv2.resize(pr_e, (orininal_w, orininal_h), interpolation = cv2.INTER_LINEAR)
            pr_d = cv2.resize(pr_d, (orininal_w, orininal_h), interpolation = cv2.INTER_LINEAR)
            pr_e = pr_e.argmax(axis=-1)
            pr_d = pr_d.argmax(axis=-1)
    
        image_e = Image.fromarray(np.uint8(pr_e))
        image_d = Image.fromarray(np.uint8(pr_d))

        return image_e, image_d