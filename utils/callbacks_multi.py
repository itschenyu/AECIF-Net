import os
import matplotlib
import torch
import torch.nn.functional as F
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import scipy.signal
import cv2
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from .utils import cvtColor, preprocess_input, resize_image
from .utils_metrics_multi import compute_mIoU

class LossHistory():
    def __init__(self, log_dir, model, input_shape):
        self.log_dir    = log_dir
        self.losses     = []
        self.losses_e   = []
        self.losses_d   = []
        self.val_loss   = []
        self.val_loss_e = []
        self.val_loss_d = []
        
        os.makedirs(self.log_dir)
        self.writer     = SummaryWriter(self.log_dir)
        try:
            dummy_input     = torch.randn(2, 3, input_shape[0], input_shape[1])
            self.writer.add_graph(model, dummy_input)
        except:
            pass

    def append_loss(self, epoch, loss, loss_e, loss_d, val_loss, val_loss_e, val_loss_d):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.losses.append(loss)
        self.losses_e.append(loss_e)
        self.losses_d.append(loss_d)
        self.val_loss.append(val_loss)
        self.val_loss_e.append(val_loss_e)
        self.val_loss_d.append(val_loss_d)

        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
            f.write(str(loss))
            f.write(", ")
            f.write(str(loss_e))
            f.write(", ")
            f.write(str(loss_d))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a') as f:
            f.write(str(val_loss))
            f.write(", ")
            f.write(str(val_loss_e))
            f.write(", ")
            f.write(str(val_loss_d))
            f.write("\n")

        self.writer.add_scalar('loss', loss, epoch)
        self.writer.add_scalar('val_loss', val_loss, epoch)
        self.loss_plot()

    def call_e(self):
        return self.losses_e

    def call_d(self):
        return self.losses_d

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth = 2, label='train loss')
        plt.plot(iters, self.val_loss, 'coral', linewidth = 2, label='val loss')
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15
            
            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle = '--', linewidth = 2, label='smooth train loss')
            plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle = '--', linewidth = 2, label='smooth val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))

        plt.cla()
        plt.close("all")

class EvalCallback():
    def __init__(self, net, input_shape, num_classes, image_ids, dataset_path, log_dir, cuda, \
            miou_out_path=".temp_miou_out", eval_flag=True, period=1):
        super(EvalCallback, self).__init__()
        
        self.net                = net
        self.input_shape        = input_shape
        self.num_classes        = num_classes
        self.image_ids          = image_ids
        self.dataset_path       = dataset_path
        self.log_dir            = log_dir
        self.cuda               = cuda
        self.miou_out_path      = miou_out_path
        self.eval_flag          = eval_flag
        self.period             = period
        
        self.image_ids          = [image_id.split()[0] for image_id in image_ids]
        self.mious_e      = [0]
        self.mious_d      = [0]
        self.epoches    = [0]
        if self.eval_flag:
            with open(os.path.join(self.log_dir, "epoch_miou.txt"), 'a') as f:
                f.write(str(0))
                f.write("\n")

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
    
    def on_epoch_end(self, epoch, model_eval):
        if epoch % self.period == 0 and self.eval_flag:
            self.net    = model_eval
            gt_dir_e      = os.path.join(self.dataset_path, "VOC2007/SegmentationClass/element")
            pred_dir_e    = os.path.join(self.miou_out_path, 'detection-results/element')
            gt_dir_d      = os.path.join(self.dataset_path, "VOC2007/SegmentationClass/defect")
            pred_dir_d    = os.path.join(self.miou_out_path, 'detection-results/defect')            
            if not os.path.exists(self.miou_out_path):
                os.makedirs(self.miou_out_path)
            if not os.path.exists(pred_dir_e):
                os.makedirs(pred_dir_e)
            if not os.path.exists(pred_dir_d):
                os.makedirs(pred_dir_d)
            print("Get miou.")
            for image_id in tqdm(self.image_ids):
                image_path  = os.path.join(self.dataset_path, "VOC2007/JPEGImages/"+image_id+".jpg")
                image       = Image.open(image_path)
                image_e       = self.get_miou_png(image)[0]
                image_e.save(os.path.join(pred_dir_e, image_id + ".png"))
                image_d       = self.get_miou_png(image)[1]
                image_d.save(os.path.join(pred_dir_d, image_id + ".png"))                
                        
            print("Calculate miou.")
            _, IoUs_e, _, _, _, IoUs_d, _, _ = compute_mIoU(gt_dir_e, pred_dir_e, gt_dir_d, pred_dir_d, self.image_ids, self.num_classes, None)  # 执行计算mIoU的函数
            temp_miou_e = np.nanmean(IoUs_e) * 100
            temp_miou_d = np.nanmean(IoUs_d) * 100

            self.mious_e.append(temp_miou_e)
            self.mious_d.append(temp_miou_d)
            self.epoches.append(epoch)

            with open(os.path.join(self.log_dir, "epoch_miou.txt"), 'a') as f:
                f.write(str(temp_miou_e))
                f.write(', ') 
                f.write( str(temp_miou_d))
                f.write("\n")
            
            plt.figure()
            plt.plot(self.epoches, self.mious_e, 'red', linewidth = 2, label='train miou')

            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('Miou')
            plt.title('A Miou Curve')
            plt.legend(loc="upper right")

            plt.savefig(os.path.join(self.log_dir, "epoch_miou_e.png"))
            plt.cla()
            plt.close("all")

            plt.figure()
            plt.plot(self.epoches, self.mious_d, 'red', linewidth = 2, label='train miou')

            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('Miou')
            plt.title('A Miou Curve')
            plt.legend(loc="upper right")

            plt.savefig(os.path.join(self.log_dir, "epoch_miou_d.png"))
            plt.cla()
            plt.close("all")

            print("Get miou done.")
            shutil.rmtree(self.miou_out_path)
