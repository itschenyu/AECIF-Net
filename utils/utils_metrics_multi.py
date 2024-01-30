import csv
import os
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

def f_score(inputs, target, beta=1, smooth = 1e-5, threhold = 0.5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)
        
    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c),-1)
    temp_target = target.view(n, -1, ct)

    temp_inputs = torch.gt(temp_inputs, threhold).float()
    tp = torch.sum(temp_target[...,:-1] * temp_inputs, axis=[0,1])
    fp = torch.sum(temp_inputs                       , axis=[0,1]) - tp
    fn = torch.sum(temp_target[...,:-1]              , axis=[0,1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    score = torch.mean(score)
    return score

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)  

def per_class_iu(hist):
    return np.diag(hist) / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist)), 1) 

def per_class_PA_Recall(hist):
    return np.diag(hist) / np.maximum(hist.sum(1), 1) 

def per_class_Precision(hist):
    return np.diag(hist) / np.maximum(hist.sum(0), 1) 

def per_Accuracy(hist):
    return np.sum(np.diag(hist)) / np.maximum(np.sum(hist), 1) 

def compute_mIoU(gt_dir_e, pred_dir_e, gt_dir_d, pred_dir_d, png_name_list, num_classes, name_classes=None):  
    print('Num classes_e', num_classes[0])
    print('Num classes_d', num_classes[1])          
    hist_e = np.zeros((num_classes[0], num_classes[0]))
    hist_d = np.zeros((num_classes[1], num_classes[1]))
    gt_imgs_e     = [join(gt_dir_e, x + ".png") for x in png_name_list]
    pred_imgs_e   = [join(pred_dir_e, x + ".png") for x in png_name_list]  
    gt_imgs_d     = [join(gt_dir_d, x + ".png") for x in png_name_list]  
    pred_imgs_d   = [join(pred_dir_d, x + ".png") for x in png_name_list]
    for ind in range(len(gt_imgs_e)): 
        pred_e = np.array(Image.open(pred_imgs_e[ind]))  
        label_e = np.array(Image.open(gt_imgs_e[ind]))  
        if len(label_e.flatten()) != len(pred_e.flatten()):  
            print(
                'Skipping: len(gt_e) = {:d}, len(pred_e) = {:d}, {:s}, {:s}'.format(
                    len(label_e.flatten()), len(pred_e.flatten()), gt_imgs_e[ind],
                    pred_imgs_e[ind]))
            continue

        hist_e += fast_hist(label_e.flatten(), pred_e.flatten(), num_classes[0])  
        if name_classes is not None and ind > 0 and ind % 10 == 0: 
            print('{:d} / {:d}: mIou_e-{:0.2f}%; mPA_e-{:0.2f}%; Accuracy_e-{:0.2f}%'.format(
                    ind, 
                    len(gt_imgs_e),
                    100 * np.nanmean(per_class_iu(hist_e)),
                    100 * np.nanmean(per_class_PA_Recall(hist_e)),
                    100 * per_Accuracy(hist_e)
                )
            )

    for ind in range(len(gt_imgs_d)): 
        pred_d = np.array(Image.open(pred_imgs_d[ind]))  
        label_d = np.array(Image.open(gt_imgs_d[ind]))  

        if len(label_d.flatten()) != len(pred_d.flatten()):  
            print(
                'Skipping: len(gt_d) = {:d}, len(pred_d) = {:d}, {:s}, {:s}'.format(
                    len(label_d.flatten()), len(pred_d.flatten()), gt_imgs_d[ind],
                    pred_imgs_d[ind]))
            continue

        hist_d += fast_hist(label_d.flatten(), pred_d.flatten(), num_classes[1])  
        if name_classes is not None and ind > 0 and ind % 10 == 0: 
            print('{:d} / {:d}: mIou_d-{:0.2f}%; mPA_d-{:0.2f}%; Accuracy_d-{:0.2f}%'.format(
                    ind, 
                    len(gt_imgs_d),
                    100 * np.nanmean(per_class_iu(hist_d)),
                    100 * np.nanmean(per_class_PA_Recall(hist_d)),
                    100 * per_Accuracy(hist_d)
                )
            )

    IoUs_e        = per_class_iu(hist_e)
    PA_Recall_e   = per_class_PA_Recall(hist_e)
    Precision_e   = per_class_Precision(hist_e)
    IoUs_d        = per_class_iu(hist_d)
    PA_Recall_d   = per_class_PA_Recall(hist_d)
    Precision_d   = per_class_Precision(hist_d)

    if name_classes is not None:
        for ind_class in range(num_classes[0]):
            print('===>' + name_classes[0][ind_class] + ':\tIou_e-' + str(round(IoUs_e[ind_class] * 100, 2)) \
                + '; Recall_e (equal to the PA)-' + str(round(PA_Recall_e[ind_class] * 100, 2))+ '; Precision_e-' + str(round(Precision_e[ind_class] * 100, 2)))
    print('===========================================')
    if name_classes is not None:
        for ind_class in range(num_classes[1]):
            print('===>' + name_classes[1][ind_class] + ':\tIou_d-' + str(round(IoUs_d[ind_class] * 100, 2)) \
                + '; Recall_d (equal to the PA)-' + str(round(PA_Recall_d[ind_class] * 100, 2))+ '; Precision_d-' + str(round(Precision_d[ind_class] * 100, 2)))
    print('===========================================')
    print('===> mIoU_e: ' + str(round(np.nanmean(IoUs_e) * 100, 2)) + '; mPA_e: ' + str(round(np.nanmean(PA_Recall_e) * 100, 2)) + '; Accuracy_e: ' + str(round(per_Accuracy(hist_e) * 100, 2)) + '; Precision_e: ' + str(round(np.nanmean(Precision_e) * 100, 2)))
    print('===> mIoU_d: ' + str(round(np.nanmean(IoUs_d) * 100, 2)) + '; mPA_d: ' + str(round(np.nanmean(PA_Recall_d) * 100, 2)) + '; Accuracy_d: ' + str(round(per_Accuracy(hist_d) * 100, 2)) + '; Precision_d: ' + str(round(np.nanmean(Precision_d) * 100, 2)))
   
    return np.array(hist_e, np.int), IoUs_e, PA_Recall_e, Precision_e, np.array(hist_d, np.int), IoUs_d, PA_Recall_d, Precision_d

def adjust_axes(r, t, fig, axes):
    bb                  = t.get_window_extent(renderer=r)
    text_width_inches   = bb.width / fig.dpi
    current_fig_width   = fig.get_figwidth()
    new_fig_width       = current_fig_width + text_width_inches
    propotion           = new_fig_width / current_fig_width
    x_lim               = axes.get_xlim()
    axes.set_xlim([x_lim[0], x_lim[1] * propotion])

def draw_plot_func(values, name_classes, plot_title, x_label, output_path, tick_font_size = 12, plt_show = True):
    fig     = plt.gcf() 
    axes    = plt.gca()
    plt.barh(range(len(values)), values, color='royalblue')
    plt.title(plot_title, fontsize=tick_font_size + 2)
    plt.xlabel(x_label, fontsize=tick_font_size)
    plt.yticks(range(len(values)), name_classes, fontsize=tick_font_size)
    r = fig.canvas.get_renderer()
    for i, val in enumerate(values):
        str_val = " " + str(val) 
        if val < 1.0:
            str_val = " {0:.2f}".format(val)
        t = plt.text(val, i, str_val, color='royalblue', va='center', fontweight='bold')
        if i == (len(values)-1):
            adjust_axes(r, t, fig, axes)

    fig.tight_layout()
    fig.savefig(output_path)
    if plt_show:
        plt.show()
    plt.close()

def show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes, tick_font_size = 12):
    draw_plot_func(IoUs, name_classes, "mIoU = {0:.2f}%".format(np.nanmean(IoUs)*100), "Intersection over Union", \
        os.path.join(miou_out_path, "mIoU.png"), tick_font_size = tick_font_size, plt_show = True)
    print("Save mIoU out to " + os.path.join(miou_out_path, "mIoU.png"))

    draw_plot_func(PA_Recall, name_classes, "mPA = {0:.2f}%".format(np.nanmean(PA_Recall)*100), "Pixel Accuracy", \
        os.path.join(miou_out_path, "mPA.png"), tick_font_size = tick_font_size, plt_show = False)
    print("Save mPA out to " + os.path.join(miou_out_path, "mPA.png"))
    
    draw_plot_func(PA_Recall, name_classes, "mRecall = {0:.2f}%".format(np.nanmean(PA_Recall)*100), "Recall", \
        os.path.join(miou_out_path, "Recall.png"), tick_font_size = tick_font_size, plt_show = False)
    print("Save Recall out to " + os.path.join(miou_out_path, "Recall.png"))

    draw_plot_func(Precision, name_classes, "mPrecision = {0:.2f}%".format(np.nanmean(Precision)*100), "Precision", \
        os.path.join(miou_out_path, "Precision.png"), tick_font_size = tick_font_size, plt_show = False)
    print("Save Precision out to " + os.path.join(miou_out_path, "Precision.png"))

    with open(os.path.join(miou_out_path, "confusion_matrix.csv"), 'w', newline='') as f:
        writer          = csv.writer(f)
        writer_list     = []
        writer_list.append([' '] + [str(c) for c in name_classes])
        for i in range(len(hist)):
            writer_list.append([name_classes[i]] + [str(x) for x in hist[i]])
        writer.writerows(writer_list)
    print("Save confusion_matrix out to " + os.path.join(miou_out_path, "confusion_matrix.csv"))
            
