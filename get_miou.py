import os

from PIL import Image
from tqdm import tqdm

from AECIF_Net import HRnet_Segmentation
from utils.utils_metrics_multi import compute_mIoU, show_results

if __name__ == "__main__":
    miou_mode       = 0
    num_classes     = [7, 2]
    name_classes    = [["background","bearing","bracing","deck","floor_beam","girder","substructure"], ["background_","Corrosion"]]
    VOCdevkit_path  = 'VOCdevkit'

    image_ids       = open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/test.txt"),'r').read().splitlines()
    gt_dir_e          = os.path.join(VOCdevkit_path, "VOC2007/SegmentationClass/element/")
    gt_dir_d          = os.path.join(VOCdevkit_path, "VOC2007/SegmentationClass/defect/")
    miou_out_path_e   = "miou_out/element"
    miou_out_path_d   = "miou_out/defect"
    pred_dir_e        = os.path.join(miou_out_path_e, 'detection-results/')
    pred_dir_d        = os.path.join(miou_out_path_d, 'detection-results/')

    if miou_mode == 0 or miou_mode == 1:
        if not os.path.exists(pred_dir_e):
            os.makedirs(pred_dir_e)
        if not os.path.exists(pred_dir_d):
            os.makedirs(pred_dir_d) 

        print("Load model.")
        hrnet = HRnet_Segmentation()
        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path  = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/"+image_id+".jpg")
            image       = Image.open(image_path)
            image       = hrnet.get_miou_png(image)
            image[0].save(os.path.join(pred_dir_e, image_id + ".png"))
            image[1].save(os.path.join(pred_dir_d, image_id + ".png"))
        print("Get predict result done.")

    if miou_mode == 0 or miou_mode == 2:
        print("Get miou.")
        hist_e, IoUs_e, PA_Recall_e, Precision_e, hist_d, IoUs_d, PA_Recall_d, Precision_d = compute_mIoU(gt_dir_e, pred_dir_e, gt_dir_d, pred_dir_d, image_ids, num_classes, name_classes)  # 执行计算mIoU的函数
        print("Get miou done.")
        show_results(miou_out_path_e, hist_e, IoUs_e, PA_Recall_e, Precision_e, name_classes[0])
        show_results(miou_out_path_d, hist_d, IoUs_d, PA_Recall_d, Precision_d, name_classes[1])