import time
import os
import cv2
import numpy as np
from PIL import Image

from AECIF_Net import HRnet_Segmentation


if __name__ == "__main__":
    hrnet = HRnet_Segmentation()
    mode = "fps"    # 'predict', 'video', 'fps' or 'dir_predict'
    count           = False
    name_classes    = [["_background_","bearing","bracing","deck","floor_beam","girder","substructure"], ["_background_","Corrosion"]]
    video_path      = 0
    video_save_path = ""
    video_fps       = 25.0
    test_interval = 100
    fps_image_path  = "img/316.jpg"
    dir_origin_path = "img/"
    dir_save_path_e   = "img_out/element"
    dir_save_path_d   = "img_out/defect"
    simplify        = True
    onnx_save_path  = "model_data/models.onnx"

    if mode == "predict":
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = hrnet.detect_image(image, count=count, name_classes=name_classes)
                r_image.show()

    elif mode == "video":
        capture=cv2.VideoCapture(video_path)
        if video_save_path!="":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        ref, frame = capture.read()
        if not ref:
            raise ValueError("Failed to read the camera (video), please check whether the camera is installed correctly (whether the video path is filled in correctly).")

        fps = 0.0
        while(True):
            t1 = time.time()
            ref, frame = capture.read()
            if not ref:
                break
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(np.uint8(frame))
            frame = np.array(hrnet.detect_image(frame))
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
            
            fps  = ( fps + (1./(time.time()-t1)) ) / 2
            print("fps= %.2f"%(fps))
            frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("video",frame)
            c= cv2.waitKey(1) & 0xff 
            if video_save_path!="":
                out.write(frame)

            if c==27:
                capture.release()
                break
        print("Video Detection Done!")
        capture.release()
        if video_save_path!="":
            print("Save processed video to the path :" + video_save_path)
            out.release()
        cv2.destroyAllWindows()

    elif mode == "fps":
        img = Image.open(fps_image_path)
        tact_time = hrnet.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1/tact_time) + 'FPS, @batch_size 1')
        
    elif mode == "dir_predict":
        import os
        from tqdm import tqdm

        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path  = os.path.join(dir_origin_path, img_name)
                image       = Image.open(image_path)
                r_image     = hrnet.detect_image(image)
                if not os.path.exists(dir_save_path_e):
                    os.makedirs(dir_save_path_e)
                if not os.path.exists(dir_save_path_d):
                    os.makedirs(dir_save_path_d)
                r_image[0].save(os.path.join(dir_save_path_e, img_name))
                r_image[1].save(os.path.join(dir_save_path_d, img_name))
                
    elif mode == "export_onnx":
        hrnet.convert_to_onnx(simplify, onnx_save_path)
        
    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")