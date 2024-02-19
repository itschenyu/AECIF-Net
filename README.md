# AECIF-Net
This is the implementation of the paper ["Attention-Enhanced Co-Interactive Fusion Network (AECIF-Net) for automated structural condition assessment in visual inspection"](https://www.sciencedirect.com/science/article/abs/pii/S0926580524000281).

## Network Architecture
![model](https://user-images.githubusercontent.com/90736946/280549214-7f17fd8a-a172-489c-b92c-214b0d81f794.png)

## Getting Started
### Installation
* Install the required dependencies in `requirements.txt`.
* Clone this repo:
~~~~
git clone https://github.com/itschenyu/AECIF-Net.git
cd AECIF-Net
~~~~
### Dataset
* Please download the SBCIV dataset from [here](https://drive.google.com/drive/folders/15fmV5aLoMnWC-IWyCLNkE2qH8MDPfvox?usp=sharing) and then place it in `./VOCdevkit/VOC2007/`.

### Pre-trained Weight
* Please download pre-trained weights on Cityscapes from [here](https://cmu.app.box.com/s/if90kw6r66q2y6c5xparflhnbwi6c2yi) and place it in `./model_data/`.

### Model Download
|   Model   | mIoU_Element | mIoU_Defect | Weight |
|:---------:|:------------:|:-----------:|:------:|
| AECIF-Net |     92.11    |    87.16    |    [Link](https://drive.google.com/file/d/1OeWRTi49QwzQzw2OZm52HqPZRayIKdM5/view?usp=sharing)   |

### Training
~~~~
python train.py
~~~~

### Testing
Evaluating the model on the test set:
~~~~
python get_miou.py
~~~~

### Inference
Place the inference images in `./img/`, and then run:
~~~~
python predict.py
~~~~

## Citation
If AECIF-Net and the SBCIV dataset are helpful to you, please cite them as:
~~~~
@article{ZHANG2024105292,
      title = {Attention-Enhanced Co-Interactive Fusion Network (AECIF-Net) for automated structural condition assessment in visual inspection},
      journal = {Automation in Construction},
      volume = {159},
      pages = {105292},
      year = {2024},
      issn = {0926-5805},
      doi = {https://doi.org/10.1016/j.autcon.2024.105292},
      url = {https://www.sciencedirect.com/science/article/pii/S0926580524000281},
      author = {Chenyu Zhang and Zhaozheng Yin and Ruwen Qin}
}
~~~~
## Note
Part of the codes are referred from <a href="https://github.com/itschenyu/Multitask-Learning-Bridge-Inspection">MTL-Bridge-Inspection</a> project.

The images and corrosion annotations in the dataset are credited to [Corrosion Condition State Semantic Segmentation Dataset](https://data.lib.vt.edu/articles/dataset/Corrosion_Condition_State_Semantic_Segmentation_Dataset/16624663/2) and [COCO-Bridge Dataset](https://data.lib.vt.edu/articles/dataset/COCO-Bridge_2021_Dataset/16624495/1).
