# ECA-UNet for Image Dehazing

This repository contains the code for mu undergraduate graduation project "ECA-UNet: A Simple Model for Image Dehazing"

## Introduction

This project is a simple model for image dehazing. The model is based on the U-Net architecture with the ECA module. The model is trained on the Haze1K dataset and the model is evaluated using the PSNR and SSIM metrics.

## Requirements

The following is my experimental environment, which does not represent the minimum requirements for the experimental environment

- python 3.12.4
- pyTorch 2.5.1
- torchvision 0.20.1
- cuda 12.4
- anaconda
- ...

## Usage

Download the repository and Haze1K dataset [(Download dataset)](https://www.dropbox.com/scl/fi/wtga5ltw5vby5x7trnp0p/Haze1k.zip?rlkey=70s52w3flhtif020nx250jru3&e=1&dl=0), and change the configuration in the `config.py` file.

Then, run the `train.py` to train the model.

During the training process, the `last` and the `best` models will be saved in the `weights` folder.

If you want to evaluate the model metrics, run the `evaluate.py` file.


## Results

In order to better show our experimental results, we train on the Haze1K dataset and test on the test set. We use PSNR and SSIM as evaluation metrics. In order to show the superiority of our model, we also compare with other models. Second, to verify the effectiveness of our work, we also conduct ablation experiments.

[//]: # (The following table shows the results of our model and other models on the Haze1K dataset.)

[//]: # (![compare_data.png]&#40;imgs/compare_data.png&#41;)

Comparing the results on Haze1K thin fog images.
![img1.png](imgs/img1.png)

Comparing the results on Haze1K moderate fog images.
![img2.png](imgs/img2.png)

Comparing the results on Haze1K thick fog images.
![img3.png](imgs/img3.png)

## Visualize

Use the Gradio framework to create a visual interface for the model


Interface overview
![img3.png](imgs/4.png)

Upload the original fog image and the corresponding real fog free image
![img3.png](imgs/2.png)

System generated result
![img3.png](imgs/3.png)

## Reference
[1] Huang B, Zhi L, Yang C, et al. Single satellite optical imagery dehazing using SAR image prior based on conditional generative adversarial networks[C]//Proceedings of the IEEE/CVF winter conference on applications of computer vision. 2020: 1806-1813.

[2] Ronneberger O, Fischer P, Brox T. U-net: Convolutional networks for biomedical image segmentation[C]//Medical image computing and computer-assisted intervention–MICCAI 2015: 18th international conference, Munich, Germany, October 5-9, 2015, proceedings, part III 18. Springer international publishing, 2015: 234-241.

[3] Wang Q, Wu B, Zhu P, et al. ECA-Net: Efficient channel attention for deep convolutional neural networks[C]//Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2020: 11534-11542.