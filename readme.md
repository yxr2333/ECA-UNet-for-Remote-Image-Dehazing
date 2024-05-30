# ECA-UNet for Image Dehazing

This repository contains the code for mu undergraduate graduation project "ECA-UNet: A Simple Model for Image Dehazing"

## Introduction

This project is a simple model for image dehazing. The model is based on the U-Net architecture with the ECA module. The model is trained on the Haze1K dataset and the model is evaluated using the PSNR and SSIM metrics.

## Requirements

The following is my experimental environment, which does not represent the minimum requirements for the experimental environment

- Python 3.8.19
- PyTorch 2.2.1
- Anaconda

## Usage

Download the repository and Haze1K dataset, and set the path in the `config.py` file.

Subsequent steps wait for updates...

## Results

In order to better show our experimental results, we train on the Haze1K dataset and test on the test set. We use PSNR and SSIM as evaluation metrics. In order to show the superiority of our model, we also compare with other models. Second, to verify the effectiveness of our work, we also conduct ablation experiments.

The following table shows the results of our model and other models on the Haze1K dataset.
![compare_data.png](imgs/compare_data.png)

Comparing the results on Haze1K thin fog images.
![img1.png](imgs/img1.png)

Comparing the results on Haze1K moderate fog images.
![img2.png](imgs/img2.png)

Comparing the results on Haze1K thick fog images.
![img3.png](imgs/img3.png)

The following table shows the results of our ablation experiments.
![1.png](imgs%2F1.png)
