import os
import numpy as np
import cv2
import scipy.io as sio
import torch
import matplotlib.image as mpimg


def file_match(s, root):
    dirs = []
    matchs = []
    for current_name in os.listdir(root):
        add_root_name = os.path.join(root, current_name)
        if os.path.isdir(add_root_name):
            dirs.append(add_root_name)
        elif os.path.isfile(add_root_name) and (s == os.path.splitext(add_root_name)[-1][1:]):
            matchs.append(add_root_name)
    for dir in dirs:
        file_match(s, dir)
    return matchs


def sensor_noise_scale(BL, GT, clip = (1E-20,1.)):
    dtype = BL.dtype
    device = BL.device

    scale = 1. + torch.rand((1), device=device, dtype=dtype) * 9.

    # BL_temp = (BL /scale) / 255.
    Fov_temp = BL[:, 1:3, :, :]
    BL_temp = BL[:, 0:1, :, :]
    BL_temp = BL_temp / 255.

    gauss_noise = 0.003 * torch.randn(BL_temp.shape, device=device, dtype=dtype) * scale

    BL_temp = BL_temp + gauss_noise
    BL_temp = torch.clamp(BL_temp, clip[0], clip[1])
    BL_output_temp = torch.cat((BL_temp, Fov_temp), dim=1)

    BL_output = BL_output_temp
    GT_output = GT / 255.

    return BL_output, GT_output

def sensor_noise_scale1(BL, GT, clip = (1E-20,1.)):
    dtype = BL.dtype
    device = BL.device

    # scale = 1. + torch.rand((1), device=device, dtype=dtype) * 9.

    # BL_temp = (BL /scale) / 255.
    Fov_temp = BL[:, 1:3, :, :]
    BL_temp = BL[:, 0:1, :, :]
    BL_temp = BL_temp / 255.

    # gauss_noise = 0.003 * torch.randn(BL_temp.shape, device=device, dtype=dtype) * scale

    # BL_temp = BL_temp + gauss_noise
    BL_temp = torch.clamp(BL_temp, clip[0], clip[1])
    BL_output_temp = torch.cat((BL_temp, Fov_temp), dim=1)

    BL_output = BL_output_temp
    GT_output = GT / 255.

    return BL_output, GT_output

def sensor_noise(BL, GT, a_poission, b_sqrt, clip=(1E-20, 1.)):
    dtype = BL.dtype
    device = BL.device
    a_poission_torch = torch.tensor(a_poission, device=device, dtype=torch.float32)
    b_sqrt_torch = torch.tensor(b_sqrt, device=device, dtype=torch.float32)

    Fov_temp = BL[:, 1:3, :, :]
    BL_temp = BL[:, 0:1, :, :]
    BL_temp = BL_temp / 65535.
    BL_temp = torch.clamp(BL_temp, a_poission_torch * 0.01, clip[1])  # 避免遇到大部分是0的时候饱和
    # 高斯噪声
    gauss_noise = b_sqrt_torch * torch.randn(BL_temp.shape, device=device, dtype=dtype)
    # 泊松噪声
    rate = BL_temp / a_poission_torch
    poisson_dist = torch.distributions.Poisson(rate = rate)
    sample = poisson_dist.sample()
    poission = (sample - rate) * a_poission_torch

    BL_temp = BL_temp + gauss_noise + poission
    BL_temp = torch.clamp(BL_temp, clip[0], clip[1])
    BL_output_temp = torch.cat((BL_temp, Fov_temp), dim=1)

    BL_output = BL_output_temp
    GT_output = GT / 65535.

    return BL_output, GT_output


def dataset_load(target_dir, filetype, color = False ):
    ## 将所有文件load进内存
    print("start data loading.....")
    file_lists = file_match(filetype, target_dir)

    input_image = [None]*8000
    
    if filetype == 'mat':
        for i in range(len(file_lists)):
            data_temp = sio.loadmat(file_lists[i])   
            input_image[i] = data_temp['BGAll'].astype(np.float32)[:,:,np.newaxis]
            # print(input_image[i].shape)
            print(i)
    else:
        for i in range(len(file_lists)):
            data_temp =  mpimg.imread(file_lists[i])        
            input_image[i] = data_temp.astype(np.float32)[:,:,np.newaxis]
            # print(input_image[i].shape)
            print(i)
        
    print(len(input_image))
    print("data load finished!")
    return (input_image, len(file_lists))

def PSF_load(target_dir, filetype, color=False):
    ## 将所有文件load进内存
    print("start data loading.....")
    file_lists = file_match(filetype, target_dir)

    input_image = [None] * len(file_lists)

    if filetype == 'mat':
        for i in range(len(file_lists)):
            data_temp = sio.loadmat(file_lists[i])
            input_image[i] = data_temp['PSF'].astype(np.float32)[:, :]
            print(input_image[i].shape)
    else:
        for i in range(len(file_lists)):
            data_temp = mpimg.imread(file_lists[i])
            input_image[i] = data_temp.astype(np.float32)[:, :]
            print(input_image[i].shape)

    print(len(input_image))
    print("data load finished!")
    return (input_image, len(file_lists))

def dataset_load_RealImage(target_dir, filetype, color=False):
    ## 将所有文件load进内存
    print("start data loading.....")
    file_lists = file_match(filetype, target_dir)

    input_image = [None] * 8000

    if filetype == 'mat':

        for i in range(len(file_lists)):
            data_temp = sio.loadmat(file_lists[i])['BGAll'].astype(np.float32)
            image_temp = data_temp[:,2,:, 0:2432]
            image_temp = np.squeeze(np.squeeze(image_temp))
            input_image[i] = image_temp[:, :, np.newaxis]
            print(input_image[i].shape)
    else:
        for i in range(len(file_lists)):
            data_temp = mpimg.imread(file_lists[i])
            input_image[i] = data_temp.astype(np.float32)[:, :, np.newaxis]
            print(input_image[i].shape)

    print(len(input_image))
    print("data load finished!")
    return (input_image, len(file_lists))

def dataset_load_RealImage1(target_dir, filetype, color=False):
    ## 将所有文件load进内存
    print("start data loading.....")
    file_lists = file_match(filetype, target_dir)

    input_image = [None] * 8000

    if filetype == 'mat':
        count = 0
        for i in range(len(file_lists)):
            data_temp = sio.loadmat(file_lists[i])['BGAll'].astype(np.float32)
            for j in range(data_temp.shape[1]):
                image_temp = data_temp[:,j,:, 0:2432]
                image_temp = np.squeeze(image_temp)
                input_image[count] = image_temp[:, :, np.newaxis]
                print(input_image[count].shape)
                count += 1

    print(len(input_image))
    print("data load finished!")
    return (input_image, count)


def dataset_load_RealImage2(target_dir, filetype, color=False):
    ## 将所有文件load进内存
    print("start data loading.....")
    file_lists = file_match(filetype, target_dir)

    input_image = [None] * 8000

    if filetype == 'mat':

        for i in range(len(file_lists)):
            data_temp = sio.loadmat(file_lists[i])['BG'].astype(np.float32)
            image_temp = data_temp[:, :, 0:2432]
            image_temp = np.squeeze(image_temp)
            data_min1 = np.min(image_temp)
            data_max1 = np.max(image_temp)
            frame_new = (image_temp - data_min1) / (data_max1 - data_min1) * 65535
            input_image[i] = frame_new[:, :, np.newaxis]
            print(input_image[i].shape)
    else:
        for i in range(len(file_lists)):
            data_temp = mpimg.imread(file_lists[i])
            input_image[i] = data_temp.astype(np.float32)[:, :, np.newaxis]
            print(input_image[i].shape)

    print(len(input_image))
    print("data load finished!")
    return (input_image, len(file_lists))

def dataset_load_RealImage3(target_dir, filetype, num):
    ## 将所有文件load进内存
    print("start data loading.....")
    file_lists = file_match(filetype, target_dir)

    input_image = [None] * 8000

    if filetype == 'mat':

        for i in range(len(file_lists)):
            data_temp = sio.loadmat(file_lists[i])['BGAll'].astype(np.float32)
            image_temp = data_temp[:, num, :, 0:2432]
            image_temp = np.squeeze(image_temp)
            data_min1 = np.min(image_temp)
            data_max1 = np.max(image_temp)
            frame_new = (image_temp - data_min1) / (data_max1 - data_min1) * 65535
            input_image[i] = frame_new[:, :, np.newaxis]
            print(input_image[i].shape)
    else:
        for i in range(len(file_lists)):
            data_temp = mpimg.imread(file_lists[i])
            input_image[i] = data_temp.astype(np.float32)[:, :, np.newaxis]
            print(input_image[i].shape)

    print(len(input_image))
    print("data load finished!")
    return (input_image, len(file_lists))
