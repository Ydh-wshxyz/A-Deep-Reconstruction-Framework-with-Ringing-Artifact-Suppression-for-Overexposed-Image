import os
import random
import sys
import math
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True


class Logger(object):
    # 控制台输出记录到文件
    def __init__(self, file_name="Default.log", stream=sys.stdout):
        self.terminal = stream
        self.log = open(file_name, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)



def step_lr_adjust(optimizer, epoch, init_lr=1e-4, step_size=20, gamma=0.1):
    lr = init_lr * gamma ** (epoch // step_size)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr






def initialize_params(args):

  # Define the `params` dictionary.
  params = dict({})

  # Meta_parameters


  # Sensor parameters
  params['sensor_height'] = 1024           # Sensor pixels  512
  params['sensor_width']  = 1216           # Sensor pixels   512
    
  params['image_width'] = params['sensor_height']
  params['load_width'] = params['sensor_height']
  params['network_width'] = params['sensor_width']
  params['network_height'] = params['sensor_height']
  params['out_width'] = params['sensor_width']
  params['out_height'] = params['sensor_height']
      

  params['batchSize'] = 2

  return params

# using ImageNet values
def normalize_tensor_transform(output, label):
    output_norm = torch.zeros_like(output)
    output_norm[:, 0, ...] = (output[:, 0, ...] - 0.485) 
    output_norm[:, 1, ...] = (output[:, 1, ...] - 0.456) 
    output_norm[:, 2, ...] = (output[:, 2, ...] - 0.406)

    label_norm = torch.zeros_like(label)
    label_norm[:, 0, ...] = (label[:, 0, ...] - 0.485) 
    label_norm[:, 1, ...] = (label[:, 1, ...] - 0.456) 
    label_norm[:, 2, ...] = (label[:, 2, ...] - 0.406) 

    return output_norm, label_norm

def compare_psnr(img1, img2):
    img1 = np.float64(img1)
    img2 = np.float64(img2)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100

    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def ssim(img1, img2):
  C1 = (0.01 * 255)**2
  C2 = (0.03 * 255)**2
  img1 = img1.astype(np.float64)
  img2 = img2.astype(np.float64)
  kernel = cv2.getGaussianKernel(11, 1.5)
  window = np.outer(kernel, kernel.transpose())
  mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5] # valid
  mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
  mu1_sq = mu1**2
  mu2_sq = mu2**2
  mu1_mu2 = mu1 * mu2
  sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
  sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
  sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
  ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                              (sigma1_sq + sigma2_sq + C2))
  return ssim_map.mean()
def calculate_ssim(img1, img2):
  '''calculate SSIM
  the same outputs as MATLAB's
  img1, img2: [0, 255]
  '''
  if not img1.shape == img2.shape:
    raise ValueError('Input images must have the same dimensions.')
  if img1.ndim == 2:
    return ssim(img1, img2)
  elif img1.ndim == 3:
    if img1.shape[2] == 3:
      ssims = []
      for i in range(3):
        ssims.append(ssim(img1, img2))
      return np.array(ssims).mean()
    elif img1.shape[2] == 1:
      return ssim(np.squeeze(img1), np.squeeze(img2))
  else:
    raise ValueError('Wrong input image dimensions.')
 