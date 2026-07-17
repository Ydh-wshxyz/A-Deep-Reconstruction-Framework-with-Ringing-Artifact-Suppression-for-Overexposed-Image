import sys
import time
from option.option import args
from utils import *
from function.train_function import train_function

if __name__ == '__main__':

    # gpu
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    args.train_dir = "D:/science_work/ring_data/test_0211/cut_65535/L1_1_L2_0_LPSF_1_DATA/train/"  # 训练数据集
    args.test_dir = "D:/science_work/ring_data/test_0211/cut_65535/L1_1_L2_0_LPSF_1_DATA/test/"  # 测试数据集
    args.psf_dir = "D:/science_work/FOV_KPN/PSF_small_cut"  # 测试数据集
    args.ckpt_dir = "./FOV_KPN_Lmg=0.1_EXI=5E8"

    args.a_poission = 8.78e-5
    args.b_sqrt = 1.4346e-3
    args.wnrgamma = 0.0427

    args.batch = 128

    args.log_dir = args.ckpt_dir
    args.NetName = 'FOV-KPN'  # 网络名称
    args.init_epoch = 50 # 初始化epoch
    args.n_epoch = 100  # epoch
    args.netlr = 2e-4  # 学习率
    args.wnrlr = 0.5e-4  # 学习率
    args.Perceptual_weight = 0.005

    train_function(args)


