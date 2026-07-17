import sys
import time
from option.option import args
from utils import *
from function.test_function import test_function


if __name__ == '__main__':

    # gpu
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # args.test_dir = "F:/RealTest/0523/0517-MTF/"  # 数据集
    # args.recovery_dir = "F:/RealTest/0523/0517-MTF/PSFFOVKPN_0523"  # 恢复数据
    # args.test_dir = "F:/RealTest/博士楼/0618/BG/"  # 数据集
    # args.recovery_dir = "F:/RealTest/博士楼/0618/BG/PSFFOVKPN/"  # 恢复数据

    args.test_dir = "D:/science_work/Ring_code_0213/Rec_data/Ours_MTF/MTFT1"  # 数据集
    args.recovery_dir = "D:/science_work/Ring_code_0213/Rec_data/Ours_MTF/RecT"  # 恢复数据

    # MTF
    # args.test_dir = "F:/VIS_Dataset_2048X2432/Simulation_Target_Noise/"  # 数据集
    # args.recovery_dir = "F:/VIS_Dataset_2048X2432/Simulation_MTFRecovery/PSF_FOV/"  # 恢复数据

    args.psf_dir = "D:/science_work/FOV_KPN/PSF_small_cut"

    args.wnrgamma = 0.0427
    args.batch = 128

    args.ckpt_dir = "./ckpt_dir_Res_100_Lmg=0.1_EXI=5E8"
    args.log_dir = args.ckpt_dir
    args.NetName = 'Fov_UNet'  # 网络名称
    args.init_epoch = 90  # 待测试的epoch序号
    args.n_epoch = 1  # epoch

    test_function(args)


