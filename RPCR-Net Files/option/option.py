import argparse

parser = argparse.ArgumentParser()
# File paths
parser.add_argument('--ckpt_dir', type=str, default="./ckpt_dir",
                    help='model directory')
parser.add_argument('--log_dir', type=str, default="./log_dir",
                    help='log directory')
parser.add_argument('--train_dir', type=str, default="D:/Backup/Desktop/20231006/train/",
                    help='train directory')
parser.add_argument('--test_dir', type=str, default="D:/Backup/Desktop/20231006/test/",
                    help='test directory')
parser.add_argument('--evaluate_dir', type=str, default="D:/Backup/Desktop/20231006/evaluate/",
                    help='evaluate directory')
# Hardware specifications
parser.add_argument('--gpu', type=str, default="0",
                    help='GPUs')

# model
parser.add_argument('--NetName', default='Unet',
                    help='model name')
parser.add_argument('--is_deconv', type=bool, default=True,
                    help='Whether to use deconv')
parser.add_argument('--is_batchnorm', type=bool, default=False,
                    help='Whether to use batchnorm')
parser.add_argument('--finetune', type=bool, default=False,
                    help='if finetune model, set True')

# Training parameters
parser.add_argument('--sigma', type=float, default=0.03,
                    help='Gaussian noise')
parser.add_argument('--sigma_min', type=float, default=0.0003,
                    help='Min gaussian noise')
parser.add_argument('--sigma_max', type=float, default=0.03,
                    help='Max gaussian noise')
parser.add_argument('--init_epoch', type=int, default=0,
                    help='if finetune model, set the initial epoch')
parser.add_argument('--n_epoch', type=int, default=200,
                    help='the number of training epochs')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--is_lr_adjustment', type=bool, default=False,
                    help='Whether to adjust the learning rate')
parser.add_argument('--milestone', type=int, default=10,
                    help='the epochs for weight decay')
parser.add_argument('--log_freq', type=int, default=100,
                    help='do validation per every N epochs')
parser.add_argument('--save_val_img', type=bool, default=True,
                    help='save the last validated image for comparison')
parser.add_argument('--val_patch_size', type=int, default=512,
                    help='patch size in validation dataset')
parser.add_argument('--save_epoch', type=int, default=1,
                    help='save model per every N epochs')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='learning rate decay factor for every milestone')


# loss
parser.add_argument('--t_loss', type=str, default='L2_wz_Perceptual',
                    help='training loss: L2, L1, L2_wz_TV, L2_wz_Perceptual, L2_wz_SSIM')
parser.add_argument('--tv_weight', type=float, default=4e-8,
                    help='tvloss weight')
parser.add_argument('--mse_weight', type=float, default=1,
                    help='style weight of perceptual loss')
parser.add_argument('--Perceptual_weight', type=float, default=5e-3,
                    help='content weight of perceptual loss')
parser.add_argument('--ssim_weight', type=float, default=2e-1,
                    help='ssim weight of ssim loss')                   

# test
parser.add_argument('--result_png_path', type=str, default="./test_result_double",
                    help='result directory')
parser.add_argument('--ckpt_dir_test', type=str, default="./ckpt_dir_20231014",
                    help='model directory')
parser.add_argument('--epoch_test', type=int, default=200,
                    help='the epoch for testing')

# test real
parser.add_argument('--blurred_src_path', type=str, default="D:/Backup/Desktop/save_1/",
                    help='blurred image directory')
parser.add_argument('--result_png_path_real', type=str, default="./test_result_real",
                    help='result directory')
parser.add_argument('--ckpt_dir_test_real', type=str, default="./ckpt_dir_20231014",
                    help='model directory')
parser.add_argument('--epoch_test_real', type=int, default=200,
                    help='the epoch for testing')

args = parser.parse_args()
