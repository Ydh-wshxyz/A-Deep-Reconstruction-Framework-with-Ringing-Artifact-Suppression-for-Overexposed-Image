import cv2
from tensorboardX import SummaryWriter
from utils import *
from model.__init__ import make_model
from loss.__init__ import select_loss
from Dataset import dataset_load, PSF_load
import time
from torchvision import transforms as transforms
import torch
import numpy as np
import scipy.io as sio
import torch.nn.functional as F
from model.fov_kpn import deconWnr
from scipy.interpolate import RegularGridInterpolator

def generate_weight(PSF_h_num, PSF_w_num, patch_length):
    M = patch_length / 2 - 1
    x = np.arange(0, 3 * patch_length, 1)
    y = np.arange(0, 3 * patch_length, 1)
    X, Y = np.meshgrid(x, y)
    x0 = np.array([M, M + patch_length, M + 2 * patch_length])
    y0 = np.array([M, M + patch_length, M + 2 * patch_length])

    weight = np.zeros([PSF_h_num, PSF_w_num, 3 * patch_length, 3 * patch_length])
    for h in range(PSF_h_num):
        for w in range(PSF_w_num):
            if h == 0 and w == 0:
                z0 = np.array([[1, 1, 0],
                               [1, 1, 0],
                               [0, 0, 0]])

            elif h == 0 and w < PSF_w_num - 1 and w > 0:
                z0 = np.array([[0, 0, 0],
                               [1, 1, 0],
                               [0, 0, 0]])
            elif h == 0 and w == PSF_w_num - 1:
                z0 = np.array([[0, 0, 0],
                               [1, 1, 0],
                               [1, 1, 0]])
            elif w == 0 and h < PSF_h_num - 1 and h > 0:
                z0 = np.array([[0, 1, 0],
                               [0, 1, 0],
                               [0, 0, 0]])
            elif w == 0 and h == PSF_h_num - 1:
                z0 = np.array([[0, 1, 1],
                               [0, 1, 1],
                               [0, 0, 0]])
            elif h == PSF_h_num - 1 and w < PSF_w_num - 1 and w > 0:
                z0 = np.array([[0, 0, 0],
                               [0, 1, 1],
                               [0, 0, 0]])
            elif h == PSF_h_num - 1 and w == PSF_w_num - 1:
                z0 = np.array([[0, 0, 0],
                               [0, 1, 1],
                               [0, 1, 1]])
            elif h < PSF_h_num - 1 and w == PSF_w_num - 1 and h > 0:
                z0 = np.array([[0, 0, 0],
                               [0, 1, 0],
                               [0, 1, 0]])
            else:
                z0 = np.array([[0, 0, 0],
                               [0, 1, 0],
                               [0, 0, 0]])

            interp = RegularGridInterpolator((x0, y0), z0,
                                             method='linear', bounds_error=False, fill_value=0)
            weight[h, w, :, :] = interp((X, Y))

    return weight

# 平均池化：n个像素平均成1个
def average_pool(mar, n):
    sz = np.shape(mar)
    _mar = mar.reshape(sz[0] // n, n, sz[1] // n, n)
    Mar = _mar.mean(axis=(1, 3))
    return Mar

def test_function(args):
    params = initialize_params(args)

    # Dataset
    test_data, test_number = dataset_load(args.test_dir, 'mat')  # 测试数据

    # 加载FOV Data
    FOVX = sio.loadmat('D:/science_work/Ring_code_0213/Pytorch_PSF_FOVKPN_3X3Split_weight/X.mat')
    FOVY = sio.loadmat('D:/science_work/Ring_code_0213/Pytorch_PSF_FOVKPN_3X3Split_weight/Y.mat')
    FOVX = FOVX['X'].astype(np.float32)[np.newaxis, :, :, np.newaxis]
    FOVY = FOVY['Y'].astype(np.float32)[np.newaxis, :, :, np.newaxis]
    FOVX = torch.from_numpy(FOVX)
    FOVY = torch.from_numpy(FOVY)
    FOVX = FOVX.transpose(2, 3).transpose(1, 2)
    FOVY = FOVY.transpose(2, 3).transpose(1, 2)

    # 加载PSF和gamma
    psf_data, psf_number = PSF_load(args.psf_dir, 'mat')
    psf_data_np = np.array(psf_data)
    count_psf1 = 0
    psf_s_start = [0, 5, 10]
    psf_t_start = [0, 6, 12]
    psf_in = np.zeros((9, 42, 61, 61))
    for s in range(3):
        for t in range(3):
            count_psf2 = 0
            for i in range(6):
                for j in range(7):
                    # print((psf_s_start[s]+i)*19+psf_t_start[t]+j)
                    psf_in[count_psf1, count_psf2, :, :] = psf_data_np[(psf_s_start[s] + i) * 19 + psf_t_start[t] + j, :, :]
                    count_psf2 += 1
            count_psf1 += 1
    psf_data_torch = torch.from_numpy(psf_in)
    wnrgamma_torch = torch.tensor(args.wnrgamma)

    # 加载weight
    weight = generate_weight(6, 7, 128)
    weight_torch = torch.from_numpy(weight)

    if torch.cuda.is_available():
        psf_data_torch, wnrgamma_torch, weight_torch = psf_data_torch.cuda(), wnrgamma_torch.cuda(), weight_torch.cuda()

    # Build model
    model_wnr = deconWnr(psf_data_torch, wnrgamma_torch, weight_torch)
    model = make_model(input_channel=3, output_channel=1, args=args)
    model.initialize_weights()

    # 读取预训练权重
    if torch.cuda.is_available():
        print('Use {} GPU, which order is {:s}th'.format(torch.cuda.device_count(), args.gpu))
        if os.path.exists(args.ckpt_dir + '/model_%04d_dict.pth' % args.init_epoch):
            model_dict = torch.load(args.ckpt_dir + '/model_%04d_dict.pth' % args.init_epoch)
            model.load_state_dict(model_dict)
            model_wnr_dict = torch.load(args.ckpt_dir + '/model_wnr_%04d_dict.pth' % args.init_epoch)
            model_wnr.load_state_dict(model_wnr_dict)
        model_wnr = model_wnr.cuda()
        model = model.cuda()

    for ind in range(test_number):

         img_test = test_data[ind].copy() / 65535.
         print(np.max(img_test[:,:,:]))
         img_test = img_test[np.newaxis,:,:,:]
         input_patch = img_test[:, :, 2432:, :]
         label = img_test[:, :, 0:2432, :]
         input = torch.from_numpy(input_patch)
         input = input.transpose(2, 3).transpose(1, 2)
         if torch.cuda.is_available():
             input = input.cuda()

         # 数据提前拼接后分割
         input_pad = F.pad(input, pad=(args.batch * 2, args.batch * 2, args.batch * 2, args.batch * 2), mode='replicate')
         count_split = 0
         s_start = [0, 640, 1280]
         t_start = [0, 768, 1536]
         input_pad_in = torch.zeros(1, 9, 1280, 1408)
         input_in = torch.zeros(1, 9, 768, 896)
         FOVX_in = torch.zeros(1, 9, 768, 896)
         FOVY_in = torch.zeros(1, 9, 768, 896)
         for s in range(3):
             for t in range(3):
                 input_pad_in[:, count_split, :, :] = input_pad[:, 0, s_start[s]:s_start[s] + 1280, t_start[t]:t_start[t] + 1408]
                 input_in[:, count_split, :, :] = input[:, 0, s_start[s]:s_start[s] + 768, t_start[t]:t_start[t] + 896]
                 FOVX_in[:, count_split, :, :] = FOVX[:, 0, s_start[s]:s_start[s] + 768, t_start[t]:t_start[t] + 896]
                 FOVY_in[:, count_split, :, :] = FOVY[:, 0, s_start[s]:s_start[s] + 768, t_start[t]:t_start[t] + 896]
                 count_split += 1

         if torch.cuda.is_available():
             input_pad_in, input_in, FOVX_in, FOVY_in = input_pad_in.cuda(), input_in.cuda(), FOVX_in.cuda(), FOVY_in.cuda()

         wnr_output = np.zeros((2048, 2432))
         net_output = np.zeros((2048, 2432))
         for i in range(9):
             # print(i)
             wnr_input_pad_in = input_pad_in[:, i:i + 1, :, :]
             wnr_input_in = input_in[:, i:i + 1, :, :]
             wnroutput = model_wnr(wnr_input_pad_in, wnr_input_in, i)
             net_FOVXTrain_in = FOVX_in[:, i:i + 1, :, :]
             net_FOVYTrain_in = FOVY_in[:, i:i + 1, :, :]
             net_input = torch.cat([wnroutput, net_FOVXTrain_in], axis=1)
             net_input = torch.cat([net_input, net_FOVYTrain_in], axis=1)
             if torch.cuda.is_available():
                 wnroutput = wnroutput.cuda()

             test_out = model(net_input)

             test_out = torch.clamp(test_out, 0., 1.)  # 输出应在0-1之间截断，避免tensorboard显示时溢出
             wnroutput = torch.clamp(wnroutput, 0., 1.)
             test_out.detach_()
             wnroutput.detach_()
             rgb_out = test_out.cpu().numpy()
             wnr_out = wnroutput.cpu().numpy()

             deblurred = np.clip(rgb_out[0, 0, :, :], 0., 1.)
             wnrdeblurred = np.clip(wnr_out[0, 0, :, :], 0., 1.)
             if i == 0:
                 wnr_output[0:640, 0:768] = wnrdeblurred[0:640, 0:768]
                 net_output[0:640, 0:768] = deblurred[0:640, 0:768]
             if i == 1:
                 wnr_output[0:640, 768:1536] = wnrdeblurred[0:640, 0:768]
                 net_output[0:640, 768:1536] = deblurred[0:640, 0:768]
             if i == 2:
                 wnr_output[0:640, 1536:2432] = wnrdeblurred[0:640, 0:896]
                 net_output[0:640, 1536:2432] = deblurred[0:640, 0:896]
             if i == 3:
                 wnr_output[640:1280, 0:768] = wnrdeblurred[0:640, 0:768]
                 net_output[640:1280, 0:768] = deblurred[0:640, 0:768]
             if i == 4:
                 wnr_output[640:1280, 768:1536] = wnrdeblurred[0:640, 0:768]
                 net_output[640:1280, 768:1536] = deblurred[0:640, 0:768]
             if i == 5:
                 wnr_output[640:1280, 1536:2432] = wnrdeblurred[0:640, 0:896]
                 net_output[640:1280, 1536:2432] = deblurred[0:640, 0:896]
             if i == 6:
                 wnr_output[1280:2048, 0:768] = wnrdeblurred[0:768, 0:768]
                 net_output[1280:2048, 0:768] = deblurred[0:768, 0:768]
             if i == 7:
                 wnr_output[1280:2048, 768:1536] = wnrdeblurred[0:768, 0:768]
                 net_output[1280:2048, 768:1536] = deblurred[0:768, 0:768]
             if i == 8:
                 wnr_output[1280:2048, 1536:2432] = wnrdeblurred[0:768, 0:896]
                 net_output[1280:2048, 1536:2432] = deblurred[0:768, 0:896]

         input_out = input.cpu().numpy().transpose((0, 2, 3, 1))
         input_out_binning = average_pool(np.squeeze(input_out), 2)

         wnr_output_binning = average_pool(wnr_output, 2)
         net_output_binning = average_pool(net_output, 2)
         label_output_binning = average_pool(np.squeeze(np.squeeze(label)), 2)
         wnr_output_binning = wnr_output_binning * 65535
         net_output_binning = net_output_binning * 65535
         input_out_binning = input_out_binning * 65535
         label_output_binning = label_output_binning * 65535

         if not os.path.exists(args.recovery_dir):
             os.makedirs(args.recovery_dir)

         model_output_mat_name = args.recovery_dir + '/blurred%05d.mat' % ind
         sio.savemat(model_output_mat_name, {'data': input_out_binning.astype(np.uint16)})
         model_output_mat_name = args.recovery_dir + '/deblurred%05d.mat' % ind
         sio.savemat(model_output_mat_name, {'data': net_output_binning.astype(np.uint16)})
         model_output_mat_name = args.recovery_dir + '/wnrdeblured%05d.mat' % ind
         sio.savemat(model_output_mat_name, {'data': wnr_output_binning.astype(np.uint16)})
         model_output_mat_name = args.recovery_dir + '/label%05d.mat' % ind
         sio.savemat(model_output_mat_name, {'data': label_output_binning.astype(np.uint16)})


