import cv2
from tensorboardX import SummaryWriter
from utils import *
from model.__init__ import make_model
from loss.__init__ import select_loss
from Dataset import dataset_load, sensor_noise, PSF_load
import time
from torchvision import transforms as transforms
import torch
import numpy as np
import scipy.io as sio
from model.fov_kpn import deconWnr
import torch.nn.functional as F
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

def train_model(input_pad_in, input_in, label_in, FOVXTrain_in, FOVYTrain_in, model, model_wnr, optimizer_wnr, optimizer_net, criterion):
    if torch.cuda.is_available():
        input_pad_in, input_in, label_in, FOVXTrain_in, FOVYTrain_in = input_pad_in.cuda(), input_in.cuda(), label_in.cuda(), FOVXTrain_in.cuda(), FOVYTrain_in.cuda()
    # 数据重新分割，进行9次循环迭代
    loss_sum = 0
    P_loss_sum = 0
    wnrgamma_sum = 0
    label_out = torch.zeros_like(input_in)
    output_out = torch.zeros_like(input_in)
    wnroutput_out = torch.zeros_like(input_in)
    for i in range(9):
        # 初始化
        model.train()  # 启用dropout和batch normalization层，然后计算模型的输出和损失，进行反向传播，并使用优化器更新模型的权重和偏差。
        model.zero_grad()  # 清空模型参数的梯度
        optimizer_net.zero_grad()
        model_wnr.train()
        model_wnr.zero_grad()
        optimizer_wnr.zero_grad()  # 清空模型参数的梯度，以确保每次迭代的梯度计算都是基于当前小批量数据的，而不会受之前迭代的影响。这是为了避免在优化过程中梯度的不正确累积。
        wnr_input_pad_in = input_pad_in[:, i:i+1, :, :]
        wnr_input_in = input_in[:, i:i+1, :, :]
        # 参数传入模型
        wnroutput = model_wnr(wnr_input_pad_in, wnr_input_in, i)
        # 融入FOV数据
        net_FOVXTrain_in = FOVXTrain_in[:, i:i+1, :, :]
        net_FOVYTrain_in = FOVYTrain_in[:, i:i+1, :, :]
        wnroutput = torch.cat([wnroutput, net_FOVXTrain_in], axis=1)
        wnroutput = torch.cat([wnroutput, net_FOVYTrain_in], axis=1)
        # 数据增强
        net_label_in = label_in[:, i:i+1, :, :]
        image_temp = torch.concat([wnroutput, net_label_in], axis=1)
        # image = transforms.RandomHorizontalFlip(p=0.5)(image_temp)
        # image = transforms.RandomVerticalFlip(p=0.5)(image)
        image = image_temp

        wnroutput = image[:, 0:3, :, :]
        label = image[:, 3:, :, :, ]
        if torch.cuda.is_available():
            wnroutput, label = wnroutput.cuda(), label.cuda()
        # Unet去噪
        # start_time = time.time()
        output = model(wnroutput)
        # end_time = time.time()
        # print(f"Runtime: {end_time - start_time} seconds")
        wnrgamma_temp = model_wnr.show_gamma()
        wnrgamma_sum += wnrgamma_temp

        label_cal = torch.tile(label, (1, 3, 1, 1))
        output_cal = torch.tile(output, (1, 3, 1, 1))

        # calculate loss
        loss, P_loss = criterion(output_cal, label_cal)
        loss_temp = loss.item()
        P_loss_temp = P_loss.item()
        loss_sum += loss_temp
        P_loss_sum += P_loss_temp
        loss.backward()
        optimizer_net.step()  # 更新优化变量参数
        optimizer_wnr.step()

        label_out[:, i:i+1, :, :] = label
        output_out[:, i:i+1, :, :] = output
        wnroutput_out[:, i, :, :] = wnroutput[:, 0, :, :]

    loss_out = loss_sum / 9
    P_loss_out = P_loss_sum / 9
    wnrgamma_out = wnrgamma_sum / 9

    return loss_out, P_loss_out, output_out, wnroutput_out, label_out, wnrgamma_out

def eval_log(file_name, input_in, output, wnroutput, labelout, wnrgammaout, psnr1, psnr2, psnr3, loss_temp, P_loss_temp, count, writer, index):

    writer.add_image(file_name + '/Input', labelout[:, index:index+1, :, :], count, dataformats='NCHW')
    writer.add_image(file_name + '/Blur', input_in[:, index:index+1, :, :], count, dataformats='NCHW')
    writer.add_image(file_name + '/Wnroutput', wnroutput[:, index:index+1, :, :], count, dataformats='NCHW')
    writer.add_image(file_name + '/Output', output[:, index:index+1, :, :], count, dataformats='NCHW')

    writer.add_scalar(file_name + '/intial_psnr', psnr1, count)
    writer.add_scalar(file_name + '/wnr_psnr', psnr2, count)
    writer.add_scalar(file_name + '/final_psnr', psnr3, count)
    writer.add_scalar(file_name + '/valid_loss', loss_temp, count)
    writer.add_scalar(file_name + '/p_loss', P_loss_temp, count)
    writer.add_scalar(file_name + '/wnr_gamma', wnrgammaout, count)

def eval_log2(file_name, input_pad_in, input_in, label_in, FOVXTrain_in, FOVYTrain_in, model, model_wnr, criterion, count, writer):

    if torch.cuda.is_available():
        input_pad_in, input_in, label_in, FOVXTrain_in, FOVYTrain_in = input_pad_in.cuda(), input_in.cuda(), label_in.cuda(), FOVXTrain_in.cuda(), FOVYTrain_in.cuda()
    output_out = torch.zeros_like(input_in)
    wnroutput_out = torch.zeros_like(input_in)
    loss_val = 0
    P_loss = 0
    for i in range(9):
        model_wnr.eval()
        model.eval()  # 在评估模型性能时禁用dropout和batch normalization的函数。它还可以用于在测试数据上进行推理。这个方法不会更新模型的权重和偏差。

        wnr_input_pad_in = input_pad_in[:, i:i + 1, :, :]
        wnr_input_in = input_in[:, i:i + 1, :, :]
        # 参数传入模型
        wnr_test_out = model_wnr(wnr_input_pad_in, wnr_input_in, i)
        net_FOVXTrain_in = FOVXTrain_in[:, i:i + 1, :, :]
        net_FOVYTrain_in = FOVYTrain_in[:, i:i + 1, :, :]
        wnr_test_out = torch.cat([wnr_test_out, net_FOVXTrain_in], axis=1)
        wnr_test_out = torch.cat([wnr_test_out, net_FOVYTrain_in], axis=1)
        label = label_in[:, i:i + 1, :, :]

        if torch.cuda.is_available():
            wnr_test_out, label = wnr_test_out.cuda(), label.cuda()
        # Unet去噪
        test_out = model(wnr_test_out)
        wnrgamma_out = model_wnr.show_gamma()

        test_out_cal = torch.tile(test_out, (1, 3, 1, 1))
        label_cal = torch.tile(label, (1, 3, 1, 1))

        loss_val_temp, P_loss_temp = criterion(test_out_cal, label_cal)
        loss_val += loss_val_temp.item()
        P_loss += P_loss_temp.item()

        output_out[:, i:i + 1, :, :] = test_out
        wnroutput_out[:, i, :, :] = wnr_test_out[:, 0, :, :]

    loss_val = loss_val / 9
    P_loss = P_loss / 9

    output_out.detach_()
    wnroutput_out.detach_()

    rgb_out = output_out.cpu().numpy()
    wnr_out = wnroutput_out.cpu().numpy()
    clean = label_in.cpu().numpy()
    blur = input_in.cpu().numpy()

    psnr1 = 0
    psnr2 = 0
    psnr3 = 0
    count_psnr = 0
    for s in range(rgb_out.shape[0]):
        for t in range(rgb_out.shape[1]):
            blurred = np.clip(blur[s, t, :, :], 0., 1.)
            deblurred = np.clip(rgb_out[s, t, :, :], 0., 1.)
            wnrdeblurred = np.clip(wnr_out[s, t, :, :], 0., 1.)
            cleaned = np.clip(clean[s, t, :, :], 0., 1.)

            psnr1 += compare_psnr(cleaned, blurred)
            psnr2 += compare_psnr(cleaned, wnrdeblurred)
            psnr3 += compare_psnr(cleaned, deblurred)
            count_psnr += 1

    psnr1 = psnr1 / count_psnr
    psnr2 = psnr2 / count_psnr
    psnr3 = psnr3 / count_psnr

    writer.add_image(file_name + '/Input', label_in[:, 0:1, :, :], count, dataformats='NCHW')
    writer.add_image(file_name + '/Blur', input_in[:, 0:1, :, :], count, dataformats='NCHW')
    writer.add_image(file_name + '/Wnroutput', output_out[:, 0:1, :, :], count, dataformats='NCHW')
    writer.add_image(file_name + '/Output', wnroutput_out[:, 0:1, :, :], count, dataformats='NCHW')

    writer.add_scalar(file_name + '/intial_psnr', psnr1, count)
    writer.add_scalar(file_name + '/wnr_psnr', psnr2, count)
    writer.add_scalar(file_name + '/final_psnr', psnr3, count)
    writer.add_scalar(file_name + '/valid_loss', loss_val, count)
    writer.add_scalar(file_name + '/p_loss', P_loss, count)
    writer.add_scalar(file_name + '/wnr_gamma', wnrgamma_out, count)

def eval_epoch(model,model_wnr,test_number,test_data,criterion,writer,epoch, args):
    model_wnr.eval()
    model.eval()

    # 加载FOV Data
    FOVX = sio.loadmat('D:/science_work/Ring_code_0213/Pytorch_PSF_FOVKPN_3X3Split_weight/X.mat')
    FOVY = sio.loadmat('D:/science_work/Ring_code_0213/Pytorch_PSF_FOVKPN_3X3Split_weight/Y.mat')
    FOVX = FOVX['X'].astype(np.float32)[np.newaxis, :, :, np.newaxis]
    FOVY = FOVY['Y'].astype(np.float32)[np.newaxis, :, :, np.newaxis]
    FOVX = torch.from_numpy(FOVX)
    FOVY = torch.from_numpy(FOVY)
    FOVX = FOVX.transpose(2, 3).transpose(1, 2)
    FOVY = FOVY.transpose(2, 3).transpose(1, 2)

    loss_val_sum = 0
    P_loss_val_sum = 0
    psnr1_sum = 0
    psnr2_sum = 0
    psnr3_sum = 0
    for ind in range(test_number):
        # print(ind)
        img_test = test_data[ind].copy()
        img_test = img_test[np.newaxis, :, :, :]
        input_patch = img_test[:, :, 2432:, :]
        label_patch = img_test[:, :, 0:2432, :]

        input = torch.from_numpy(input_patch)
        label = torch.from_numpy(label_patch)

        input = input.transpose(2, 3).transpose(1, 2)
        label = label.transpose(2, 3).transpose(1, 2)

        if torch.cuda.is_available():
            input, label, FOVX, FOVY = input.cuda(), label.cuda(), FOVX.cuda(), FOVY.cuda()

        input, label = sensor_noise(input, label, args.a_poission, args.b_sqrt)  # 模拟噪声

        # 数据提前拼接后分割
        input_pad = F.pad(input, pad=(args.batch * 2, args.batch * 2, args.batch * 2, args.batch * 2), mode='replicate')
        count_split = 0
        s_start = [0, 640, 1280]
        t_start = [0, 768, 1536]
        input_pad_in = torch.zeros(1, 9, 1280, 1408)
        input_in = torch.zeros(1, 9, 768, 896)
        label_in = torch.zeros(1, 9, 768, 896)
        FOVX_in = torch.zeros(1, 9, 768, 896)
        FOVY_in = torch.zeros(1, 9, 768, 896)
        for s in range(3):
            for t in range(3):
                input_pad_in[:, count_split, :, :] = input_pad[:, 0, s_start[s]:s_start[s] + 1280, t_start[t]:t_start[t] + 1408]
                input_in[:, count_split, :, :] = input[:, 0, s_start[s]:s_start[s] + 768, t_start[t]:t_start[t] + 896]
                label_in[:, count_split, :, :] = label[:, 0, s_start[s]:s_start[s] + 768, t_start[t]:t_start[t] + 896]
                FOVX_in[:, count_split, :, :] = FOVX[:, 0, s_start[s]:s_start[s] + 768, t_start[t]:t_start[t] + 896]
                FOVY_in[:, count_split, :, :] = FOVY[:, 0, s_start[s]:s_start[s] + 768, t_start[t]:t_start[t] + 896]
                count_split += 1

        if torch.cuda.is_available():
            input_pad_in, input_in, label_in, FOVX_in, FOVY_in = input_pad_in.cuda(), input_in.cuda(), label_in.cuda(), FOVX_in.cuda(), FOVY_in.cuda()

        loss_val = 0
        P_loss = 0
        psnr1 = 0  # blur与gt的PSNR
        psnr2 = 0  # wnrdeblur与gt的PSNR
        psnr3 = 0  # deblur与gt的PSNR
        # PSNR1 = np.zeros((9,1))
        # PSNR2 = np.zeros((9,1))
        # PSNR3 = np.zeros((9,1))
        for i in range(9):
            # print(i)
            wnr_input_pad_in = input_pad_in[:, i:i + 1, :, :]
            wnr_input_in = input_in[:, i:i + 1, :, :]
            wnroutput = model_wnr(wnr_input_pad_in, wnr_input_in, i)
            net_FOVXTrain_in = FOVX_in[:, i:i + 1, :, :]
            net_FOVYTrain_in = FOVY_in[:, i:i + 1, :, :]
            net_input = torch.cat([wnroutput, net_FOVXTrain_in], axis=1)
            net_input = torch.cat([net_input, net_FOVYTrain_in], axis=1)
            label_out = label_in[:, i:i + 1, :, :]
            if torch.cuda.is_available():
                wnroutput, label_out = wnroutput.cuda(), label_out.cuda()

            # start_time = time.time()
            test_out = model(net_input)
            # end_time = time.time()
            # print(f"Runtime: {end_time - start_time} seconds")
            wnrgamma_out = model_wnr.show_gamma()

            test_out_cal = torch.tile(test_out, (1, 3, 1, 1))
            label_cal = torch.tile(label_out, (1, 3, 1, 1))
            loss_val_temp, P_loss_temp = criterion(test_out_cal, label_cal)
            loss_val += loss_val_temp.item()
            P_loss += P_loss_temp.item()

            test_out = torch.clamp(test_out, 0., 1.)  # 输出应在0-1之间截断，避免tensorboard显示时溢出
            wnroutput = torch.clamp(wnroutput, 0., 1.)
            test_out.detach_()
            wnroutput.detach_()
            rgb_out = test_out.cpu().numpy()
            wnr_out = wnroutput.cpu().numpy()
            clean = label_out.cpu().numpy()
            blur = wnr_input_in.cpu().numpy()

            blurred = np.clip(blur[0, 0, :, :], 0., 1.)
            deblurred = np.clip(rgb_out[0, 0, :, :], 0., 1.)
            wnrdeblurred = np.clip(wnr_out[0, 0, :, :], 0., 1.)
            cleaned = np.clip(clean[0, 0, :, :], 0., 1.)


            psnr1 += compare_psnr(cleaned, blurred)
            psnr2 += compare_psnr(cleaned, wnrdeblurred)
            psnr3 += compare_psnr(cleaned, deblurred)

        psnr1 = psnr1 / 9
        psnr2 = psnr2 / 9
        psnr3 = psnr3 / 9
        loss_val = loss_val / 9
        P_loss = P_loss / 9

        # compute loss
        loss_val_sum += loss_val
        P_loss_val_sum += P_loss
        psnr1_sum = psnr1_sum + psnr1
        psnr2_sum = psnr2_sum + psnr2
        psnr3_sum = psnr3_sum + psnr3

    loss_val_avg = loss_val_sum / test_number
    psnr1_avg = psnr1_sum / test_number
    psnr2_avg = psnr2_sum / test_number
    psnr3_avg = psnr3_sum / test_number
    print('Validating: {:0>3} , loss: {:.8f}, PSNR1: {:4.4f}, PSNR2: {:4.4f}, PSNR3: {:4.4f}'.format(test_number, loss_val_avg, psnr1_avg, psnr2_avg, psnr3_avg))
    writer.add_scalars('Loss_group', {'valid_loss': loss_val_avg}, epoch)

    writer.add_scalars('Gamma_group', {'valid_Wnrgamma': wnrgamma_out}, epoch)
    writer.add_scalars('PSNR_group', {'valid_intialPSNR': psnr1_avg}, epoch)
    writer.add_scalars('PSNR_group', {'valid_wnrPSNR': psnr2_avg}, epoch)
    writer.add_scalars('PSNR_group', {'valid_totalPSNR': psnr3_avg}, epoch)

def eval_test_log(file_name, test_data, test_number, model, model_wnr, criterion, count, writer, args, index1, index2):
    model_wnr.eval()
    model.eval()

    # 加载FOV Data
    FOVX = sio.loadmat('D:/science_work/Ring_code_0213/Pytorch_PSF_FOVKPN_3X3Split_weight/X.mat')
    FOVY = sio.loadmat('D:/science_work/Ring_code_0213/Pytorch_PSF_FOVKPN_3X3Split_weight/Y.mat')
    FOVX = FOVX['X'].astype(np.float32)[np.newaxis, :, :, np.newaxis]
    FOVY = FOVY['Y'].astype(np.float32)[np.newaxis, :, :, np.newaxis]
    FOVX = torch.from_numpy(FOVX)
    FOVY = torch.from_numpy(FOVY)
    FOVX = FOVX.transpose(2, 3).transpose(1, 2)
    FOVY = FOVY.transpose(2, 3).transpose(1, 2)

    loss_val_sum = 0
    P_loss_val_sum = 0
    psnr1_sum = 0
    psnr2_sum = 0
    psnr3_sum = 0
    for ind in range(test_number):
        # print(ind)
        img_test = test_data[ind].copy()
        img_test = img_test[np.newaxis,:,:,:]
        input_patch = img_test[:,:,2432:,:]
        label_patch = img_test[:,:,0:2432,:]

        input = torch.from_numpy(input_patch)
        label = torch.from_numpy(label_patch)

        input = input.transpose(2,3).transpose(1,2)
        label = label.transpose(2,3).transpose(1,2)

        if torch.cuda.is_available():
            input, label, FOVX, FOVY = input.cuda(), label.cuda(), FOVX.cuda(), FOVY.cuda()

        input, label = sensor_noise(input, label, args.a_poission, args.b_sqrt)  # 模拟噪声

        # 数据提前拼接后分割
        input_pad = F.pad(input, pad=(args.batch * 2, args.batch * 2, args.batch * 2, args.batch * 2), mode='replicate')
        count_split = 0
        s_start = [0, 640, 1280]
        t_start = [0, 768, 1536]
        input_pad_in = torch.zeros(1, 9, 1280, 1408)
        input_in = torch.zeros(1, 9, 768, 896)
        label_in = torch.zeros(1, 9, 768, 896)
        FOVX_in = torch.zeros(1, 9, 768, 896)
        FOVY_in = torch.zeros(1, 9, 768, 896)
        for s in range(3):
            for t in range(3):
                # print(s, t)
                input_pad_in[:, count_split, :, :] = input_pad[:, 0, s_start[s]:s_start[s] + 1280,t_start[t]:t_start[t] + 1408]
                input_in[:, count_split, :, :] = input[:, 0, s_start[s]:s_start[s] + 768, t_start[t]:t_start[t] + 896]
                label_in[:, count_split, :, :] = label[:, 0, s_start[s]:s_start[s] + 768, t_start[t]:t_start[t] + 896]
                FOVX_in[:, count_split, :, :] = FOVX[:, 0, s_start[s]:s_start[s] + 768,t_start[t]:t_start[t] + 896]
                FOVY_in[:, count_split, :, :] = FOVY[:, 0, s_start[s]:s_start[s] + 768,t_start[t]:t_start[t] + 896]
                count_split += 1

        if torch.cuda.is_available():
            input_pad_in, input_in, label_in, FOVX_in, FOVY_in = input_pad_in.cuda(), input_in.cuda(), label_in.cuda(), FOVX_in.cuda(), FOVY_in.cuda()

        loss_val = 0
        P_loss = 0
        psnr1 = 0  # blur与gt的PSNR
        psnr2 = 0  # wnrdeblur与gt的PSNR
        psnr3 = 0  # deblur与gt的PSNR
        for i in range(9):
            # print(i)
            wnr_input_pad_in = input_pad_in[:, i:i + 1, :, :]
            wnr_input_in = input_in[:, i:i + 1, :, :]
            wnroutput = model_wnr(wnr_input_pad_in, wnr_input_in, i)
            net_FOVXTrain_in = FOVX_in[:, i:i + 1, :, :]
            net_FOVYTrain_in = FOVY_in[:, i:i + 1, :, :]
            net_input = torch.cat([wnroutput, net_FOVXTrain_in], axis=1)
            net_input = torch.cat([net_input, net_FOVYTrain_in], axis=1)
            label_out = label_in[:, i:i + 1, :, :]
            if torch.cuda.is_available():
                wnroutput, label_out = wnroutput.cuda(), label_out.cuda()

            # start_time = time.time()
            test_out = model(net_input)
            # end_time = time.time()
            # print(f"Runtime: {end_time - start_time} seconds")
            wnrgamma_out = model_wnr.show_gamma()

            test_out_cal = torch.tile(test_out, (1, 3, 1, 1))
            label_cal = torch.tile(label_out, (1, 3, 1, 1))
            loss_val_temp, P_loss_temp = criterion(test_out_cal, label_cal)
            loss_val += loss_val_temp.item()
            P_loss += P_loss_temp.item()

            if ind == index1 and i == index2:
                writer.add_image(file_name + '/Input', label_out, count, dataformats='NCHW')
                writer.add_image(file_name + '/Blur', wnr_input_in, count, dataformats='NCHW')
                writer.add_image(file_name + '/Wnroutput', wnroutput, count, dataformats='NCHW')
                writer.add_image(file_name + '/Output', test_out, count, dataformats='NCHW')

            test_out = torch.clamp(test_out, 0., 1.)  # 输出应在0-1之间截断，避免tensorboard显示时溢出
            wnroutput = torch.clamp(wnroutput, 0., 1.)
            test_out.detach_()
            wnroutput.detach_()
            rgb_out = test_out.cpu().numpy()
            wnr_out = wnroutput.cpu().numpy()
            clean = label_out.cpu().numpy()
            blur = wnr_input_in.cpu().numpy()

            blurred = np.clip(blur[0, 0, :, :], 0., 1.)
            deblurred = np.clip(rgb_out[0, 0, :, :], 0., 1.)
            wnrdeblurred = np.clip(wnr_out[0, 0, :, :], 0., 1.)
            cleaned = np.clip(clean[0, 0, :, :], 0., 1.)

            psnr1 += compare_psnr(cleaned, blurred)
            psnr2 += compare_psnr(cleaned, wnrdeblurred)
            psnr3 += compare_psnr(cleaned, deblurred)

        psnr1 = psnr1 / 9
        psnr2 = psnr2 / 9
        psnr3 = psnr3 / 9
        loss_val = loss_val / 9
        P_loss = P_loss / 9

        # compute loss
        loss_val_sum += loss_val
        P_loss_val_sum += P_loss
        psnr1_sum = psnr1_sum + psnr1
        psnr2_sum = psnr2_sum + psnr2
        psnr3_sum = psnr3_sum + psnr3

    loss_val_avg = loss_val_sum / test_number
    P_loss_val_avg = P_loss_val_sum / test_number
    psnr1_avg = psnr1_sum / test_number
    psnr2_avg = psnr2_sum / test_number
    psnr3_avg = psnr3_sum / test_number

    writer.add_scalar(file_name + '/intial_psnr', psnr1_avg, count)
    writer.add_scalar(file_name + '/wnr_psnr', psnr2_avg, count)
    writer.add_scalar(file_name + '/final_psnr', psnr3_avg, count)
    writer.add_scalar(file_name + '/valid_loss', loss_val_avg, count)
    writer.add_scalar(file_name + '/p_loss', P_loss_val_avg, count)
    writer.add_scalar(file_name + '/wnr_gamma', wnrgamma_out, count)

def train_function(args):

    params = initialize_params(args)
    
    # Dataset
    train_data, train_number = dataset_load(args.train_dir, 'mat')   #  训练数据   uint16
    test_data, test_number = dataset_load(args.test_dir, 'mat')   # 测试数据     uint16

    # 加载FOV Data
    FOVX = sio.loadmat('D:/science_work/Ring_code_0213/Pytorch_PSF_FOVKPN_3X3Split_weight/X.mat')
    FOVY = sio.loadmat('D:/science_work/Ring_code_0213/Pytorch_PSF_FOVKPN_3X3Split_weight/Y.mat')
    FOVX = FOVX['X'].astype(np.float32)[np.newaxis, :, :, np.newaxis]
    FOVY = FOVY['Y'].astype(np.float32)[np.newaxis, :, :, np.newaxis]


    weight = generate_weight(6, 7, 128)
    weight_torch = torch.from_numpy(weight)

    # 存储路径
    create_dir(args.log_dir)
    create_dir(args.ckpt_dir)

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
                    psf_in[count_psf1, count_psf2, :, :] = psf_data_np[(psf_s_start[s]+i)*19+psf_t_start[t]+j, :, :]
                    count_psf2 += 1
            count_psf1 += 1
    psf_data_torch = torch.from_numpy(psf_in)
    wnrgamma_torch = torch.tensor(args.wnrgamma)

    if torch.cuda.is_available():
        psf_data_torch, wnrgamma_torch, weight_torch = psf_data_torch.cuda(), wnrgamma_torch.cuda(), weight_torch.cuda()

    # Build model
    model_wnr = deconWnr(psf_data_torch, wnrgamma_torch, weight_torch)
    model = make_model(input_channel=3, output_channel=1, args=args)
    model.initialize_weights()
    
    # define loss
    criterion = select_loss(args)

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
        criterion = criterion.cuda()

    # 设置优化器
    optimizer_wnr = torch.optim.Adam(model_wnr.parameters(), lr=args.wnrlr)
    optimizer_net = torch.optim.Adam(model.parameters(), lr=args.netlr)
    writer = SummaryWriter(args.log_dir)
    
    # Training
    count = 0
    for epoch in range(args.n_epoch):
        loss_sum = 0
        #step_lr_adjust(optimizer, epoch, init_lr=args.lr, step_size=5, gamma=0.9)
        print('Epoch {}, lr {}'.format(epoch + 1, optimizer_net.param_groups[0]['lr']))
        
        start_time = time.time()

        batch_number = np.floor(train_number / params['batchSize']).astype(np.int16) 
        flag_number = np.random.permutation(train_number)

        loss_sum = 0.0
        psnr1_sum = 0.0
        psnr2_sum = 0.0
        psnr3_sum = 0.0
        index = 0
        index_block = 0
        wnrgammsum = 0.0

        # traning
        for i in range(batch_number):

            # 以固定的batch进行训练
            img = train_data[flag_number[i*params['batchSize']]].copy()
            img = img[np.newaxis,:,:,:]
            FOVXTrain = FOVX
            FOVYTrain = FOVY
            FovXtemp = FOVX
            FovYtemp = FOVY
            for j in range(params['batchSize']-1):
                img_temp = train_data[flag_number[i*params['batchSize']+j+1]]
                img_temp = img_temp[np.newaxis,:,:,:]
                img = np.concatenate([img,img_temp],axis = 0)
                FOVXTrain = np.concatenate([FOVXTrain, FovXtemp], axis=0)
                FOVYTrain = np.concatenate([FOVYTrain, FovYtemp], axis=0)
            
            # 对数据进行拆分且进行常规数据增强操作
            BL = img[:,:,2432:,:]
            GT = img[:,:,0:2432,:]

            BL = torch.from_numpy(BL)
            GT = torch.from_numpy(GT)
            FOVXTrain = torch.from_numpy(FOVXTrain)
            FOVYTrain = torch.from_numpy(FOVYTrain)

            input = BL.transpose(2, 3).transpose(1, 2)
            label = GT.transpose(2, 3).transpose(1, 2)
            FOVXTrain = FOVXTrain.transpose(2, 3).transpose(1, 2)
            FOVYTrain = FOVYTrain.transpose(2, 3).transpose(1, 2)

            # 加噪声处理
            if torch.cuda.is_available():
                input, label, FOVXTrain, FOVYTrain = input.cuda(), label.cuda(), FOVXTrain.cuda(), FOVYTrain.cuda()

            input, label = sensor_noise(input, label, args.a_poission, args.b_sqrt)  # 模拟噪声       这里面会将16位数据转换到[0 1]

            # 数据提前拼接后再分割
            input_pad = F.pad(input, pad=(args.batch*2, args.batch*2, args.batch*2, args.batch*2), mode='replicate')
            count_split = 0
            s_start = [0, 640, 1280]
            t_start = [0, 768, 1536]
            input_pad_in = torch.zeros(params['batchSize'], 9, 1280, 1408)
            input_in = torch.zeros(params['batchSize'], 9, 768, 896)
            label_in = torch.zeros(params['batchSize'], 9, 768, 896)
            FOVXTrain_in = torch.zeros(params['batchSize'], 9, 768, 896)
            FOVYTrain_in = torch.zeros(params['batchSize'], 9, 768, 896)
            for s in range(3):
                for t in range(3):
                    input_pad_in[:, count_split, :, :] = input_pad[:, 0, s_start[s]:s_start[s] + 1280, t_start[t]:t_start[t] + 1408]
                    input_in[:, count_split, :, :] = input[:, 0, s_start[s]:s_start[s] + 768, t_start[t]:t_start[t] + 896]
                    label_in[:, count_split, :, :] = label[:, 0, s_start[s]:s_start[s] + 768, t_start[t]:t_start[t] + 896]
                    FOVXTrain_in[:, count_split, :, :] = FOVXTrain[:, 0, s_start[s]:s_start[s] + 768, t_start[t]:t_start[t] + 896]
                    FOVYTrain_in[:, count_split, :, :] = FOVYTrain[:, 0, s_start[s]:s_start[s] + 768, t_start[t]:t_start[t] + 896]
                    count_split += 1

            # 模型训练
            loss_temp, P_loss_temp, output, wnroutput, labelout, wnrgammaout = train_model(input_pad_in, input_in, label_in, FOVXTrain_in, FOVYTrain_in, model, model_wnr, optimizer_wnr, optimizer_net, criterion)

            output = torch.clamp(output, 0., 1.)  # 输出应在0-1之间截断，避免tensorboard显示时溢出
            wnroutput = torch.clamp(wnroutput, 0., 1.)

            output.detach_()
            wnroutput.detach_()
            labelout.detach_()

            rgb_out = output.cpu().numpy()
            wnr_out = wnroutput.cpu().numpy()
            clean = labelout.cpu().detach().numpy()
            blur = input_in.cpu().numpy()
            blur_clean = label_in.cpu().numpy()

            psnr1 = 0  # blur与gt的PSNR
            psnr2 = 0  # wnrdeblur与gt的PSNR
            psnr3 = 0  # deblur与gt的PSNR
            count_psnr = 0

            for s in range(rgb_out.shape[0]):
                for t in range(rgb_out.shape[1]):
                    blurred = np.clip(blur[s,t,:,:], 0., 1.)
                    blurred_cleaned = np.clip(blur_clean[s,t,:,:], 0., 1.)
                    deblurred = np.clip(rgb_out[s,t,:,:], 0., 1.)
                    wnrdeblurred = np.clip(wnr_out[s,t,:,:], 0., 1.)
                    cleaned = np.clip(clean[s,t,:,:], 0., 1.)

                    psnr1 += compare_psnr(blurred_cleaned, blurred)
                    psnr2 += compare_psnr(cleaned, wnrdeblurred)
                    psnr3 += compare_psnr(cleaned, deblurred)
                    count_psnr += 1

            psnr1 = psnr1 / count_psnr
            psnr2 = psnr2 / count_psnr
            psnr3 = psnr3 / count_psnr

            loss_sum = loss_sum + loss_temp
            psnr1_sum = psnr1_sum + psnr1
            psnr2_sum = psnr2_sum + psnr2
            psnr3_sum = psnr3_sum + psnr3

            loss_avg = loss_sum / (i + 1)
            psnr1_avg = psnr1_sum / (i + 1)
            psnr2_avg = psnr2_sum / (i + 1)
            psnr3_avg = psnr3_sum / (i + 1)

            wnrgammsum = wnrgammsum + wnrgammaout
            wnrgammavg = wnrgammsum / (i + 1)

            print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.8f} PSNR1: {:.4f} PSNR2: {:.4f} PSNR3: {:.4f} Gamma: {:.8f} Time: {:4.4f}s".format(
                    epoch + 1, args.n_epoch, i + 1, batch_number, loss_avg, psnr1_avg, psnr2_avg, psnr3_avg, wnrgammaout, time.time() - start_time))
            start_time = time.time()

            if count % args.log_freq == 0:
                if index_block >= 9:  # 测试的时候batch始终等于1
                    index_block = 0

                eval_log('train', input_in, output, wnroutput, labelout, wnrgammaout, psnr1, psnr2, psnr3, loss_temp, P_loss_temp, count, writer, index_block)

                if index >= test_number:  # 测试的时候batch始终等于1
                    index = 0

                eval_test_log('test', test_data, test_number, model, model_wnr, criterion, count, writer, args, index, index_block)
                index += 1
                index_block += 1

            count += 1

        # Record train loss
        writer.add_scalars('Loss_group', {'train_loss': loss_avg}, epoch)
        writer.add_scalars('Gamma_group', {'train_wnrgamma': wnrgammavg}, epoch)
        writer.add_scalars('PSNR_group', {'train_intialPSNR': psnr1_avg}, epoch)
        writer.add_scalars('PSNR_group', {'train_wnrPSNR': psnr2_avg}, epoch)
        writer.add_scalars('PSNR_group', {'train_totalPSNR': psnr3_avg}, epoch)

        # save model
        torch.save(model.state_dict(), os.path.join(args.ckpt_dir, 'model_%04d_dict.pth' % (epoch)))
        torch.save(model_wnr.state_dict(), os.path.join(args.ckpt_dir, 'model_wnr_%04d_dict.pth' % (epoch)))
        # # validation each epoch
        eval_epoch(model, model_wnr, test_number, test_data, criterion, writer, epoch, args)

    writer.close()


