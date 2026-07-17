import scipy.io as sio
import torch
import os
import numpy as np
from model.__init__ import make_model
from model.fov_kpn import deconWnr
import torch.nn.functional as F
from option.option import args

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

    print(len(input_image))
    print("data load finished!")
    return (input_image, len(file_lists))

def PSFFOVKPNInit(CKdir1, CKdir2, psf_dir, wnrgamma):
    # 加载FOV Data
    FOVX = sio.loadmat('./X.mat')
    FOVY = sio.loadmat('./Y.mat')
    FOVX = FOVX['X'].astype(np.float32)[np.newaxis, :, :, np.newaxis]
    FOVY = FOVY['Y'].astype(np.float32)[np.newaxis, :, :, np.newaxis]
    FOVX = torch.from_numpy(FOVX)
    FOVY = torch.from_numpy(FOVY)
    FOVX = FOVX.transpose(2, 3).transpose(1, 2)
    FOVY = FOVY.transpose(2, 3).transpose(1, 2)

    # 加载PSF和gamma
    psf_data, psf_number = PSF_load(psf_dir, 'mat')
    psf_data_np = np.array(psf_data)
    count_psf1 = 0
    psf_s_start = [0, 5, 10]
    psf_t_start = [0, 6, 12]
    psf_in = np.zeros((9, 42, 121, 121))
    for s in range(3):
        for t in range(3):
            count_psf2 = 0
            for i in range(6):
                for j in range(7):
                    # print((psf_s_start[s]+i)*19+psf_t_start[t]+j)
                    psf_in[count_psf1, count_psf2, :, :] = psf_data_np[(psf_s_start[s] + i) * 19 + psf_t_start[t] + j,
                                                           :, :]
                    count_psf2 += 1
            count_psf1 += 1
    psf_data_torch = torch.from_numpy(psf_in)
    wnrgamma_torch = torch.tensor(wnrgamma)

    if torch.cuda.is_available():
        psf_data_torch, wnrgamma_torch = psf_data_torch.cuda(), wnrgamma_torch.cuda()

    # Build model
    model_wnr = deconWnr(psf_data_torch, wnrgamma_torch)
    model = make_model(input_channel=3, output_channel=1, args=args)
    model.initialize_weights()

    # 读取预训练权重
    if torch.cuda.is_available():
        if os.path.exists(CKdir1):
            model_dict1 = torch.load(CKdir1)
            model_wnr.load_state_dict(model_dict1)
        model_wnr = model_wnr.cuda()
    if torch.cuda.is_available():
        if os.path.exists(CKdir2):
            model_dict2 = torch.load(CKdir2)
            model.load_state_dict(model_dict2)
        model = model.cuda()

    return model_wnr, model, FOVX, FOVY

def PSFFOVKPNImage(data, FOVX, FOVY, batch, model_wnr, model):
    data = data / 65535.
    data = data.astype(np.float32)
    img_test = data[np.newaxis, :, :]
    img_test = img_test[np.newaxis, :, :, :]
    input_patch = img_test

    input = torch.from_numpy(input_patch)
    if torch.cuda.is_available():
        input = input.cuda()

    # 数据提前拼接后分割
    input_pad = F.pad(input, pad=(batch, batch, batch, batch), mode='replicate')
    count_split = 0
    s_start = [0, 640, 1280]
    t_start = [0, 768, 1536]
    input_pad_in = torch.zeros(1, 9, 1024, 1152)
    input_in = torch.zeros(1, 9, 768, 896)
    FOVX_in = torch.zeros(1, 9, 768, 896)
    FOVY_in = torch.zeros(1, 9, 768, 896)
    for s in range(3):
        for t in range(3):
            input_pad_in[:, count_split, :, :] = input_pad[:, 0, s_start[s]:s_start[s] + 1024, t_start[t]:t_start[t] + 1152]
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

    wnr_output_16 = wnr_output * 65535
    net_output_16 = net_output * 65535

    wnroutput = wnr_output_16.astype(np.uint16)
    netoutput = net_output_16.astype(np.uint16)

    return wnroutput, netoutput

def average_pool(mar, n):
    sz = np.shape(mar)
    _mar = mar.reshape(sz[0] // n, n, sz[1] // n, n)
    Mar = _mar.mean(axis=(1, 3))
    return Mar

if __name__ == '__main__':
    CKdir1 = './ckpt_dir_PSFFovKpn521_300/model_wnr_0000_dict.pth'
    CKdir2 = './ckpt_dir_PSFFovKpn521_300/model_0000_dict.pth'
    psfdir = 'F:/VIS_PSF Data/PSFtest240517/PSF_Small/'
    recovery_dir = 'F:/RealTest/0528/TestDebug2/AssemblyRecovery'
    gamma = 0.0427
    scale = 0.5
    args.NetName = 'FOV-KPN'

    model_wnr, model, FOVX, FOVY = PSFFOVKPNInit(CKdir1, CKdir2, psfdir, gamma)

    # 导入数据
    data_temp = sio.loadmat('F:/RealTest/0528/TestDebug2/BGAll_020.mat')['BGAll'].astype(np.float32)
    image_temp = data_temp[:, 2, :, 0:2432]
    image_temp = np.squeeze(image_temp)
    image2binning = average_pool(image_temp, 2)

    data_min1 = np.min(image_temp)
    data_max1 = np.max(image_temp)
    frame_new = (image_temp - data_min1) / (data_max1 - data_min1) * 65535
    frame_block_new = frame_new[:, 0:2432]
    Wnrdata, Udata = PSFFOVKPNImage(frame_block_new * scale, FOVX, FOVY, 128, model_wnr, model)
    Wnrdata2binning = average_pool(Wnrdata, 2)
    Udata2binning = average_pool(Udata, 2)

    if not os.path.exists(recovery_dir):
        os.makedirs(recovery_dir)

    model_output_mat_name = recovery_dir + '/blurred%05d.mat' % 0
    sio.savemat(model_output_mat_name, {'data': image2binning.astype(np.uint16)})
    model_output_mat_name = recovery_dir + '/deblurred%05d.mat' % 0
    sio.savemat(model_output_mat_name, {'data': Udata2binning.astype(np.uint16)})
    model_output_mat_name = recovery_dir + '/wnrdeblured%05d.mat' % 0
    sio.savemat(model_output_mat_name, {'data': Wnrdata2binning.astype(np.uint16)})

