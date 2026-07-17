import os
import numpy as np
import scipy.io as sio
import torch
import lpips
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage import img_as_float
from skimage import io

# ---------- 参数设置 ----------
folder = 'D:/science_work/Ring_code_0213/Rec_data/251225/L1_1_L2_1'
num_files = 100
max_val = 65535.0

# ---------- 初始化 LPIPS 模型 ----------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
lpips_fn = lpips.LPIPS(net='alex').to(device)

# ---------- 存储结果 ----------
lpips_vals = []
niqe_vals = []

# ---------- 循环处理 ----------
for i in range(num_files):
    deblurred_name = f'deblurred{i:05d}.mat'
    label_name     = f'label{i:05d}.mat'

    deblurred_path = os.path.join(folder, deblurred_name)
    label_path     = os.path.join(folder, label_name)

    if not os.path.isfile(deblurred_path) or not os.path.isfile(label_path):
        print(f'文件缺失: {deblurred_name} 或 {label_name}，跳过')
        continue

    # 读取 .mat
    d_data = sio.loadmat(deblurred_path)
    l_data = sio.loadmat(label_path)

    # 提取第一个非系统变量
    img_deblurred = None
    img_label = None
    for k in d_data:
        if not k.startswith('__'):
            img_deblurred = d_data[k]
            break
    for k in l_data:
        if not k.startswith('__'):
            img_label = l_data[k]
            break

    if img_deblurred is None or img_label is None:
        print(f'{deblurred_name} 或 {label_name} 无有效变量，跳过')
        continue

    # 检查尺寸和类型
    if img_deblurred.shape != (1024, 1216) or img_deblurred.dtype != np.uint16:
        print(f'{deblurred_name} 尺寸或类型错误，跳过')
        continue
    if img_label.shape != (1024, 1216) or img_label.dtype != np.uint16:
        print(f'{label_name} 尺寸或类型错误，跳过')
        continue

    # 归一化到 [0,1]
    img_deblurred_float = img_deblurred.astype(np.float32) / max_val
    img_label_float     = img_label.astype(np.float32) / max_val

    # ---- 计算 LPIPS ----
    # 灰度扩展为 3 通道 RGB，形状 (1, 3, H, W)，值域 [0,1]
    deblur_rgb = np.stack([img_deblurred_float]*3, axis=0)   # (3, H, W)
    label_rgb  = np.stack([img_label_float]*3, axis=0)

    # 转 torch tensor，增加 batch 维度
    deblur_tensor = torch.from_numpy(deblur_rgb).unsqueeze(0).to(device)  # (1, 3, H, W)
    label_tensor  = torch.from_numpy(label_rgb).unsqueeze(0).to(device)

    # 计算 LPIPS，normalize=True 会缩放到 [-1,1]
    with torch.no_grad():
        lpips_val = lpips_fn(deblur_tensor, label_tensor, normalize=True).item()
    lpips_vals.append(lpips_val)

    # ---- 计算 NIQE（无参考，只处理去模糊图像）----
    # scikit-image 的 niqe 要求输入是 0-255 的灰度图像或 0-1 的浮点图像（会自动处理）
    # 这里直接传入 float (0-1)
    niqe_val = niqe(img_deblurred_float)  # 注意：需先 from skimage.metrics import niqe
    niqe_vals.append(niqe_val)

    print(f'已处理 {i+1}/{num_files} : LPIPS={lpips_val:.4f}, NIQE={niqe_val:.4f}')

# ---------- 输出结果 ----------
if lpips_vals:
    lpips_mean = np.mean(lpips_vals)
    lpips_std  = np.std(lpips_vals)
    niqe_mean  = np.mean(niqe_vals)
    niqe_std   = np.std(niqe_vals)

    print(f'\n===== 最终结果 =====')
    print(f'LPIPS 平均值: {lpips_mean:.4f}, 标准差: {lpips_std:.4f}')
    print(f'NIQE 平均值: {niqe_mean:.4f}, 标准差: {niqe_std:.4f}')
else:
    print('没有成功计算任何指标，请检查文件。')