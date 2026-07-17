import torch
import torch.nn.functional as F
from math import exp
from torch import nn
from torchvision import models
from torch.autograd import Variable
# from torchvision.models import VGG19_Weights
from utils import normalize_tensor_transform

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class L2_wz_Perceptual(nn.Module):
    def __init__(self, args):
        super(L2_wz_Perceptual, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.per_loss = PerceptualLossWithRinging()
        self.CONTENT_WEIGHT = args.Perceptual_weight

    def forward(self, out_images, target_images):
        # MSELoss
        image_loss = self.mse_loss(out_images, target_images)
        # Perceptual Loss

        out_images_norm, target_images_norm = normalize_tensor_transform(out_images, target_images)

        content_loss = self.per_loss(out_images_norm, target_images_norm)

        return image_loss +  self.CONTENT_WEIGHT * content_loss.data , self.CONTENT_WEIGHT *content_loss.data

#----------------------------------------------------------------------------------

class PerceptualLossWithRinging(nn.Module):
    def __init__(self, L=3, input_channels=3, device='cuda'):  # 添加 device 参数
        super(PerceptualLossWithRinging, self).__init__()

        # 确保设备是 cuda 或 cpu
        self.device = device

        # 定义VGG19特征提取层
        features = models.vgg19(pretrained=True).features
        self.to_relu_2_2 = nn.Sequential()
        self.to_relu_3_2 = nn.Sequential()

        self.input_channels = input_channels  # 假设输入图像是3通道

        for x in range(9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 14):
            self.to_relu_3_2.add_module(str(x), features[x])

        # 固定VGG网络的参数
        for param in self.parameters():
            param.requires_grad = False

        self.conv_input = nn.Conv2d(in_channels=self.input_channels, out_channels=3, kernel_size=3, padding=1)

        self.L = L

        # 振铃伪影的卷积核初始化
        self.H_h = (1/32) * torch.tensor([[-1, -2, 0, 2, 1], [-2, -4, 0, 4, 2], [-1, -2, 0, 2, 1]]).float()
        self.H_v = self.H_h.T

        # 使卷积核与输入图像通道数匹配
        self.H_h = self.H_h.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)  # [3, 3, 3, 5]
        self.H_v = self.H_v.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)  # [3, 3, 3, 5]

        # 确保卷积核在正确的设备上
        self.H_h = self.H_h.to(self.device)
        self.H_v = self.H_v.to(self.device)

        # 定义 MSE Loss
        self.mse_loss = nn.MSELoss()

    def compute_gradients(self, I):
        # Sobel 卷积核 (1通道)
        sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(
            0).cuda()
        sobel_y = sobel_x.transpose(2, 3).cuda()

        # 对每个通道独立计算梯度
        Ix = []
        Iy = []
        for c in range(I.size(1)):
            Ix_c = F.conv2d(I[:, c:c + 1], sobel_x, padding=1)
            Iy_c = F.conv2d(I[:, c:c + 1], sobel_y, padding=1)
            Ix.append(Ix_c)
            Iy.append(Iy_c)

        # 将每个通道的结果拼接起来
        Ix = torch.cat(Ix, dim=1)
        Iy = torch.cat(Iy, dim=1)

        # 计算梯度幅值
        G = torch.sqrt(Ix ** 2 + Iy ** 2)
        return G

    def local_max_gradient(self, G, patch_size=3):
        # 展开为局部区域并计算最大值
        unfolded = F.unfold(G, kernel_size=patch_size, stride=1, padding=patch_size // 2)
        unfolded_max, _ = unfolded.max(dim=1)
        unfolded_max = unfolded_max.view(G.size(0), G.size(2), G.size(3))
        return unfolded_max

    def forward(self, pred_img, targ_img):
        # 确保输入图像和目标图像在同一设备上
        pred_img = pred_img.to(self.device)
        targ_img = targ_img.to(self.device)

        # 对输入进行处理
        pred_img = self.conv_input(pred_img)
        targ_img = self.conv_input(targ_img)

        h_relu_2_2_pred_img = self.to_relu_2_2(pred_img)
        h_relu_2_2_targ_img = self.to_relu_2_2(targ_img)
        style_loss_2_2 = self.mse_loss(h_relu_2_2_pred_img, h_relu_2_2_targ_img)

        h_relu_3_2_pred_img = self.to_relu_3_2(h_relu_2_2_pred_img)
        h_relu_3_2_targ_img = self.to_relu_3_2(h_relu_2_2_targ_img)
        style_loss_3_2 = self.mse_loss(h_relu_3_2_pred_img, h_relu_3_2_targ_img)

        content_loss_tol = style_loss_2_2 + style_loss_3_2
        ringing_loss = self.measure_ringing_artifacts(targ_img, pred_img)

        # 计算局部最大梯度 (LMG)
        G_pred = self.compute_gradients(pred_img)
        G_targ = self.compute_gradients(targ_img)

        # 计算局部最大梯度
        LMG_pred = self.local_max_gradient(G_pred)
        LMG_targ = self.local_max_gradient(G_targ)

        # LMG 损失 (简单地计算预测和目标之间的差异)
        lmg_loss = self.mse_loss(LMG_pred, LMG_targ)
        lmg_loss = lmg_loss * 0.1

        total_loss = content_loss_tol + ringing_loss/0.005 * 0.00000005 + lmg_loss
        return total_loss

    def measure_ringing_artifacts(self, b, x_hat):
        E = 0
        for l in range(self.L):
            if l > 0:
                b_l = F.interpolate(b, scale_factor=1 / (2 ** (l)), mode='bilinear', align_corners=False)
                x_hat_l = F.interpolate(x_hat, scale_factor=1 / (2 ** (l)), mode='bilinear', align_corners=False)
            else:
                b_l = b
                x_hat_l = x_hat

            # 卷积操作
            H_h_x_hat_l = F.conv2d(x_hat_l, self.H_h, padding=2, groups=3)
            H_v_x_hat_l = F.conv2d(x_hat_l, self.H_v, padding=2, groups=3)

            H_h_b_l = F.conv2d(b_l, self.H_h, padding=2, groups=3)
            H_v_b_l = F.conv2d(b_l, self.H_v, padding=2, groups=3)

            # 计算delta
            delta_l_h = torch.max(torch.zeros_like(H_h_x_hat_l), torch.abs(H_h_x_hat_l) - torch.abs(H_h_b_l))
            delta_l_v = torch.max(torch.zeros_like(H_v_x_hat_l), torch.abs(H_v_x_hat_l) - torch.abs(H_v_b_l))


            # 确保delta_l_v和delta_l_h尺寸匹配
            delta_l_v = F.interpolate(delta_l_v, size=delta_l_h.shape[2:], mode='bilinear', align_corners=False)

            delta_l = delta_l_h + delta_l_v
            E_l = delta_l.sum()
            E += E_l

        return E


