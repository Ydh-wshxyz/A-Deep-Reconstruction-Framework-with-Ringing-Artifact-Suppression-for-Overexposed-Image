import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
import torch
import torch.fft

from DefConv2d.DeformableConv2d import ModulatedDeformConvPack2 as DCN  # 需验证与DCNv2的差异


# try:
#     from dcn.deform_conv import ModulatedDeformConvPack2 as DCN
# except ImportError:
#     raise ImportError('Failed to import DCNv2 module.')

def psf2otf(psf, h, w):
    # PSF归一化，防止PSF优化后总和不唯一, 0422添加
    psfNormal = psf / torch.sum(psf)
    psf_shape = psf.shape
    h_psf = psf_shape[0]
    w_psf = psf_shape[1]
    psfNormalpad = F.pad(psfNormal, pad=(
    int(np.ceil((h - h_psf) / 2)), int(np.floor((h - h_psf) / 2)), int(np.ceil((w - w_psf) / 2)),
    int(np.floor((w - w_psf) / 2))), mode='constant', value=0)
    otf = torch.fft.fft2(torch.fft.fftshift(psfNormalpad))
    return otf


def psf2otf1(psf, h, w):
    # PSF归一化，防止PSF优化后总和不唯一, 0422添加
    psfNormal = psf
    psf_shape = psf.shape
    h_psf = psf_shape[0]
    w_psf = psf_shape[1]
    psfNormalpad = F.pad(psfNormal, pad=(
    int(np.ceil((h - h_psf) / 2)), int(np.floor((h - h_psf) / 2)), int(np.ceil((w - w_psf) / 2)),
    int(np.floor((w - w_psf) / 2))), mode='constant', value=0)
    otf = torch.fft.fft2(torch.fft.fftshift(psfNormalpad))
    return otf


def deconvolve_wnr(blur, psf, gamma):
    # 获取图像和PSF的形状
    img_shape = blur.shape
    lambda_param = 0
    # print(f"Image shape: {img_shape}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 计算PSF对应的OTF
    otf = psf2otf(psf, img_shape[2], img_shape[3])

    # 生成拉普拉斯算子并转换为频域（OTF）
    laplacian = torch.tensor([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=torch.float32, device=device)  # 放到正确的device

    laplacian_otf = psf2otf1(laplacian, img_shape[2], img_shape[3])  # 确保laplacian_otf在指定的device上
    # print(laplacian_otf)

    laplacian_otf_temp1 = lambda_param * torch.abs(laplacian_otf) ** 2
    # print(laplacian_otf_temp1)
    # 计算维纳滤波器，加入拉普拉斯算子的影响
    wiener_filter_temp = torch.conj(otf) / (torch.abs(otf) ** 2 + gamma + laplacian_otf_temp1)
    # print(wiener_filter_temp.shape)

    # 对wiener_filter_temp做尺寸扩展，以适配图像的batch和channel维度
    wiener_filter_temp = torch.unsqueeze(wiener_filter_temp, dim=0)  # 增加batch维度
    wiener_filter_temp = torch.unsqueeze(wiener_filter_temp, dim=0)  # 增加channel维度
    # print(wiener_filter_temp.shape)

    # 如果图像有2个通道，复制wiener_filter_temp
    if img_shape[0] == 2:
        wiener_filter = torch.cat([wiener_filter_temp, wiener_filter_temp], dim=0)
    else:
        wiener_filter = wiener_filter_temp
    # print(wiener_filter.shape)

    # 反卷积操作（维纳滤波反卷积）
    output = torch.abs(torch.fft.ifft2(wiener_filter * torch.fft.fft2(blur)))

    # print(output.shape)
    return output


class deconWnr(nn.Module):
    def __init__(self, psf, gamma, weight):
        super(deconWnr, self).__init__()
        self.psf = torch.tensor(psf)
        self.gammafinal = nn.Parameter(torch.tensor(gamma), requires_grad=True)
        self.weight = torch.tensor(weight)
        # self.gammafinal = gammatemp ** 2

    def forward(self, inputs_pad, inputs, s):

        out = torch.zeros_like(inputs)
        img_shape = inputs.shape
        count = 0
        for i in range(6):
            for j in range(7):
                out_sum = torch.zeros(img_shape[0], img_shape[1], img_shape[2] + 128 * 2, img_shape[3] + 128 * 2)
                out_sum = out_sum.cuda()

                psftemp = self.psf[s, count, :, :]
                gammatemp = self.gammafinal ** 2
                weighttemp1 = self.weight[i, j, :, :]
                weighttemp2 = torch.unsqueeze(weighttemp1, dim=0)
                weighttemp2 = torch.unsqueeze(weighttemp2, dim=0)
                if img_shape[0] == 2:
                    weighttemp = torch.cat([weighttemp2, weighttemp2], dim=0)
                else:
                    weighttemp = weighttemp2
                BL_image_temp = inputs_pad[:, :, i * 128:i * 128 + 640, j * 128:j * 128 + 640]
                out_temp = deconvolve_wnr(BL_image_temp, psftemp, gammatemp)
                out_block = out_temp[:, :, 128:512, 128:512]
                # print(out_block.shape)
                # 假设 out_block 的形状是 torch.Size([2, 1, 0, 0, 640, 640])

                out_block_temp = weighttemp * out_block
                out_sum[:, :, i * 128:(i + 3) * 128, j * 128:(j + 3) * 128] = out_block_temp
                out = out + out_sum[:, :, 128:-128, 128:-128]
                count = count + 1

        # output = torch.clamp(out, 1E-20, 1.)
        output = out

        return output

    def show_gamma(self):

        return self.gammafinal ** 2


# ==============================================================================#
class ResBlock(nn.Module):

    def __init__(self, input_channel=32, output_channel=32):
        super().__init__()
        self.in_channel = input_channel
        self.out_channel = output_channel
        if self.in_channel != self.out_channel:
            self.conv0 = nn.Conv2d(input_channel, output_channel, 1, 1)
        self.conv1 = nn.Conv2d(output_channel, output_channel, 3, 1, 1)
        self.conv2 = nn.Conv2d(output_channel, output_channel, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.initialize_weights()

    def forward(self, x):
        if self.in_channel != self.out_channel:
            x = self.conv0(x)
        conv1 = self.lrelu(self.conv1(x))
        conv2 = self.conv2(conv1)
        out = x + conv2
        return out

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()


class RSABlock(nn.Module):

    def __init__(self, input_channel=32, output_channel=32, offset_channel=32):
        super().__init__()
        self.in_channel = input_channel
        self.out_channel = output_channel
        if self.in_channel != self.out_channel:
            self.conv0 = nn.Conv2d(input_channel, output_channel, 1, 1)
        self.dcnpack = DCN(output_channel, output_channel, 3, stride=1, padding=1, dilation=1, deformable_groups=8,
                           extra_offset_mask=True, offset_in_channel=offset_channel)
        self.conv1 = nn.Conv2d(output_channel, output_channel, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.initialize_weights()

    def forward(self, x, offset):
        if self.in_channel != self.out_channel:
            x = self.conv0(x)
        fea = self.lrelu(self.dcnpack([x, offset]))
        out = self.conv1(fea) + x
        return out

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()


class OffsetBlock(nn.Module):

    def __init__(self, input_channel=32, offset_channel=32, last_offset=False):
        super().__init__()
        self.offset_conv1 = nn.Conv2d(input_channel, offset_channel, 3, 1, 1)  # concat for diff
        if last_offset:
            self.offset_conv2 = nn.Conv2d(offset_channel * 2, offset_channel, 3, 1, 1)  # concat for offset
        self.offset_conv3 = nn.Conv2d(offset_channel, offset_channel, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.initialize_weights()

    def forward(self, x, last_offset=None):
        offset = self.lrelu(self.offset_conv1(x))
        if last_offset is not None:
            last_offset = F.interpolate(last_offset, scale_factor=2, mode='bilinear', align_corners=False)  # 线性采样
            offset = self.lrelu(self.offset_conv2(torch.cat([offset, last_offset * 2], dim=1)))
        offset = self.lrelu(self.offset_conv3(offset))
        return offset

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()


class FOVBlock(nn.Module):

    def __init__(self, input_channel=3, output_channel=32, nf=32):
        super().__init__()
        # spatial attention
        self.sAtt_fov_1 = nn.Conv2d(input_channel - 1, nf, 3, 1, 1,
                                    bias=True)  # 输入两个视场通道，输出32个通道，卷积和3X3，stride=1, padding=1
        self.sAtt_fov_2 = nn.Conv2d(nf, nf, 1, 1, bias=True)  # 输入32个通道，输出32个通道，卷积1X1，stride=1

        self.sAtt_img_1 = nn.Conv2d(1, nf, 3, 1, 1, bias=True)
        self.sAtt_img_2 = nn.Conv2d(nf, nf, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, imgwzfov):
        B, N, H, W = imgwzfov.size()  # img with fov information
        img = imgwzfov[:, 0:1, :, :]
        fov = imgwzfov[:, 1:N, :, :]

        att_img = self.lrelu(self.sAtt_img_1(img))
        att_fov = self.lrelu(self.sAtt_fov_1(fov))
        att_img = self.lrelu(self.sAtt_img_2(att_img))
        att_fov = self.lrelu(self.sAtt_fov_2(att_fov))

        att_fov = torch.sigmoid(att_fov)

        att_fea = att_img * att_fov + att_img
        return att_fea


class KernelConv(nn.Module):
    """
    the class of kernel prediction
    """

    def __init__(self, kernel_size=[5]):
        super(KernelConv, self).__init__()
        self.kernel_size = sorted(kernel_size)

    def _convert_dict(self, core, batch_size, color, height, width):
        """
        make sure the core to be a dict, generally, only one kind of kernel size is suitable for the func.
        :param core: shape: batch_size*(N*K*K)*height*width
        :return: core_out, a dict
        """
        core_out = {}
        core = core.view(batch_size, color, -1, height, width)
        core_out[self.kernel_size[0]] = core[:, 0:self.kernel_size[0] ** 2, ...]
        return core_out

    def forward(self, inputs, core, rate=1):
        """
        compute the pred image according to the input and core
        :param inputs: [batch_size, color, height, width]
        :param core: [batch_size, dict(kernel), color, height, width]
        """
        img_stack = []
        pred_img = []
        batch_size, color, height, width = inputs.size()
        # print(batch_size, color, height, width)
        core = self._convert_dict(core, batch_size, color, height, width)

        K = self.kernel_size[0]
        padding_num = (K // 2) * rate
        inputs_pad = F.pad(inputs, [padding_num, padding_num, padding_num, padding_num])
        for i in range(0, K):
            for j in range(0, K):
                img_stack.append(inputs_pad[..., i * rate:i * rate + height, j * rate:j * rate + width])
        img_stack = torch.stack(img_stack, dim=2)
        pred_img.append(torch.sum(
            core[K].mul(img_stack), dim=2, keepdim=False
        ))
        pred_img = torch.stack(pred_img, dim=0)
        pred_img = pred_img.squeeze(0)
        return pred_img


# ===============================================================================#
class FOVKPN(nn.Module):

    def __init__(self, input_channel=3, output_channel=1, n_channel=32, offset_channel=32,
                 fov_att=False, kernel_size=[5], color=False):
        super().__init__()
        output_kernel_channel = (3 if color else 1) * np.sum(np.array(kernel_size) ** 2)

        self.fovblock = FOVBlock(input_channel, n_channel, n_channel)
        self.res1 = ResBlock(n_channel, n_channel)
        self.down1 = nn.Conv2d(n_channel, n_channel * 2, 2, 2)
        self.res2 = ResBlock(n_channel * 2, n_channel * 2)
        self.down2 = nn.Conv2d(n_channel * 2, n_channel * 4, 2, 2)
        self.res3 = ResBlock(n_channel * 4, n_channel * 4)
        self.down3 = nn.Conv2d(n_channel * 4, n_channel * 8, 2, 2)
        self.res4 = ResBlock(n_channel * 8, n_channel * 8)

        self.offset4 = OffsetBlock(n_channel * 8, offset_channel, False)
        self.dres4 = RSABlock(n_channel * 8, n_channel * 8, offset_channel)

        self.up3 = nn.ConvTranspose2d(n_channel * 8, n_channel * 4, 2, 2)
        self.dconv3_1 = nn.Conv2d(n_channel * 8, n_channel * 4, 1, 1)
        self.offset3 = OffsetBlock(n_channel * 4, offset_channel, True)
        self.dres3 = RSABlock(n_channel * 4, n_channel * 4, offset_channel)

        self.up2 = nn.ConvTranspose2d(n_channel * 4, n_channel * 2, 2, 2)
        self.dconv2_1 = nn.Conv2d(n_channel * 4, n_channel * 2, 1, 1)
        self.offset2 = OffsetBlock(n_channel * 2, offset_channel, True)
        self.dres2 = RSABlock(n_channel * 2, n_channel * 2, offset_channel)

        self.up1 = nn.ConvTranspose2d(n_channel * 2, n_channel, 2, 2)
        self.dconv1_1 = nn.Conv2d(n_channel * 2, n_channel, 1, 1)
        self.offset1 = OffsetBlock(n_channel, offset_channel, True)
        self.dres1 = RSABlock(n_channel, n_channel, offset_channel)

        self.outc = nn.Conv2d(n_channel, output_kernel_channel, kernel_size=1, stride=1, padding=0)

        self.kernel_pred = KernelConv(kernel_size)

        self.conv_final = nn.Conv2d(in_channels=4, out_channels=output_channel, kernel_size=3, stride=1, padding=1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        att_fov = self.fovblock(x)
        conv1 = self.res1(att_fov)
        pool1 = self.lrelu(self.down1(conv1))
        conv2 = self.res2(pool1)
        pool2 = self.lrelu(self.down2(conv2))
        conv3 = self.res3(pool2)
        pool3 = self.lrelu(self.down3(conv3))
        conv4 = self.res4(pool3)

        L4_offset = self.offset4(conv4, None)
        dconv4 = self.dres4(conv4, L4_offset)

        up3 = torch.cat([self.up3(dconv4), conv3], 1)
        up3 = self.dconv3_1(up3)
        L3_offset = self.offset3(up3, L4_offset)
        dconv3 = self.dres3(up3, L3_offset)

        up2 = torch.cat([self.up2(dconv3), conv2], 1)
        up2 = self.dconv2_1(up2)
        L2_offset = self.offset2(up2, L3_offset)
        dconv2 = self.dres2(up2, L2_offset)

        up1 = torch.cat([self.up1(dconv2), conv1], 1)
        up1 = self.dconv1_1(up1)
        L1_offset = self.offset1(up1, L2_offset)
        dconv1 = self.dres1(up1, L1_offset)

        core = self.outc(dconv1)

        pred1 = self.kernel_pred(x[:, 0:1, :, :], core, rate=1)
        pred2 = self.kernel_pred(x[:, 0:1, :, :], core, rate=2)
        pred3 = self.kernel_pred(x[:, 0:1, :, :], core, rate=3)
        pred4 = self.kernel_pred(x[:, 0:1, :, :], core, rate=4)

        pred_cat = torch.cat([torch.cat([torch.cat([pred1, pred2], dim=1), pred3], dim=1), pred4], dim=1)

        out = self.conv_final(pred_cat) + x[:, 0:1, :, :]
        return out

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                # torch.nn.init.xavier_uniform_(m.weight.data)
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                # m.weight.data.fill_(1)
                # m.bias.data.zero_()

                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()


# ==============================================================================#


if __name__ == '__main__':
    input = torch.zeros([2, 3, 1024, 1216]).cuda()
    model = FOVKPN(input_channel=3, output_channel=1, n_channel=32, offset_channel=32).cuda()
    output = model(input)
    print(output.shape)