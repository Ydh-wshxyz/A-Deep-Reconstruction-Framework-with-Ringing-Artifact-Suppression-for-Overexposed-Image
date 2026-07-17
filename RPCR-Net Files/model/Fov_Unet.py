import torch
import torch.nn as nn
import numpy as np


class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm):
        super(unetConv2, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_size, out_size, 3, 1, padding=1, bias=True), nn.BatchNorm2d(out_size), nn.LeakyReLU()
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_size, out_size, 3, 1, padding=1, bias=True), nn.BatchNorm2d(out_size), nn.LeakyReLU()
            )
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, padding=1, bias=True), nn.LeakyReLU())
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, padding=1, bias=True), nn.LeakyReLU())

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class FOVBlock(nn.Module):

    def __init__(self, input_channel=3, nf=32):
        # spatial attention
        super(FOVBlock,self).__init__()  # 后加的

        self.sAtt_fov_1 = nn.Conv2d(input_channel - 1, nf, 3, 1, 1, bias=True)
        self.sAtt_fov_2 = nn.Conv2d(nf, nf, 1, 1, bias=True)

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


class unetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv):
        super(unetUp, self).__init__()
        if is_deconv:
            self.up = nn.Sequential(
                nn.ConvTranspose2d(in_size, out_size, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.LeakyReLU(0.2))
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        return torch.cat([inputs1, outputs2], 1)


class unet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, is_deconv=True,  is_batchnorm=False):
        super(unet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm

        filters = [32, 64, 128, 256, 512]

        # downsampling
        # self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.conv1 = FOVBlock(self.in_channels, filters[0])
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)

        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)

        # upsampling
        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.conv5 = unetConv2(filters[3]*2, filters[3], self.is_batchnorm)

        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.conv6 = unetConv2(filters[2]*2, filters[2], self.is_batchnorm)

        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.conv7 = unetConv2(filters[1]*2, filters[1], self.is_batchnorm)

        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)
        self.conv8 = unetConv2(filters[0]*2, filters[0], self.is_batchnorm)

        # final conv 
        self.final = nn.Conv2d(filters[0], out_channels, 3,1,1,bias=True)

    def forward(self, inputs):

        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)
        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)
        up4 = self.up_concat4(conv4, center)
        conv5 = self.conv5(up4)
        up3 = self.up_concat3(conv3, conv5)
        conv6 = self.conv6(up3)
        up2 = self.up_concat2(conv2, conv6)
        conv7 = self.conv7(up2)
        up1 = self.up_concat1(conv1, conv7)
        conv8 = self.conv8(up1)

        final = self.final(conv8)

        out = torch.add(final,inputs[:,0:1,:,:])

        return out

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                #torch.nn.init.xavier_normal_(m.weight.data)
                #torch.nn.init.xavier_uniform_(m.weight.data)
                m.weight.data.normal_(0.0, 0.02)
                #torch.nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                #m.weight.data.fill_(1)
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()


if __name__ == '__main__':
    input = torch.zeros([2,3,1024,1216]).cuda()
    model = unet(in_channels=3, out_channels=1, is_deconv=True,  is_batchnorm=False).cuda()
    output = model(input)
    print(output.shape)