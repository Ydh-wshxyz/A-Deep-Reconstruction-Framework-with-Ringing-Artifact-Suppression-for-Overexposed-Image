from model.fov_kpn import FOVKPN
from model.Fov_Unet import unet

def make_model(input_channel, output_channel, args):
    if args.NetName == 'Fov_UNet':
        print('Training with Fov_Unet')
        return unet(input_channel, output_channel, is_deconv=True, is_batchnorm=False)
    if args.NetName == 'FOV-KPN':
        print('Training with FOV-KPN')
        return FOVKPN(input_channel, output_channel, n_channel=32, offset_channel=32,
                      fov_att=False, kernel_size=[5], color=False)


