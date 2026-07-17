from loss.loss import *
import torch


def select_loss(args):
    global criterion

    if args.t_loss == 'L2_wz_Perceptual':
        criterion = L2_wz_Perceptual(args)
        print('Training with L2 and Perceptual Loss!')
    return criterion


