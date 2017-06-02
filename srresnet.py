from __future__ import division

from model import SRResNet, Residual, SubPixelConv
import dataprovider
import argparse
import os
import time

from PIL import Image
import random
from scipy.misc import imread
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import torchvision.models as models
from torch.autograd import Variable
from torch.utils.data import DataLoader


HEIGHT = 224
WIDTH = 224
SCALE = 4

# Training settings
parser = argparse.ArgumentParser(description='PyTorch VDSR')
parser.add_argument('--batchSize',
                    type=int,
                    default=64,
                    help='Training batch size')
parser.add_argument('--nEpochs',
                    type=int,
                    default=150,
                    help='Number of epochs to train for')
parser.add_argument('--lr',
                    type=float,
                    default=0.01,
                    help='Learning Rate. Default=0.1')
parser.add_argument('--step',
                    type=int,
                    default=10,
                    help='learning rate decayed every n epochs, Default: n=10')
parser.add_argument('--cuda',
                    action='store_true',
                    help='Use cuda?')
parser.add_argument('--resume',
                    default='',
                    type=str,
                    help='Path to checkpoint (default: none)')
parser.add_argument('--start-epoch',
                    default=1,
                    type=int,
                    help='Manual epoch number (useful on restarts)')
parser.add_argument('--clip',
                    type=float,
                    default=0.4,
                    help='Clipping Gradients. Default=0.4')
parser.add_argument('--threads',
                    type=int,
                    default=0,
                    help='Number of threads for data loader, Default: 4')
parser.add_argument('--images',
                    type=int,
                    default=400,
                    help='Number of threads for data loader, Default: 400')
parser.add_argument('--test-image',
                    default='',
                    type=str,
                    help='Path to image that should be scaled up')
parser.add_argument('--momentum',
                    default=0.9,
                    type=float,
                    help='Momentum, Default: 0.9')
parser.add_argument('--weight-decay',
                    '--wd',
                    default=1e-4,
                    type=float,
                    help='Weight decay, Default: 1e-4')
parser.add_argument('--percep-scale',
                    default=0.006,
                    type=float,
                    help='weight to content vs pixel')
parser.add_argument('--pretrained',
                    default='',
                    type=str,
                    help='path to pretrained model (default: none)')
parser.add_argument('--image-dir',
                    default='',
                    type=str,
                    help='directory with images to train on (default: none)')
parser.add_argument('--pretraining',
                    action='store_true',
                    help='pretraining step?')
parser.add_argument('--testing',
                    action='store_true',
                    help='inference step?')


def main():

    global opt, model, HEIGHT, WIDTH, SCALE
    opt = parser.parse_args()
    print(opt)
    test_image = None
    if opt.testing:
        opt.batchSize = 1
        img = imread(opt.test_image)
        HEIGHT, WIDTH = img.shape[0], img.shape[1]
        test_image = Image.fromarray(np.uint8(img))
        test_image = np.asarray(test_image)

        if test_image.ndim == 3:
            if test_image.shape[2] != 3:
                test_image = test_image[:, :, 0:3]

            test_image = torch.ByteTensor(
                torch.ByteStorage.from_buffer(test_image.transpose(2, 0, 1).tobytes())).float().div(255).view(-1, 3, HEIGHT, WIDTH)
        else:
            print('not good... we do not upscale non color images yet')
            return

    cuda = opt.cuda
    if cuda and not torch.cuda.is_available():
        raise Exception('No GPU found, please run without --cuda')

    opt.seed = random.randint(1, 10000)
    print('Random Seed: ', opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    model = SRResNet()

    #  clean this mess up!
    if opt.testing:
        model.eval()
        mean = torch.zeros(opt.batchSize, 3, HEIGHT * SCALE, WIDTH * SCALE)
        mean[:, 0, :, :] = 0.485
        mean[:, 1, :, :] = 0.456
        mean[:, 2, :, :] = 0.406

        std = torch.zeros(opt.batchSize, 3, HEIGHT * SCALE, WIDTH * SCALE)
        std[:, 0, :, :] = 0.229
        std[:, 1, :, :] = 0.224
        std[:, 2, :, :] = 0.225

        tmean = torch.zeros(opt.batchSize, 3, HEIGHT, WIDTH)
        tmean[:, 0, :, :] = 0.485
        tmean[:, 1, :, :] = 0.456
        tmean[:, 2, :, :] = 0.406

        tstd = torch.zeros(opt.batchSize, 3, HEIGHT, WIDTH)
        tstd[:, 0, :, :] = 0.229
        tstd[:, 1, :, :] = 0.224
        tstd[:, 2, :, :] = 0.225

    else:
        model.train()
        mean = torch.zeros(opt.batchSize, 3, HEIGHT, WIDTH)
        mean[:, 0, :, :] = 0.485
        mean[:, 1, :, :] = 0.456
        mean[:, 2, :, :] = 0.406

        std = torch.zeros(opt.batchSize, 3, HEIGHT, WIDTH)
        std[:, 0, :, :] = 0.229
        std[:, 1, :, :] = 0.224
        std[:, 2, :, :] = 0.225

        tmean = torch.zeros(opt.batchSize, 3, HEIGHT // SCALE, WIDTH // SCALE)
        tmean[:, 0, :, :] = 0.485
        tmean[:, 1, :, :] = 0.456
        tmean[:, 2, :, :] = 0.406

        tstd = torch.zeros(opt.batchSize, 3, HEIGHT // SCALE, WIDTH // SCALE)
        tstd[:, 0, :, :] = 0.229
        tstd[:, 1, :, :] = 0.224
        tstd[:, 2, :, :] = 0.225

    if not opt.pretraining and not opt.testing:
        percep_model = models.__dict__['vgg19'](pretrained=True)
        percep_model.features = nn.Sequential(
            *list(percep_model.features.children())[:-14])
        percep_model.eval()

    criterion = nn.MSELoss(size_average=False)
    lr = opt.lr

    if cuda:
        model = torch.nn.DataParallel(model).cuda()
        criterion = criterion.cuda()
        if not opt.pretraining and not opt.testing:
            percep_model = percep_model.cuda()
        mean = Variable(mean).cuda()
        std = Variable(std).cuda()
        tmean = Variable(tmean).cuda()
        tstd = Variable(tstd).cuda()

    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            print('=> loading model {}'.format(opt.pretrained))
            weights = torch.load(opt.pretrained)
            model.load_state_dict(weights['model'].state_dict())
        else:
            print('=> no model found at {}'.format(opt.pretrained))

    if opt.testing:
        test_image = Variable(test_image)
        if cuda:
            test_image = test_image.cuda()

        test_image = test_image.sub(tmean).div(tstd)
        gen = model(test_image)
        gened = torch.clamp(gen.mul(std).add(mean).mul(255.0), min=0., max=255.0).byte()[
            0].data.cpu().numpy().transpose(1, 2, 0)
        gened = Image.fromarray(gened)
        gened.save('testing-sr.jpg')

    else:
        train_set = dataprovider.DatasetFromDir(
            opt.image_dir,
            samples=opt.images,
            width=WIDTH,
            height=HEIGHT)

        training_data_loader = DataLoader(
            dataset=train_set,
            num_workers=opt.threads,
            batch_size=opt.batchSize,
            shuffle=True)

        optimizer = optim.Adam(model.parameters(), lr=lr)

        counter = 0
        for epoch in range(opt.nEpochs):

            loss_sum = Variable(torch.zeros(1), requires_grad=False)
            if cuda:
                loss_sum = loss_sum.cuda()

            for iteration, batch in enumerate(training_data_loader, 1):
                counter = counter + 1
                input, target = (
                    Variable(batch[0]),
                    Variable(batch[1], requires_grad=False))

                if cuda:
                    input = input.cuda()
                    target = target.cuda()

                input = input.sub(tmean).div(tstd)
                target = target.sub(mean).div(std)

                gen = model(input)
                optimizer.zero_grad()
                loss = criterion(gen, target)

                if not opt.pretraining:
                    out_percep = percep_model.features(gen)
                    out_percep_real = Variable(percep_model.features(
                        target).data, requires_grad=False)
                    percep_loss = criterion(out_percep, out_percep_real)
#                    loss_relation = percep_loss.div(loss)

                    loss = loss.add(percep_loss.mul(opt.percep_scale))  # loss_relation))

                loss.backward()
                nn.utils.clip_grad_norm(model.parameters(), opt.clip)
                loss_sum.add_(loss)
                optimizer.step()

                if counter % 400 == 0:
                    print('sum_of_loss = {}'.format(
                        loss_sum.data.select(0, 0)))
                    loss_sum = Variable(torch.zeros(1), requires_grad=False)
                    if cuda:
                        loss_sum = loss_sum.cuda()

                    save_checkpoint(model, epoch)
                    input = torch.clamp(input.mul(tstd).add(tmean).mul(
                        255.0), min=0., max=255.0).byte()[0].data.cpu().numpy().transpose(1, 2, 0)
                    inp = Image.fromarray(input)
                    label = torch.clamp(target.mul(std).add(mean).mul(255.0), min=0., max=255.0).byte()[
                        0].data.cpu().numpy().transpose(1, 2, 0)
                    lab = Image.fromarray(label)
                    gened = torch.clamp(gen.mul(std).add(mean).mul(255.0), min=0., max=255.0).byte()[
                        0].data.cpu().numpy().transpose(1, 2, 0)
                    gened = Image.fromarray(gened)
                    inp.save('input.jpg')
                    lab.save('gt.jpg')
                    gened.save('sr.jpg')


def save_checkpoint(model, epoch):
    model_out_path = 'model/' + 'model_epoch_{}.pth'.format(epoch)
    state = {'epoch': epoch, 'model': model}
    if not os.path.exists('model/'):
        os.makedirs('model/')

    torch.save(state, model_out_path)

    print('Checkpoint saved to {}'.format(model_out_path))


if __name__ == '__main__':
    main()
