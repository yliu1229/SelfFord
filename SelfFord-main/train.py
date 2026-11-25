import argparse
import os
import sys
import time

import torch
import torch.backends.cudnn as cudnn
import yaml

from criterion import my_criterion
from data import ImageDataset
from torch.utils.data import DataLoader
from noiseExtractor import NoiseExtractor
from utils import AverageMeter, save_checkpoint

# cudnn setting
cudnn.benchmark = True
cudnn.enabled = True

def set_path(config):
    if config['train']['checkpoint']:
        model_path = os.path.dirname(config['train']['checkpoint'])
    else:
        model_path = './log_tmp/{0}-bs{1}/model'.format(config["data"]["dataset_name"],
                                                            config["train"]["batch_size"])

    if not os.path.exists(model_path): os.makedirs(model_path)
    return model_path


def exclude_from_wt_decay(named_params, weight_decay: float, lr: float):
    params = []
    excluded_params = []

    for name, param in named_params:
        if not param.requires_grad:
            continue
        # do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            excluded_params.append(param)
        else:
            params.append(param)
    return [{'params': params, 'weight_decay': weight_decay, 'lr': lr},
            {'params': excluded_params, 'weight_decay': 0., 'lr': lr}]


def configure_optimizers(model, train_config):
    cnn_params_named = []
    head_params_named = []
    for name, param in model.named_parameters():
        if name.startswith("dnCNN"):
            cnn_params_named.append((name, param))
        else:
            head_params_named.append((name, param))

    # Prepare param groups. Exclude norm and bias from weight decay if flag set.
    if train_config['exclude_norm_bias']:
        cnn_params = [param for _, param in cnn_params_named]
        params = exclude_from_wt_decay(head_params_named,
                                       weight_decay=train_config["weight_decay"],
                                       lr=train_config['lr_head'])
        params.append({'params': cnn_params, 'lr': train_config['lr_cnn']})
    else:
        cnn_params = [param for _, param in cnn_params_named]
        head_params = [param for _, param in head_params_named]
        params = [{'params': cnn_params, 'lr': train_config['lr_cnn']},
                  {'params': head_params, 'lr': train_config['lr_head']}]

    # Init optimizer and lr schedule
    optimizer = torch.optim.AdamW(params, weight_decay=train_config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    return optimizer, scheduler


def start_train():
    with open(args.config_path) as file:
        config = yaml.safe_load(file.read())
    # print('Config: ', config)

    data_config = config['data']
    train_config = config['train']
    torch.manual_seed(train_config['seed'])
    torch.cuda.manual_seed_all(train_config['seed'])
    dataset_name = data_config["dataset_name"]

    # Setup train data
    if dataset_name == "world":
        train_data_module = ImageDataset(root=data_config["root"],
                                         crop_size=data_config["crop_size"],
                                         patch_num=data_config["patch_num"],
                                         QFs=data_config["QFs"])
        dataloader = DataLoader(train_data_module, batch_size=train_config["batch_size"],
                                shuffle=True, num_workers=config["num_workers"],
                                drop_last=True, pin_memory=True)
    else:
        raise ValueError(f"Data set {dataset_name} not supported")

    model_path = set_path(config)

    model = NoiseExtractor(3, kernels=[3, ] * train_config["num_levels"],
                          features=[64, ] * (train_config["num_levels"] - 1) + [train_config["out_channel"]],
                          bns=[False, ] + [True, ] * (train_config["num_levels"] - 2) + [False, ],
                          acts=['relu', ] * (train_config["num_levels"] - 1) + ['linear', ],
                          dilats=[1, ] * train_config["num_levels"],
                          bn_momentum=0.1, padding=1, dncnn_path=train_config["dnCNN"])
    model = model.to(cuda)

    criterion = my_criterion(patch_num=data_config["patch_num"], temperature=train_config["ce_temperature"])
    criterion = criterion.to(cuda)

    # Initialize model
    start_epoch = 0
    if train_config["checkpoint"] is not None:
        checkpoint = torch.load(train_config["checkpoint"])
        start_epoch = checkpoint['epoch']
        msg = model.load_state_dict(checkpoint["state_dict"], strict=True)
        print(msg)

    optimizer, scheduler = configure_optimizers(model, train_config)

    '''
    print('\n===========Check Grad============')
    for name, param in model.named_parameters():
        print(name, param.requires_grad)
    print('=================================\n')
    '''

    for epoch in range(start_epoch, train_config['max_epochs']):

        train(dataloader, model, optimizer, criterion, epoch)

        scheduler.step()
        print('\t Epoch: ', epoch, 'with lr: ', scheduler.get_last_lr())

        if epoch % train_config['save_checkpoint_every_n_epochs'] == 0:
            # save check_point
            save_checkpoint({'epoch': epoch + 1,
                             'state_dict': model.state_dict(),
                             }, gap=train_config['save_checkpoint_every_n_epochs'],
                            filename=os.path.join(model_path, 'epoch%s.pth' % str(epoch + 1)), keep_all=False)

    print('Training %d epochs finished' % (train_config['max_epochs']))


def train(data_loader, model, optimizer, criterion, epoch):
    losses = AverageMeter()
    model.train()

    for idx, inputs in enumerate(data_loader):
        # inputs = [(B, 3, w, h), ...]
        B = inputs[0].size(0)

        tic = time.time()
        for i in range(len(inputs)):
            inputs[i] = inputs[i].to(cuda, non_blocking=True)

        results = model(inputs)

        # Calculate loss
        loss = criterion(results)
        losses.update(loss.item(), B, step=len(data_loader))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if idx % 1 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.local_avg:.4f}) Time:{3:.2f}\t'.
                  format(epoch, idx, len(data_loader), time.time() - tic, loss=losses))

    return losses.local_avg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='./train_config.yml', type=str)
    parser.add_argument('--gpu', default='0', type=str)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    cuda = torch.device('cuda')
    start_train()
