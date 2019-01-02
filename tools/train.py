#!/usr/bin/env python
# coding: utf-8
#
# Author:   ranjiewen
# URL:      
# Created:  2019-01-01

from __future__ import absolute_import, division, print_function

import os.path as osp

import click
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from addict import Dict
from tensorboardX import SummaryWriter
from torchnet.meter import MovingAverageValueMeter
from tqdm import tqdm

from libs.datasets import get_dataset
from libs.models import DeepLabV2_ResNet101_MSC
from libs.loss.ce_loss import CrossEntropyLoss2d
from libs.solver.lr_scheduler import get_params,poly_lr_scheduler


def main(args):
    cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    if cuda:
        current_device = torch.cuda.current_device()
        print("Running on", torch.cuda.get_device_name(current_device))
    else:
        print("Running on CPU")

    # Configuration
    CONFIG = Dict(yaml.load(open(args.config)))

    # Dataset 10k or 164k
    dataset = get_dataset(CONFIG.DATASET)(
        root=CONFIG.ROOT,
        split=CONFIG.SPLIT.TRAIN,
        base_size=513,
        crop_size=CONFIG.IMAGE.SIZE.TRAIN,
        mean=(CONFIG.IMAGE.MEAN.B, CONFIG.IMAGE.MEAN.G, CONFIG.IMAGE.MEAN.R),
        warp=CONFIG.WARP_IMAGE,
        scale=(0.5, 0.75, 1.0, 1.25, 1.5),
        flip=True,
    )

    # DataLoader
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=CONFIG.BATCH_SIZE.TRAIN,
        num_workers=CONFIG.NUM_WORKERS,
        shuffle=True,
    )
    loader_iter = iter(loader)

    # Model
    model = DeepLabV2_ResNet101_MSC(n_classes=CONFIG.N_CLASSES)
    state_dict = torch.load(CONFIG.INIT_MODEL)
    model.load_state_dict(state_dict, strict=False)  # Skip "aspp" layer
    model = nn.DataParallel(model)
    model.to(device)

    # Optimizer
    optimizer = torch.optim.SGD(
        # cf lr_mult and decay_mult in train.prototxt
        params=[
            {
                "params": get_params(model.module, key="1x"),
                "lr": CONFIG.LR,
                "weight_decay": CONFIG.WEIGHT_DECAY,
            },
            {
                "params": get_params(model.module, key="10x"),
                "lr": 10 * CONFIG.LR,
                "weight_decay": CONFIG.WEIGHT_DECAY,
            },
            {
                "params": get_params(model.module, key="20x"),
                "lr": 20 * CONFIG.LR,
                "weight_decay": 0.0,
            },
        ],
        momentum=CONFIG.MOMENTUM,
    )

    # Loss definition
    criterion = CrossEntropyLoss2d(ignore_index=CONFIG.IGNORE_LABEL)
    criterion.to(device)

    # TensorBoard Logger
    writer = SummaryWriter(CONFIG.LOG_DIR)
    loss_meter = MovingAverageValueMeter(20)

    model.train()
    model.module.scale.freeze_bn()

    for iteration in tqdm(
        range(1, CONFIG.ITER_MAX + 1),
        total=CONFIG.ITER_MAX,
        leave=False,
        dynamic_ncols=True,
    ):

        # Set a learning rate
        poly_lr_scheduler(
            optimizer=optimizer,
            init_lr=CONFIG.LR,
            iter=iteration - 1,
            lr_decay_iter=CONFIG.LR_DECAY,
            max_iter=CONFIG.ITER_MAX,
            power=CONFIG.POLY_POWER,
        )

        # Clear gradients (ready to accumulate)
        optimizer.zero_grad()

        iter_loss = 0
        for i in range(1, CONFIG.ITER_SIZE + 1):
            try:
                images, labels = next(loader_iter)
            except:
                loader_iter = iter(loader)
                images, labels = next(loader_iter)

            images = images.to(device)
            labels = labels.to(device).unsqueeze(1).float()

            # Propagate forward
            logits = model(images)

            # Loss
            loss = 0
            for logit in logits:
                # Resize labels for {100%, 75%, 50%, Max} logits
                labels_ = F.interpolate(labels, logit.shape[2:], mode="nearest")
                labels_ = labels_.squeeze(1).long()
                # Compute crossentropy loss
                loss += criterion(logit, labels_)

            # Backpropagate (just compute gradients wrt the loss)
            loss /= float(CONFIG.ITER_SIZE)
            loss.backward()

            iter_loss += float(loss)

        loss_meter.add(iter_loss)

        # Update weights with accumulated gradients
        optimizer.step()

        # TensorBoard
        if iteration % CONFIG.ITER_TB == 0:
            writer.add_scalar("train_loss", loss_meter.value()[0], iteration)
            for i, o in enumerate(optimizer.param_groups):
                writer.add_scalar("train_lr_group{}".format(i), o["lr"], iteration)
            if False:  # This produces a large log file
                for name, param in model.named_parameters():
                    name = name.replace(".", "/")
                    writer.add_histogram(name, param, iteration, bins="auto")
                    if param.requires_grad:
                        writer.add_histogram(
                            name + "/grad", param.grad, iteration, bins="auto"
                        )

        # Save a model
        if iteration % CONFIG.ITER_SAVE == 0:
            torch.save(
                model.module.state_dict(),
                osp.join(CONFIG.SAVE_DIR, "checkpoint_{}.pth".format(iteration)),
            )

        # Save a model (short term)
        if iteration % 100 == 0:
            torch.save(
                model.module.state_dict(),
                osp.join(CONFIG.SAVE_DIR, "checkpoint_current.pth"),
            )

    torch.save(
        model.module.state_dict(), osp.join(CONFIG.SAVE_DIR, "checkpoint_final.pth")
    )


import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Pytorch implement deeplab train for image semantic segmetation.")
    parser.add_argument("--config",default="",metavar="FILE",help="path to config file",type=str)
    parser.add_argument("--use_cuda",dest="use_cuda",help="use cuda for accelerate",action="use_true",default=True)
    args=parser.parse_args()

    main(args)
