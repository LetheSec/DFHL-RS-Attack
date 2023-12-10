import logging
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torchattacks


def save_batch_fake(images, labels, save_dir, epoch):
    images = images.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    images_filename = os.path.join(save_dir, "fake_images.npy")
    labels_filename = os.path.join(save_dir, "fake_labels.npy")

    if epoch > 1:
        org_images = np.load(images_filename)
        org_labels = np.load(labels_filename)

        images = np.concatenate((org_images, images), 0)
        labels = np.concatenate((org_labels, labels), 0)

    np.save(images_filename, images)
    np.save(labels_filename, labels)


@torch.no_grad()
def get_rank2_label(logit, y):
    batch_size = len(logit)
    tmp = logit.clone()
    tmp[torch.arange(batch_size), y] = -float("inf")
    return tmp.argmax(1)


def clean_test(model, test_loader):
    correct = 0
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_accuracy = correct / len(test_loader.dataset)
    return test_accuracy


def adv_test(model, test_loader):
    correct = 0
    model.eval()
    attack = torchattacks.PGD(
        model, eps=8 / 255, alpha=2 / 255, steps=10, random_start=True
    )
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            with torch.enable_grad():
                adv_data = attack(data, target)
            output = model(adv_data)

            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_accuracy = correct / len(test_loader.dataset)
    return test_accuracy


def adv_test_l2(model, test_loader):
    correct = 0
    model.eval()
    # attack = torchattacks.PGD(model, eps=8 / 255, alpha=2 / 255, steps=10, random_start=True)

    attack = torchattacks.PGDL2(
        model, eps=128.0 / 255, alpha=15.0 / 255, steps=10, random_start=True
    )

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            with torch.enable_grad():
                adv_data = attack(data, target)
            output = model(adv_data)

            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_accuracy = correct / len(test_loader.dataset)
    return test_accuracy


def reset_model(model):
    for m in model.modules():
        if isinstance(m, (nn.ConvTranspose2d, nn.Linear, nn.Conv2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, (nn.BatchNorm2d)):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)


# 固定随即种子
def random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Logger:
    def __init__(self, logfile="output.log"):
        self.logfile = logfile
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            format="[%(asctime)s] - %(message)s",
            datefmt="%Y/%m/%d %H:%M:%S",
            level=logging.INFO,
            filename=self.logfile,
        )

    def info(self, msg, *args):
        msg = str(msg)
        if args:
            print(msg % args)
            self.logger.info(msg, *args)
        else:
            print(msg)
            self.logger.info(msg)


def save_checkpoint(
        state,
        epoch,
        is_best,
        which_best,
        save_path,
        save_freq=10,
):
    filename = os.path.join(save_path, "checkpoint_" + str(epoch) + ".tar")
    if epoch % save_freq == 0:
        if not os.path.exists(filename):
            torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(
            save_path, "best_" + str(which_best) + "_checkpoint.tar"
        )
        torch.save(state, best_filename)
