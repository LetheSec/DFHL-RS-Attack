import torch
import torch.nn as nn
import torch.nn.functional as F

import losses as L


def generate_hee(args, model, x):
    device = x.device

    model.eval()
    x_hee = x.detach() + 0.001 * torch.torch.randn(x.shape).to(device).detach()
    for _ in range(args.steps_hee):
        x_hee.requires_grad_()
        with torch.enable_grad():
            loss = L.Entropy_Loss(reduction="mean")(model(x_hee))
        grad = torch.autograd.grad(loss, [x_hee])[0]
        x_hee = x_hee.detach() + args.lr_hee * torch.sign(grad.detach())
        x_hee = torch.clamp(x_hee, 0.0, 1.0)
    model.train()

    return x_hee


def generate_ue(args, model, x, num_classes):
    device = x.device
    pes_label = torch.full((x.size(0), num_classes), 1 / num_classes).to(device)

    model.eval()
    # generate uncertain example
    x_ue = x.detach() + 0.001 * torch.torch.randn(x.shape).to(device).detach()
    for _ in range(args.num_steps):
        x_ue.requires_grad_()
        with torch.enable_grad():
            loss = -F.kl_div(
                F.log_softmax(model(x_ue), dim=1),
                F.softmax(pes_label, dim=1),
                size_average=False,
            )
        grad = torch.autograd.grad(loss, [x_ue])[0]
        x_ue = x_ue.detach() + args.step_size * torch.sign(grad.detach())
        x_ue = torch.min(torch.max(x_ue, x - args.epsilon), x + args.epsilon)
        x_ue = torch.clamp(x_ue, 0.0, 1.0)

    model.train()

    return x_ue


def generate_adv(args, model, x, target):
    device = x.device

    model.eval()
    # random_start
    x_adv = x.detach() + 0.001 * torch.torch.randn(x.shape).to(device).detach()

    for _ in range(args.num_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss_ce = nn.CrossEntropyLoss()(model(x_adv), target)
        grad = torch.autograd.grad(loss_ce, [x_adv])[0]
        x_adv = x_adv.detach() + args.step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x - args.epsilon), x + args.epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    model.train()

    return x_adv
