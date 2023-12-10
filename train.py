import argparse
import datetime
import time
import warnings

import losses as L
import models
import torch.nn.functional as F
from dataset import *
from kornia import augmentation
from query_sample import generate_adv, generate_hee, generate_ue
from robust_test import robust_eval
from torchvision import datasets, transforms
from utils import *

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="Data-Free Hard-Label Robustness Stealing")

# model configuration
parser.add_argument(
    "--arch",
    type=str,
    choices=["ResNet18", "ResNet34", "WideResNet", "MobileNet"],
    default="ResNet18",
)
parser.add_argument(
    "--target_arch", type=str, default="ResNet18", choices=["ResNet18", "WideResNet"]
)
parser.add_argument(
    "--target_defense",
    type=str,
    default="AT",
    choices=["AT", "TRADES", "STAT_AWP"],
)
parser.add_argument("--target_dir", type=str, default="./checkpoints/")
# generator configuration
parser.add_argument(
    "--gen_dim_z",
    "-gdz",
    type=int,
    default=256,
    help="Dimension of generator input noise.",
)
parser.add_argument(
    "--gen_distribution",
    "-gd",
    type=str,
    default="normal",
    help="Input noise distribution: normal (default) or uniform.",
)

# dataset configuration
parser.add_argument(
    "--data", type=str, default="CIFAR10", choices=["CIFAR10", "CIFAR100"]
)
parser.add_argument(
    "--data_path", type=str, default="~/datasets/", help="where is the dataset CIFAR-10"
)
parser.add_argument(
    "--test_batch_size",
    type=int,
    default=512,
    metavar="N",
    help="input batch size for testing",
)

# training configuration
parser.add_argument(
    "--batch_size",
    type=int,
    default=256,
    metavar="N",
    help="input batch size for training",
)
parser.add_argument(
    "--epochs", type=int, default=300, metavar="N", help="number of epochs to train"
)
parser.add_argument(
    "--lr", type=float, default=0.1, metavar="N", help="learning rate of clone model"
)
parser.add_argument(
    "--momentum", default=0.9, type=float, metavar="M", help="momentum of SGD solver"
)
parser.add_argument(
    "--weight_decay",
    default=1e-4,
    type=float,
)
parser.add_argument(
    "--N_C", type=int, default=500, metavar="N", help="iterations of clone model"
)
parser.add_argument(
    "--N_G", type=int, default=10, metavar="N", help="iterations of generator"
)
parser.add_argument(
    "--lr_G", type=float, default=0.002, metavar="N", help="learning rate of generator"
)
parser.add_argument(
    "--lr_z", type=float, default=0.01, help="learning rate of latent code"
)
parser.add_argument(
    "--lam", type=float, default=3, help="hyperparameter for balancing two loss terms"
)
parser.add_argument(
    "--label_smooth_factor",
    default=0.2,
    type=float,
    help="0.2 for CIFAR 10, 0.02 for CIFAR100",
)

# HEE configuration
parser.add_argument(
    "--lr_hee", type=float, default=0.03, metavar="N", help="number of epochs to train"
)
parser.add_argument("--steps_hee", default=10, type=int, help="perturb number of steps")
parser.add_argument(
    "--query_mode",
    default="HEE",
    type=str,
    choices=[
        "UE",
        "AE",
        "HEE",
        "AT",
    ],
)
# for AE/UE
parser.add_argument("--epsilon", default=8.0 / 255, type=eval)
parser.add_argument("--num_steps", default=10, type=int)
parser.add_argument("--step_size", default=2.0 / 255, type=eval)
# other configuration
parser.add_argument(
    "--result_dir", default="results", help="directory of model for saving checkpoint"
)
parser.add_argument(
    "--save_freq", "-s", default=50, type=int, metavar="N", help="save frequency"
)
parser.add_argument(
    "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
)

args = parser.parse_args()

if args.data == "CIFAR100":
    NUM_CLASSES = 100
else:
    NUM_CLASSES = 10

target_path = os.path.join(
    args.target_dir,
    args.data,
    args.target_defense,
    args.target_arch,
    "best_robust_checkpoint.tar",
)
exp_time = datetime.datetime.now().strftime("%y%m%d_%H%M")
checkpoint_path = os.path.join(
    args.result_dir,
    args.data,
    args.target_defense + "_" + args.target_arch + "-to-" + args.arch,
    args.query_mode,
    exp_time,
    "checkpoints",
)

save_dir = os.path.join(
    args.result_dir,
    args.data,
    args.target_defense + "_" + args.target_arch + "-to-" + args.arch,
    args.query_mode,
    exp_time,
    "runs_imgs",
)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)

logger = Logger(
    os.path.join(
        args.result_dir,
        args.data,
        args.target_defense + "_" + args.target_arch + "-to-" + args.arch,
        args.query_mode,
        exp_time,
        "output.log",
    )
)

if args.data == "CIFAR10" or args.data == "CIFAR100":
    img_size = 32
    img_shape = (3, 32, 32)
    nc = 3

if args.seed is not None:
    random_seed(args.seed)

# Standard Augmentation
std_aug = augmentation.container.ImageSequential(
    augmentation.RandomCrop(size=[img_shape[-2], img_shape[-1]], padding=4),
    augmentation.RandomHorizontalFlip(),
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

best_nature_acc = 0
best_robust_acc = 0
curr_query_times = 0


def data_generation(args, generator, clone_model, target_model, epoch):
    generator.train()
    clone_model.eval()
    target_model.eval()

    best_fake = None
    best_loss = 1e6

    z = torch.randn(size=(args.batch_size, args.gen_dim_z)).to(device)
    z.requires_grad = True

    optimizer_G = torch.optim.Adam(
        [{"params": generator.parameters()}, {"params": [z], "lr": args.lr_z}],
        lr=args.lr_G,
        betas=[0.5, 0.999],
    )

    # get pseudo soft labels
    pseudo_y = torch.randint(low=0, high=NUM_CLASSES, size=(args.batch_size,)).to(
        device
    )
    soft_labels = L.smooth_one_hot(
        pseudo_y, classes=NUM_CLASSES, smoothing=args.label_smooth_factor
    )

    for step in range(args.N_G):
        # generator a batch of fake images
        fake = generator(z)
        aug_fake = std_aug(fake)

        # forward pass by clone model
        logits = clone_model(aug_fake)

        loss_cls = L.cross_entropy(logits, soft_labels)
        loss_div = L.div_loss(logits)

        loss = loss_cls + loss_div * args.lam

        with torch.no_grad():
            if best_loss > loss.item() or best_fake is None:
                best_loss = loss.item()
                best_fake = fake

        optimizer_G.zero_grad()
        loss.backward()
        optimizer_G.step()

    # our DFHL-RS need no query budget in this stage, only Data-Free AE needs this pseudo labels.
    pseudo_labels = target_model(best_fake).topk(1, 1)[1].reshape(-1)
    # save synthetic samples
    save_batch_fake(best_fake.data, pseudo_labels, save_dir, epoch)


def train_clone_model(args, clone_model, target_model, optimizer, epoch):
    global curr_query_times

    target_model.eval()
    clone_model.train()

    tmp_time = time.time()

    # get synthetic samples from memory bank
    dataset = FakeDataset(root=save_dir)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    data_iter = DataIter(data_loader)

    # train the clone model
    for step in range(args.N_C):
        fake, labels = data_iter.next()
        fake, labels = fake.to(device), labels.to(device)

        # Standard augmentation
        aug_fake = std_aug(fake)

        if args.query_mode == "HEE":
            # Strong augmentation to imporvove the diversity
            fake_hee = generate_hee(args, clone_model, strong_aug(aug_fake))
            # query the target model, get hard labels
            logits_T = target_model(fake_hee).detach()
            hard_labels = logits_T.topk(1, 1)[1].reshape(-1)

            logits = clone_model(fake_hee)
            loss = F.cross_entropy(logits, hard_labels)
            curr_query_times += fake_hee.size(0)

        elif args.query_mode == "UE":
            # Strong augmentation to imporvove the diversity of uct
            fake_ue = generate_ue(args, clone_model, strong_aug(aug_fake), NUM_CLASSES)
            # query the target model, get hard labels
            logits_T = target_model(fake_ue).detach()
            hard_labels = logits_T.topk(1, 1)[1].reshape(-1)

            logits = clone_model(fake_ue)
            loss = F.cross_entropy(logits, hard_labels)
            curr_query_times += fake_ue.size(0)

        elif args.query_mode == "AE":
            # construct AE with synthetic samples
            fake_adv = generate_adv(args, clone_model, aug_fake, labels)  # query
            # query the target model for hard labels
            logits_T = target_model(fake_adv).detach()
            hard_labels = logits_T.topk(1, 1)[1].reshape(-1)

            logits = clone_model(fake_adv)
            loss = F.cross_entropy(logits, hard_labels)
            curr_query_times += fake_adv.size(0)

        elif args.query_mode == "AT":
            # query the target model, get hard labels
            logits_T = target_model(aug_fake).detach()
            hard_labels = logits_T.topk(1, 1)[1].reshape(-1)
            fake_adv = generate_adv(args, clone_model, aug_fake, hard_labels)

            # perform AT
            logits = clone_model(fake_adv)
            loss = F.cross_entropy(logits, hard_labels)
            curr_query_times += fake_adv.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def main():
    global best_nature_acc, best_robust_acc
    logger.info(args)

    testset = getattr(datasets, args.data)(
        root=args.data_path, train=False, download=True, transform=transforms.ToTensor()
    )
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.test_batch_size, shuffle=False
    )

    # get clone model
    clone_model = getattr(models, args.arch)(num_classes=NUM_CLASSES)
    clone_model = nn.DataParallel(clone_model).to(device)

    # get target model
    target_model = getattr(models, args.target_arch)(num_classes=NUM_CLASSES)
    target_model = nn.DataParallel(target_model).to(device)
    state_dict = torch.load(target_path, map_location=device)
    target_model.load_state_dict(state_dict["model_state_dict"])
    target_model.eval()

    generator = models.Generator(nz=args.gen_dim_z, ngf=64, img_size=img_size, nc=nc)
    generator = nn.DataParallel(generator).to(device)

    optimizer = torch.optim.SGD(
        clone_model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, eta_min=2e-4
    )

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        data_generation(args, generator, clone_model, target_model, epoch)
        train_clone_model(args, clone_model, target_model, optimizer, epoch)

        scheduler.step()

        nature_acc = clean_test(clone_model, test_loader)
        robust_acc = adv_test(clone_model, test_loader)

        epoch_time = time.time() - start_time

        logger.info(
            "Epoch %d Finish, Time Cost %d, Nature Acc %.4f, Robust Acc %.4f"
            % (epoch, epoch_time, nature_acc, robust_acc)
        )

        is_best_robust = robust_acc > best_robust_acc
        best_robust_acc = max(robust_acc, best_robust_acc)
        save_checkpoint(
            {
                "epoch": epoch,
                "model_state_dict": clone_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "nature_acc": float(nature_acc),
                "robust_acc": float(robust_acc),
            },
            epoch,
            is_best_robust,
            "robust",
            save_path=checkpoint_path,
            save_freq=args.save_freq,
        )

        # Save checkpoint
        is_best = nature_acc > best_nature_acc
        best_nature_acc = max(nature_acc, best_nature_acc)
        save_checkpoint(
            {
                "epoch": epoch,
                "model_state_dict": clone_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "nature_acc": float(nature_acc),
                "robust_acc": float(robust_acc),
            },
            epoch,
            is_best,
            "nature",
            save_path=checkpoint_path,
            save_freq=args.save_freq,
        )

    logger.info("Best Nature ACC %.4f", best_nature_acc)
    logger.info("Best Robust ACC %.4f", best_robust_acc)

    logger.info("Eval Results")
    best_robust_model = getattr(models, args.arch)(num_classes=NUM_CLASSES)
    best_robust_model = torch.nn.DataParallel(best_robust_model)
    best_robust_model.load_state_dict(
        torch.load(os.path.join(checkpoint_path, "best_robust_checkpoint.tar"))[
            "model_state_dict"
        ]
    )
    best_robust_model = best_robust_model.to(device)
    best_robust_model.eval()
    eval_results = robust_eval(best_robust_model, test_loader, device)
    logger.info(eval_results)


if __name__ == "__main__":
    main()
