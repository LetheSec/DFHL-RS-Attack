import torch
import torch.nn as nn


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return torch.flatten(x, 1)


class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, img_size=32, nc=3):
        super(Generator, self).__init__()
        self.params = (nz, ngf, img_size, nc)
        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(nz, ngf * 2 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(ngf * 2),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(ngf * 2, ngf * 2, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(ngf * 2, ngf, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

    # return a copy of its own
    def clone(self):
        clone = Generator(self.params[0], self.params[1], self.params[2], self.params[3])
        clone.load_state_dict(self.state_dict())
        return clone.cuda()

#
# class Generator(nn.Module):
#     def __init__(self, nz=100, ngf=64, img_size=32, nc=3):
#         super(Generator, self).__init__()
#
#         self.init_size = img_size // 4
#         self.l1 = nn.Sequential(nn.Linear(nz, ngf * 2 * self.init_size ** 2))
#
#         self.conv_blocks = nn.Sequential(
#             nn.BatchNorm2d(ngf * 2),
#             nn.Upsample(scale_factor=2),
#
#             nn.Conv2d(ngf * 2, ngf * 2, 3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(ngf * 2),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Upsample(scale_factor=2),
#
#             nn.Conv2d(ngf * 2, ngf, 3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(ngf),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
#             nn.Sigmoid(),
#         )
#
#     def forward(self, z):
#         out = self.l1(z)
#         out = out.view(out.shape[0], -1, self.init_size, self.init_size)
#         img = self.conv_blocks(out)
#         return img
#
#
# class ConditionalGenerator(nn.Module):
#     def __init__(self, nz=100, ngf=64, img_size=32, nc=3, num_classes=10):
#         super(ConditionalGenerator, self).__init__()
#
#         self.label_emb = nn.Embedding(num_classes, nz)
#
#         self.init_size = img_size // 4
#         self.l1 = nn.Sequential(nn.Linear(nz * 2, ngf * 2 * self.init_size ** 2))
#
#         self.conv_blocks = nn.Sequential(
#             nn.BatchNorm2d(ngf * 2),
#             nn.Upsample(scale_factor=2),
#
#             nn.Conv2d(ngf * 2, ngf * 2, 3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(ngf * 2),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Upsample(scale_factor=2),
#
#             nn.Conv2d(ngf * 2, ngf, 3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(ngf),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
#             nn.Sigmoid(),
#         )
#
#     def forward(self, z, label):
#         label_inp = self.label_emb(label)
#         gen_input = torch.cat((label_inp, z), -1)
#
#         out = self.l1(gen_input)
#         out = out.view(out.shape[0], -1, self.init_size, self.init_size)
#         img = self.conv_blocks(out)
#         return img
