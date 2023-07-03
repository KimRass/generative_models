# References: https://deep-learning-study.tistory.com/646, https://www.tensorflow.org/tutorials/generative/pix2pix

import torch
import torch.nn as nn
import torch.nn.functional as F

from pix2pix.models import Generator, Discriminator


if __name__ == "__main__":
    g = Generator(in_ch=3, out_ch=3)
    input = torch.randn((4, 3, 256, 256))

    d = Discriminator(in_ch=3, out_ch=3)
    input = torch.randn((4, 6, 256, 256))
    d(input).shape
    d(input)[0]

    cgan_loss = nn.BCELoss()
    l1_loss = nn.L1Loss()
    lamb = 100
    loss = cgan_loss + lamb * l1_loss
