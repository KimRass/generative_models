# References
    # https://github.com/Aleadinglight/Pytorch-VGG-19/blob/master/VGG_19.ipynb

import torch
import torch.nn as nn
import torch.functional as F
import torchvision
import torchvision.transforms as T
from torchvision.models import vgg19_bn, VGG19_BN_Weights


class ContentLoss(nn.Module):
    def __init__(self, layer_idx):
        super().__init__()

        self.layer_idx = layer_idx

        self.mse_loss = nn.MSELoss(reduction="sum")

    def forward(self, origin_image, gen_image):
        origin_feat_map = cnn.features[: self.layer_idx](origin_image)
        gen_feat_map = cnn.features[: self.layer_idx](gen_image)
        x = self.mse_loss(origin_feat_map, gen_feat_map)
        x /= 2
        return x


class GramMatrix(nn.Module):
    def __init__(self, layer_idx):
        super().__init__()

        self.layer_idx = layer_idx

    def forward(self, image):
        image = torch.randn((4, 3, 224, 224))
        layer_idx=52
        feat_map = cnn.features[: layer_idx](image)
        b, c, _, _ = feat_map.shape
        x1 = feat_map.view((b, c, -1))
        x2 = feat_map.view((b, -1, c))
        x = torch.matmul(x1, x2)
        return x
        

cnn = vgg19_bn(weights=VGG19_BN_Weights.IMAGENET1K_V1)
cnn.eval()
# with torch.no_grad():


content_img = load_image("/Users/jongbeomkim/Downloads/download.png")
style_img = load_image("/Users/jongbeomkim/Downloads/horses.jpeg")

input = torch.randn((4, 3, 224, 224))
cnn.features[0](input)
cnn.features[: layer_idx](input).shape

transform = T.Compose(
    [
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]
)
transform(content_img).shape

origin_image = torch.randn((4, 3, 224, 224))
gen_image = torch.randn((4, 3, 224, 224))


content_loss = ContentLoss(layer_idx=52)
content_loss(origin_image=origin_image, gen_image=gen_image)