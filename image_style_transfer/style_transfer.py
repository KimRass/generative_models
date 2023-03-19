# References
    # https://github.com/Aleadinglight/Pytorch-VGG-19/blob/master/VGG_19.ipynb
    # https://deep-learning-study.tistory.com/680?category=983681

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
        origin_feat_map = model.features[: self.layer_idx](origin_image)
        gen_feat_map = model.features[: self.layer_idx](gen_image)
        x = self.mse_loss(origin_feat_map, gen_feat_map)
        x /= 2
        return x


# class GramMatrix(nn.Module):
#     def __init__(self, layer_idx):
#         super().__init__()

#         self.layer_idx = layer_idx

#     def forward(self, image):
#         image = torch.randn((4, 3, 224, 224))
#         layer_idx=52
#         feat_map = model.features[: layer_idx](image)

#         b, c, _, _ = feat_map.shape
#         x1 = feat_map.view((b, c, -1))
#         x2 = torch.transpose(x1, dim0=1, dim1=2)
#         x = torch.matmul(x1, x2)
#         return x


def get_gram_matrix(feat_map):
    b, c, _, _ = feat_map.shape
    x1 = feat_map.view((b, c, -1))
    x2 = torch.transpose(x1, dim0=1, dim1=2)
    x = torch.matmul(x1, x2)
    return x


def _get_layers(model):
    return [
        (name, type(module))
        for name, module
        in model.named_modules()
        if isinstance(
            module,
            (nn.Linear, nn.Conv2d, nn.MaxPool2d, nn.AdaptiveMaxPool2d, nn.ReLU)
        )
    ]


def _get_target_layer(layer_name):
    return eval(
        "model" + "".join(
            [f"""[{i}]""" if i.isdigit() else f""".{i}""" for i in layer_name.split(".")]
        )
    )


class FeatureMapExtractor():
    def __init__(self, model):
        self.model = model

        self.feat_map = None

    def get_feature_map(self, image, layer_name):
        def forward_hook_fn(module, input, output):
            self.feat_map = output

        trg_layer = _get_target_layer(layer_name)
        trg_layer.register_forward_hook(forward_hook_fn)

        self.model(image)
        return self.feat_map

model = vgg19_bn(weights=VGG19_BN_Weights.IMAGENET1K_V1)
model.eval()
layers = _get_layers(model)

feat_map_extractor = FeatureMapExtractor(model)
feat_map = feat_map_extractor.get_feature_map(image=content_image, layer_name="features.45")
feat_map.shape
gram_matrix = get_gram_matrix(feat_map)
gram_matrix.shape


content_img = load_image("/Users/jongbeomkim/Downloads/download.png")
style_img = load_image("/Users/jongbeomkim/Downloads/horses.jpeg")
# input = torch.randn((4, 3, 224, 224))

transform = T.Compose(
    [
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]
)
content_image = transform(content_img).unsqueeze(0)
temp = model(content_image)

origin_image = torch.randn((4, 3, 224, 224))
gen_image = torch.randn((4, 3, 224, 224))


content_loss = ContentLoss(layer_idx=52)
content_loss(origin_image=origin_image, gen_image=gen_image)