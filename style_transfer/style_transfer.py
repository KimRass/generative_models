# References
    # https://github.com/Aleadinglight/Pytorch-VGG-19/blob/master/VGG_19.ipynb
    # https://deep-learning-study.tistory.com/680?category=983681

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torchvision.models import vgg19_bn, VGG19_BN_Weights


class FeatureMapExtractor():
    def __init__(self, model):
        self.model = model

        self.feat_map = None

    def get_feature_map(self, image, layer):
        def forward_hook_fn(module, input, output):
            self.feat_map = output

        trg_layer = _get_target_layer(layer)
        trg_layer.register_forward_hook(forward_hook_fn)

        self.model(image)
        return self.feat_map


class ContentLoss(nn.Module):
    def __init__(self, model, layer):
        super().__init__()

        self.layer = layer

        self.feat_map_extractor = FeatureMapExtractor(model)

    def forward(self, gen_image, content_image):
        feat_map_gen = self.feat_map_extractor.get_feature_map(image=gen_image, layer=self.layer)
        feat_map_content = self.feat_map_extractor.get_feature_map(image=content_image, layer=self.layer)
        x = F.mse_loss(feat_map_gen, feat_map_content, reduction="sum")
        x /= 2
        return x


def get_gram_matrix(feat_map):
    b, c, _, _ = feat_map.shape
    x1 = feat_map.view((b, c, -1))
    x2 = torch.transpose(x1, dim0=1, dim1=2)
    x = torch.matmul(x1, x2)
    return x


def _get_contribution_of_layer(feat_map1, feat_map2):
    gram_mat1 = get_gram_matrix(feat_map1)
    gram_mat2 = get_gram_matrix(feat_map2)

    _, c, h, w = feat_map1.shape
    contrib = 0.5 * F.mse_loss(gram_mat1, gram_mat2, reduction="sum") / (c * h * w) ** 2
    return contrib


class StyleLoss(nn.Module):
    def __init__(self, model, weights, layers):
        super().__init__()

        self.weights = weights
        self.layers = layers

        self.feat_map_extractor = FeatureMapExtractor(model)

    def forward(self, gen_image, style_image):
        x = 0
        for weight, layer in zip(self.weights, self.layers):
            feat_map_gen = self.feat_map_extractor.get_feature_map(image=gen_image, layer=layer)
            feat_map_style = self.feat_map_extractor.get_feature_map(image=style_image, layer=layer)
            contrib = _get_contribution_of_layer(feat_map1=feat_map_gen, feat_map2=feat_map_style)
            x += weight * contrib
        return x


class TotalLoss(nn.Module):
    def __init__(
        self,
        model,
        content_layer="features.40",
        style_weights=(0.2, 0.2, 0.2, 0.2, 0.2),
        style_layers=("features.0", "features.7", "features.14", "features.27", "features.40"),
        lamb=100
    ):
        super().__init__()

        self.content_layer = content_layer
        self.style_weights = style_weights
        self.style_layers = style_layers
        self.lamb = lamb

        self.content_loss = ContentLoss(model=model, layer=content_layer)
        self.style_loss = StyleLoss(model=model, weights=style_weights, layers=style_layers)
    
    def forward(self, gen_image, content_image, style_image):
        assert (
            gen_image.shape[0] == 1 and content_image.shape[0] == 1 and style_image.shape[0] == 1,
            "Each batch size should be 1!"
        )

        x1 = self.content_loss(gen_image=gen_image, content_image=content_image)
        x2 = self.style_loss(gen_image=gen_image, style_image=style_image)
        x = x1 + self.lamb * x2
        return x


def print_layers_information(model):
    for name, module in model.named_modules():
        if isinstance(
            module,
            (nn.Linear, nn.Conv2d, nn.MaxPool2d, nn.AdaptiveMaxPool2d, nn.ReLU)
        ):
            print(f"""| {name:20s}: {str(type(module)):50s} |""")


def _get_target_layer(layer):
    return eval(
        "model" + "".join(
            [f"""[{i}]""" if i.isdigit() else f""".{i}""" for i in layer.split(".")]
        )
    )


if __name__ == "__main__":
    content_img = load_image("D:/golden-retriever-royalty-free-image-506756303-1560962726.jpg")
    style_img = load_image("D:/stary-night_orig.jpg")
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
    style_image = transform(style_img).unsqueeze(0)
    gen_image = content_image.clone()

    model = vgg19_bn(weights=VGG19_BN_Weights.IMAGENET1K_V1)
    model.eval()
    # print_layers_information(model)

    gen_image.requires_grad_()
    optimizer = optim.Adam(params=[gen_image], lr=0.01)
    # content_loss = ContentLoss(model=model, layer="features.40")
    # style_loss = StyleLoss(model=model, weights=(0.2, 0.2, 0.2, 0.2, 0.2), layers=("features.0", "features.7", "features.14", "features.27", "features.40"))
    # content_loss(gen_image=gen_image, content_image=content_image)
    # style_loss(gen_image=gen_image, style_image=style_image)

    n_epochs = 300
    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad()

        total_loss = TotalLoss(model=model, lamb=500)
        loss = total_loss(gen_image=gen_image, content_image=content_image, style_image=style_image)

        loss.backward()
        print(loss.item())

        optimizer.step()