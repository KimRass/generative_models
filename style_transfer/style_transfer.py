# References
    # https://github.com/Aleadinglight/Pytorch-VGG-19/blob/master/VGG_19.ipynb
    # https://deep-learning-study.tistory.com/680?category=983681

import argparse
from pathlib import Path
import requests
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torchvision.models import vgg19_bn, VGG19_BN_Weights


def _get_target_layer(layer):
    return eval(
        "model" + "".join(
            [f"""[{i}]""" if i.isdigit() else f""".{i}""" for i in layer.split(".")]
        )
    )


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

        self.model = model
        self.layer = layer

        self.content_feat_map = FeatureMapExtractor(model).get_feature_map(image=content_image, layer=layer)

    def forward(self, gen_image):
        gen_feat_map = FeatureMapExtractor(self.model).get_feature_map(image=gen_image, layer=self.layer)
        x = F.mse_loss(gen_feat_map, self.content_feat_map, reduction="sum")
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

        self.model = model
        self.weights = weights
        self.layers = layers

        self.style_feat_maps = [
            FeatureMapExtractor(model).get_feature_map(image=style_image, layer=layer) for layer in self.layers
        ]

    def forward(self, gen_image):
        x = 0
        for weight, layer, style_feat_map in zip(self.weights, self.layers, self.style_feat_maps):
            gen_feat_map = FeatureMapExtractor(self.model).get_feature_map(image=gen_image, layer=layer)
            contrib = _get_contribution_of_layer(feat_map1=gen_feat_map, feat_map2=style_feat_map)
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

        self.model = model
        self.content_layer = content_layer
        self.style_weights = style_weights
        self.style_layers = style_layers
        self.lamb = lamb

        self.content_loss = ContentLoss(model=model, layer=content_layer)
        self.style_loss = StyleLoss(model=model, weights=style_weights, layers=style_layers)
    
    def forward(self, gen_image):
        assert (
            gen_image.shape[0] == 1 and content_image.shape[0] == 1 and style_image.shape[0] == 1,
            "The batch size should be 1!"
        )

        x1 = self.content_loss(gen_image)
        x2 = self.style_loss(gen_image)
        x = x1 + self.lamb * x2
        return x


def print_all_layers(model):
    for name, module in model.named_modules():
        if isinstance(
            module,
            (nn.Linear, nn.Conv2d, nn.MaxPool2d, nn.AdaptiveMaxPool2d, nn.ReLU)
        ):
            print(f"""| {name:20s}: {str(type(module)):50s} |""")


def denormalize_array(img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    copied_img = img.copy()
    copied_img *= variance
    copied_img += mean
    copied_img *= 255.0
    copied_img = np.clip(a=copied_img, a_min=0, a_max=255).astype("uint8")
    return copied_img


def convert_tensor_to_array(tensor):
    copied_tensor = tensor.clone().squeeze().permute((1, 2, 0)).detach().cpu().numpy()
    copied_tensor = denormalize_array(copied_tensor)
    return copied_tensor


def load_image(url_or_path=""):
    url_or_path = str(url_or_path)

    if "http" in url_or_path:
        img_arr = np.asarray(
            bytearray(requests.get(url_or_path).content), dtype="uint8"
        )
        img = cv2.imdecode(img_arr, flags=cv2.IMREAD_COLOR)
        img = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2RGB)
    else:
        img = cv2.imread(url_or_path, flags=cv2.IMREAD_COLOR)
        img = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2RGB)
    return img


def _downsample_image(img):
    return cv2.pyrDown(img)


def save_image(img, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(
        filename=str(path), img=img[:, :, :: -1], params=[cv2.IMWRITE_JPEG_QUALITY, 100]
    )


def get_arguments():
    parser = argparse.ArgumentParser(description="train_craft")

    parser.add_argument("--content_image")
    parser.add_argument("--style_image")
    parser.add_argument("--save_dir", default="samples")
    parser.add_argument("--style_weight", type=int, default=1_000)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_arguments()

    cuda = torch.cuda.is_available()

    content_img = load_image(args.content_image)
    content_img = _downsample_image(content_img)
    style_img = load_image(args.style_image)
    h, w, _ = content_img.shape
    # gen_img = np.random.randint(low=0, high=256, size=(h, w, 3), dtype="uint8")
    gen_img = content_img.copy()


    transform1 = T.Compose(
        [
            T.ToTensor(),
            # T.CenterCrop(224),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]
    )
    transform2 = T.Compose(
        [
            T.ToTensor(),
            T.Resize((h, w)),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]
    )
    content_image = transform1(content_img).unsqueeze(0)
    style_image = transform2(style_img).unsqueeze(0)
    gen_image = transform1(gen_img).unsqueeze(0)
    if cuda:
        content_image = content_image.cuda()
        style_image = style_image.cuda()
        gen_image = gen_image.cuda()
    # temp = convert_tensor_to_array(gen_image)
    # show_image(temp)

    model = vgg19_bn(weights=VGG19_BN_Weights.IMAGENET1K_V1)
    model.eval()
    if cuda:
        model = model.cuda()

    feat_map_extractor = FeatureMapExtractor(model)

    gen_image.requires_grad_()
    optimizer = optim.Adam(params=[gen_image], lr=0.03)

    criterion = TotalLoss(model=model, lamb=args.style_weight)

    n_epochs = 30_000
    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad()

        loss = criterion(gen_image)

        loss.backward(retain_graph=True)

        optimizer.step()
        if epoch % 200 == 0:
            print(f"""| Epoch: {epoch:5d} | Loss: {loss.item(): .2f} |""")

            gen_img = convert_tensor_to_array(gen_image)
            save_image(
                img=gen_img,
                path=Path(args.save_dir)/f"""{Path(args.content_image).stem}_{Path(args.style_image).stem}_lambda{args.style_weight}_epoch{epoch}.jpg"""
            )
