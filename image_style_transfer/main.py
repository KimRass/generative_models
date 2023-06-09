# 뭔가 이미지가 이상하게 생성된다! 코드 재점검 필요!!

# References:
    # https://github.com/Aleadinglight/Pytorch-VGG-19/blob/master/VGG_19.ipynb
    # https://deep-learning-study.tistory.com/680?category=983681

from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from torchvision.models import vgg19_bn, VGG19_BN_Weights

from utils import get_args
from image_utils import load_image, save_image, downsample
from torch_utils import tensor_to_array, print_all_layers, FeatureMapExtractor


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


if __name__ == "__main__":
    args = get_args()

    cuda = torch.cuda.is_available()

    content_img = load_image(args.content_image)
    content_img = downsample(content_img)
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
    # temp = tensor_to_array(gen_image)
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

            gen_img = tensor_to_array(gen_image)
            save_image(
                img=gen_img,
                path=Path(args.save_dir)/f"""{Path(args.content_image).stem}_{Path(args.style_image).stem}_lambda{args.style_weight}_epoch{epoch}.jpg"""
            )
