# Paper Summary
- [Image Style Transfer Using Convolutional Neural Networks](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)
- 원본 논문의 수식과 다른 표기를 사용합니다:
    - $\vec{a}\rightarrow\vec{s}$, $\vec{p} \rightarrow \vec{c}$, $A \rightarrow G_{s}$, $P \rightarrow F_{c}$ and $\frac{\alpha}{\beta} \rightarrow \lambda$
## Methodology
- Figure 1
    - <img src="https://user-images.githubusercontent.com/67457712/226185150-e19f3a4e-457f-4534-94f6-1a080b56528b.png" width="800">
    - While the number of different filters increases along the processing hierarchy, the size of the filtered images is reduced by some downsampling mechanism (e.g. max-pooling) leading to a decrease in the total number of units per layer of the network.
    - Content Reconstructions
        - We can visualise the information at different processing stages in the CNN by reconstructing the input image from only knowing the network's responses in a particular layer. We reconstruct the input image from from layers 'conv1_2' (a), 'conv2_2' (b), 'conv3_2' (c), 'conv4_2' (d) and 'conv5_2' (e) of the original VGG-Network.
        - (a ~ c) Reconstructions from the lower layers simply reproduce the exact pixel values of the original image.
        - (d ~ e) In contrast, higher layers in the network capture the high-level content in terms of objects and their arrangement in the input image but do not constrain the exact pixel values of the reconstruction very much. We therefore refer to the feature responses in higher layers of the network as the content representation.
    - Style Reconstructions
        - The style representation computes correlations between the different features in different layers of the CNN. We reconstruct the style of the input image from a style representation built on different subsets of CNN layers ('conv1 1' (a), 'conv1_1' and 'conv2_1' (b), 'conv1_1', 'conv2_1' and 'conv3_1' (c), 'conv1_1', 'conv2_1', 'conv3_1' and 'conv4_1' (d), 'conv1_1', 'conv2_1', 'conv3_1', 'conv4_1' and 'conv5_1' (e)). This creates images that match the style of a given image on an increasing scale while discarding information of the global arrangement of the scene.
        - We can visualise the information captured by these style feature spaces built on different layers of the network by constructing an image that matches the style representation of a given input image.
    - ***We find that matching the style representations up to higher layers in the network preserves local images structures an increasingly large scale, leading to a smoother and more continuous visual experience.*** Thus, the visually most appealing images are usually created by matching the style representation up to high layers in the network, which is why for all images shown we match the style features in layers 'conv1_1', 'conv2_1', 'conv3_1', 'conv4_1' and 'conv5_1' of the network.
- Figure 2
    - <img src="https://user-images.githubusercontent.com/67457712/226184028-5db9cb50-fae1-459d-8ad6-25597e60eedc.png" width="800">
    - First content and style features are extracted and stored. ***The style image*** $\vec{s}$ ***is passed through the network and its style representation*** $G^{l}_{s}$ ***on all layers included are computed and stored (left). The content image*** $\vec{c}$ ***is passed through the network and the content representation*** $F^{l}_{c}$ ***in one layer is stored (right).***
    - ***Then a random white noise image*** $\vec{x}$ ***is passed through the network and its style features*** $G^{l}_{x}$ ***and content features*** $F^{l}_{x}$ ***are computed. On each layer included in the style representation, the element-wise mean squared difference between*** $G^{l}_{x}$ ***and*** $G^{l}_{c}$ ***is computed to give the style loss*** $L_{style}$ ***(left). Also the mean squared difference between*** $F^{l}_{x}$ ***and*** $F^{l}_{c}$ ***is computed to give the content loss*** $L_{content}$ ***(right).***
    - The total loss $L_{total}$ is then a linear combination between the content and the style loss. Its derivative with respect to the pixel values can be computed using error back-propagation (middle). ***This gradient is used to iteratively update the image*** $\vec{x}$ ***until it simultaneously matches the style features of the style image*** $\vec{s}$ ***and the content features of the content image*** $\vec{c}$ ***(middle, bottom).***
    - ***We jointly minimise the distance of the fea- ture representations of a white noise image from the content representation of the photograph in one layer and the style representation of the painting defined on a number of layers of the Convolutional Neural Network.***
- The key finding of this paper is that the representations of content and style in the Convolutional Neural Network are well separable.
- Figure 5
    - <img src="https://user-images.githubusercontent.com/67457712/226240563-b39b3987-d9aa-40cd-bf5b-018e056c69cb.png" width="500">
    - ***When matching the content on a lower layer of the network, the algorithm matches much of the detailed pixel information.***
    - ***In contrast, when matching the content features on a higher layer of the network, detailed pixel information of the photograph is not as strongly constraint. That is, the fine structure of the image, for example the edges and color map, is altered such that it agrees with the style of the artwork while displaying the content of the photograph.***
- Image initialization
    - We have initialized all images shown so far with white noise. However, one could also initialize the image synthesis with either the content image or the style image. Although they bias the final image somewhat towards the spatial structure of the initialization, ***the different initializations do not seem to have a strong effect on the outcome of the synthesis procedure. It should be noted that only initializing with noise allows to generate an arbitrary number of new images. Initializing with a fixed image always deterministically leads to the same outcome (up to stochasticity in the gradient descent procedure).***
## Architecture
- ***We used the feature space provided by a normalized version of the 16 convolutional and 5 pooling layers of the 19-layer VGG network. We normalized the network by scaling the weights such that the mean activation of each convolutional filter over images and positions is equal to one. Such re-scaling can be done for the VGG network without changing its output, because it contains only rectifying linear activation functions and no normalization or pooling over feature maps. We do not use any of the fully connected layers. For image synthesis we found that replacing the maximum pooling operation by average pooling yields slightly more appealing results.***
## Inference
### Loss
- Content loss
    - A layer with $N_{l}$ distinct filters has $N_{l}$ feature maps each of size $M_{l}$ , where $M_{l}$ is the height times the width of the feature map.
    - So the responses (Comment: i.e., outputs) in a layer $l$ can be stored in a matrix $F^{l} \in \mathbb{R}^{N_{l} \times M_{l}}$ where $F^{l}_{ij}$ is the activation of the $i$ th filter at position $j$ in layer $l$. Let $\vec{x}$ and $\vec{c}$ be the image to be generated and the content image, and $F^{l}_{x}$ and $F^{l}_{c}$ their respective feature representation in layer $l$. We then define the squared-error loss between the two feature representations
    $$L_{content}(\vec{x}, \vec{c}, l) = \frac{1}{2}\sum_{i, j}\big(F^{l}_{x, ij} - F^{l}_{c, ij}\big)^{2}$$
    - Thus we can change the initially random image $\vec{x}$ until it generates the same response in a certain layer of the Convolutional Neural Network as the content image $\vec{c}$.
- Style loss
    - The feature space can be built on top of the filter responses in any layer of the network. It consists of the correlations between the different filter responses, where the expectation is taken over the spatial extent of the feature maps. These feature correlations are given by the Gram matrix $G^{l} \in \mathbb{R}^{N_{l} \times N_{l}}$, where $G^{l}_{ij}$ is the inner product between the vectorized feature maps $i$ and $j$ in layer $l$
    $$G^{l}_{ij} = \sum_{k}F^{l}_{ik}F^{l}_{jk}$$
    - By including the feature correlations of multiple layers, we obtain a stationary, multi-scale representation of the input image, which captures its texture information but not the global arrangement. This is done by using gradient descent from a white noise image to minimize the mean-squared distance between the entries of the Gram matrices from the style image and the Gram matrices of the image to be generated.
    - Let $\vec{x}$ be $\vec{s}$ the image to be generated and the style image, and $G^{l}_{x}$ and $G^{l}_{s}$ their respective style representation in layer $l$. The contribution of layer $l$ to the total loss is then
    $$E_{l} = \frac{1}{2}\frac{1}{(N_{l}M_{l})^{2}} \sum_{i, j}\big(G^{l}_{x, ij} - G^{l}_{s, ij}\big)^{2}$$
    - and the total style loss is
    $$L_{style}(\vec{x}, \vec{s}) = \sum_{l = 0}^{L}w_{l}E_{l}$$
    - where $w_{l}$ are weighting factors of the contribution of each layer to the total loss. $w_{l} = \frac{1}{5}$ in layers 'conv1_1', 'conv2_1', 'conv3_1', 'conv4_1' and 'conv5_1', $w_{l} = 0$ in all other layers.
- ***The loss function we minimize is***
$$L(\vec{x}, \vec{c}, \vec{s}) = L_{content}(\vec{x}, \vec{c}) + \lambda L_{style}(\vec{x}, \vec{s})$$
- where $\alpha$ and $\beta$ are the weighting factors for content and style reconstruction, respectively.
- ***To extract image information on comparable scales, we always resized the style image to the same size as the content image before computing its feature representations.***
- Figure 4
    - <img src="https://user-images.githubusercontent.com/67457712/226239095-829be6cc-c691-4022-803b-54d49ba35c88.png" width="600">
    - For a specific pair of content and style images one can adjust the trade-off between content and style to create visually appealing images.
    - Top left, $\lambda = 10000$: A strong emphasis on style will result in images that match the appearance of the artwork, effectively giving a texturised ver- sion of it, but show hardly any of the photograph's content.
    - Bottom right, $\lambda = 10$: When placing strong emphasis on content, one can clearly identify the photograph, but the style of the painting is not as well-matched.
## References
- [28] [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556.pdf)
