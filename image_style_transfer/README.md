# Paper Summary
- [Image Style Transfer Using Convolutional Neural Networks](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)
## Related Works
## Methodology
- Figure 1
    - <img src=>
    - (a ~ c) In contrast, reconstructions from the lower layers simply reproduce the exact pixel values of the original image.
    - (d ~ e) Higher layers in the network capture the high-level content in terms of objects and their arrangement in the input image but do not constrain the exact pixel values of the reconstruction very much. We therefore refer to the feature responses in higher layers of the network as the content representation.
## Architecture
- ***We used the feature space provided by a normalized version of the 16 convolutional and 5 pooling layers of the 19-layer VGG network. We normalized the network by scaling the weights such that the mean activation of each convolutional filter over images and positions is equal to one. Such re-scaling can be done for the VGG network without changing its output, because it contains only rectifying linear activation functions and no normalization or pooling over feature maps. We do not use any of the fully connected layers. For image synthesis we found that replacing the maximum pooling operation by average pooling yields slightly more appealing results.***
## Training
### Loss
- Content loss
    - A layer with $N_{l}$ distinct filters has $N_{l}$ feature maps each of size $M_{l}$ , where $M_{l}$ is the height times the width of the feature map.
    - So the responses (Comment: i.e., outputs) in a layer $l$ can be stored in a matrix $F^{l} \in \mathbb{R}^{N_{l} \times M_{l}}$ where $F^{l}_{ij}$ is the activation of the $i$ th filter at position $j$ in layer $l$. Let $\vec{p}$ and $\vec{x}$ be the original image and the image that is generated, and $P^{l}$ and $F^{l}$ their respective feature representation in layer $l$. We then define the squared-error loss between the two feature representations
    $$\mathcal{L}_{content}(\vec{p}, \vec{x}, l) = \frac{1}{2}\sum_{i, j}\big(F^{l}_{ij} - P^{l}_{ij}\big)^{2}$$
    - Thus we can change the initially random image $\vec{x}$ until it generates the same response in a certain layer of the Convolutional Neural Network as the original image $\vec{p}$.
- Style loss
    - To obtain a representation of the style of an input image, we use a feature space designed to capture texture information. This feature space can be built on top of the filter responses in any layer of the network. It consists of the correlations between the different filter responses, where the expectation is taken over the spatial extent of the feature maps. These feature correlations are given by the Gram matrix $G^{l} \in \mathbb{R}^{N_{l} \times N_{l}}$, where $G^{l}_{ij}$ is the inner product between the vectorized feature maps $i$ and $j$ in layer $l$
    $$G^{l}_{ij} = \sum_{k}F^{l}_{ik}F^{l}_{jk}$$
- Figure 2
    - <img src="https://user-images.githubusercontent.com/67457712/226184028-5db9cb50-fae1-459d-8ad6-25597e60eedc.png" width="800">
    - First content and style features are extracted and stored. ***The style image*** $\vec{a}$ ***is passed through the network and its style representation*** $A^{l}$ ***on all layers included are computed and stored (left). The content image*** $\vec{p}$ ***is passed through the network and the content representation*** $P^{l}$ ***in one layer is stored (right). Then a random white noise image*** $\vec{x}$ ***is passed through the network and its style features*** $G^{l}$ ***and content features*** $F^{l}$ ***are computed. On each layer included in the style representation, the element-wise mean squared difference between*** $G^{l}$ ***and*** $A^{l}$ ***is computed to give the style loss*** $L_{style}$ ***(left). Also the mean squared difference between*** $F^{l}$ ***and*** $P^{l}$ ***is computed to give the content loss*** $L_{content}$ ***(right).***
    - The total loss $L_{total}$ is then a linear combination between the content and the style loss. Its derivative with respect to the pixel values can be computed using error back-propagation (middle). ***This gradient is used to iteratively update the image $\vec{x}$ until it simultaneously matches the style features of the style image $\vec{a}$ and the content features of the content image $\vec{p}$ (middle, bottom).***
## References
