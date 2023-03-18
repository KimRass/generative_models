# Paper Summary
- [Image Style Transfer Using Convolutional Neural Networks](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)
## Related Works
## Methodology
## Architecture
- ***We used the feature space provided by a normalized version of the 16 convolutional and 5 pooling layers of the 19-layer VGG network. We normalized the network by scaling the weights such that the mean activation of each convolutional filter over images and positions is equal to one. Such re-scaling can be done for the VGG network without changing its output, because it contains only rectifying linear activation functions and no normalization or pooling over feature maps. We do not use any of the fully connected layers. For image synthesis we found that replacing the maximum pooling operation by average pooling yields slightly more appealing results.***
## Training
### Loss
- A layer with $N_{l}$ distinct filters has $N_{l}$ feature maps each of size $M_{l}$ , where $M_{l}$ is the height times the width of the feature map. So the responses (Comment: outputs) in a layer $l$ can be stored in a matrix $F^{l} \in \mathbb{R}^{N_{l} \times M_{l}}$ where $F^{l}_{ij}$ is the activation of the $i$th filter at position $j$ in layer $l$. Let $\vec{p}$ and $\vec{x}$ be the original image and the image that is generated, and $P^{l}$ and $F^{l}$ their respective feature representation in layer $l$. We then define the squared-error loss between the two feature representations
$$\mathcal{L}_{content}(\vec{p}, \vec{x}, l) = \frac{1}{2}\sum_{i, j}\big(F^{l}_{ij} - P^{l}_{ij}\big)^{2}$$
## References
