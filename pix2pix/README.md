# Paper Summary
- [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/pdf/1611.07004.pdf)
- We demonstrate that this approach is effective at synthesizing photos from label maps, reconstructing objects from edge maps, and colorizing images, among other tasks.
- If we take a naive approach and ask the CNN to minimize the Euclidean distance between predicted and ground truth pixels, it will tend to produce blurry results. This is because Euclidean distance is minimized by averaging all plausible outputs, which causes blurring.
- It would be highly desirable if we could instead specify only a high-level goal, like "make the output indistinguishable from reality", and then automatically learn a loss function appropriate for satisfying this goal. Fortunately, this is exactly what is done by the recently proposed Generative Adversarial Networks (GANs) [24]. GANs learn a loss that tries to classify if the output image is real or fake, while simultaneously training a generative model to minimize this loss. Blurry images will not be tolerated since they look obviously fake. Because GANs learn a loss that adapts to the data, they can be applied to a multitude of tasks that traditionally would require very different kinds of loss functions.
- In this paper, we explore GANs in the conditional setting. Just as GANs learn a generative model of data, conditional GANs (cGANs) learn a conditional generative model [24]. This makes cGANs suitable for image-to-image translation tasks, where we condition on an input image and generate a corresponding output image. Our primary contribution is to demonstrate that on a wide variety of problems, conditional GANs produce reasonable results.
## Methodology
- ***Unlike past work, for our generator we use a "U-Net"-based architecture [50], and for our discriminator we use a convolutional "PatchGAN" classifier, which only penalizes structure at the scale of image patches. A similar PatchGAN architecture was previously proposed in to capture local style statistics. Here we show that this approach is effective on a wider range of problems, and we investigate the effect of changing the patch size.***
- GANs are generative models that learn a mapping from random noise vector $z$ to output image $y$, $G : z → y$ [24]. In contrast, ***conditional GANs learn a mapping from observed image ***$x$*** and random noise vector ***$z$***, to ***$y$***, ***$G : \{x, z\} → y$***. The generator G is trained to produce outputs that cannot be distinguished from "real" images by an adversarially trained discriminator, ***$D$***, which is trained to do as well as possible at detecting the generator’s "fakes".***
- The objective of a conditional GAN can be expressed as
$$\mathcal{L}_{cGAN}(G, D) = \mathbb{E}_{x,y}[\log D(x, y)] + \mathbb{E}_{x,z}[log(1 − D(x, G(x, z))]$$
- ***where ***$G$*** tries to minimize this objective against an adversarial ***$D$*** that tries to maximize it, i.e. ***$G^{*} = \arg \min_{G} \max_{D} \mathcal{L}_{cGAN}(G, D)$***.***
## Related Works
- Structured loss
    - Conditional GANs instead learn a structured loss.
## References
- [24] [Generative Adversarial Nets](https://arxiv.org/pdf/1406.2661.pdf)
- [50] [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597.pdf)