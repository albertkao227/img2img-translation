# Image-to-Image Translation with Conditional Adversarial Networks 

Paper: https://arxiv.org/abs/1611.07004 

Pix2pix - a common framework for image-to-image translation based on conditional generative adversarial networks (cGANs). The framework is not application specific, and is designed to reasonably translate one possible representation of a scene into another. The general-purpose solution is effective for, but not limited to 
- 1. synthesize photos from label maps 
- 2. reconstruct objects from edge maps 
- 3. colorize images.  

pix2pix architecture consists of a few major components:

- Generator and Discriminator: Pix2Pix model consists of a generator and a discriminator. The generator architecture is based on U-Net, and the discriminator architecture is PatchGAN.

- Loss Functions: The generator loss is a combination of a GAN loss (to fool the discriminator) and an L1 loss (to make the output image close to the target image).

- Optimizers: Different Adam optimizers are used for the generator and discriminator.

- Dataset Loading: Each item returned by DataLoader contains both the input and target images.

- Training Loop: In each epoch, both the generator and discriminator are updated.
