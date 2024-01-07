# Image-to-Image Translation with Conditional Adversarial Networks 

  
- Generator and Discriminator: Pix2Pix model consists of a generator and a discriminator. The generator architecture is based on U-Net, and the discriminator architecture is PatchGAN.

- Loss Functions: The generator loss is a combination of a GAN loss (to fool the discriminator) and an L1 loss (to make the output image close to the target image).

- Optimizers: Different Adam optimizers are used for the generator and discriminator.

- Dataset Loading: Each item returned by DataLoader contains both the input and target images.

- Training Loop: In each epoch, both the generator and discriminator are updated.
