# code adapted from https://www.coursera.org/learn/apply-generative-adversarial-networks-gans 

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model import *


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)


def get_gen_loss(gen, disc, real, condition, adv_criterion, recon_criterion, lambda_recon):
    '''
    Return the loss of the generator given inputs.
    Parameters:
        gen: the generator; takes the condition and returns potential images
        disc: the discriminator; takes images and the condition and
          returns real/fake prediction matrices
        real: the real images (e.g. maps) to be used to evaluate the reconstruction
        condition: the source images (e.g. satellite imagery) which are used to produce the real images
        adv_criterion: the adversarial loss function; takes the discriminator 
                  predictions and the true labels and returns a adversarial 
                  loss (which you aim to minimize)
        recon_criterion: the reconstruction loss function; takes the generator 
                    outputs and the real images and returns a reconstructuion 
                    loss (which you aim to minimize)
        lambda_recon: the degree to which the reconstruction loss should be weighted in the sum
    '''
    # Steps: 1) Generate the fake images, based on the conditions.
    #        2) Evaluate the fake images and the condition with the discriminator.
    #        3) Calculate the adversarial and reconstruction losses.
    #        4) Add the two losses, weighting the reconstruction loss appropriately.

    fake = gen(condition)
    disc_fake_hat = disc(fake, condition)
    adv_loss = adv_criterion(disc_fake_hat, torch.ones_like(disc_fake_hat))
    recon_loss = recon_criterion(real, fake)
    gen_loss = adv_loss + lambda_recon * recon_loss

    return gen_loss


def show_tensor_images(images, num_images=25, size=(1, 28, 28)):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    def transform_img(image_tensor, num_images, size):
        image_shifted = image_tensor
        image_unflat = image_shifted.detach().cpu().view(-1, *size)
        image_grid = make_grid(image_unflat[:num_images], nrow=5)
        return image_grid.permute(1, 2, 0).squeeze()
    
    condition, real, fake = images 
    condition = transform_img(condition, num_images, size)
    real = transform_img(real, num_images, size)
    fake = transform_img(fake, num_images, size)
    fig, ax = plt.subplots(3, 1, figsize=(15, 15)) 
    ax[0].imshow(condition)
    ax[0].set_title('Condition')
    ax[1].imshow(real)
    ax[1].set_title('Real')
    ax[2].imshow(fake)
    ax[2].set_title('Fake') 
    

def train(save_model=False):
    mean_generator_loss = 0
    mean_discriminator_loss = 0
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    cur_step = 0

    for epoch in range(n_epochs):
        # Dataloader returns the batches
        for image, _ in tqdm(dataloader):
                        
            image_width = image.shape[3]
            real = image[:, :, :, :image_width // 2]
            real = nn.functional.interpolate(real, size=target_shape)
            condition = image[:, :, :, image_width // 2:]
            condition = nn.functional.interpolate(condition, size=target_shape)
            cur_batch_size = len(condition)
            condition = condition.to(device)
            real = real.to(device)

            ### Update discriminator ###
            disc_opt.zero_grad() # Zero out the gradient before backpropagation
            with torch.no_grad():
                fake = gen(condition)
            disc_fake_hat = disc(fake.detach(), condition) # Detach generator
            disc_fake_loss = adv_criterion(disc_fake_hat, torch.zeros_like(disc_fake_hat))
            disc_real_hat = disc(real, condition)
            disc_real_loss = adv_criterion(disc_real_hat, torch.ones_like(disc_real_hat))
            disc_loss = (disc_fake_loss + disc_real_loss) / 2
            disc_loss.backward(retain_graph=True) # Update gradients
            disc_opt.step() # Update optimizer

            ### Update generator ###
            gen_opt.zero_grad()
            gen_loss = get_gen_loss(gen, disc, real, condition, adv_criterion, recon_criterion, lambda_recon)
            gen_loss.backward() # Update gradients
            gen_opt.step() # Update optimizer

            # Keep track of the average discriminator loss
            mean_discriminator_loss += disc_loss.item() / display_step
            # Keep track of the average generator loss
            mean_generator_loss += gen_loss.item() / display_step

            ### Visualization code ###
            if cur_step % display_step == 0:
                if cur_step > 0:
                    print(f"Epoch {epoch}: Step {cur_step}: Generator (U-Net) loss: {mean_generator_loss}, Discriminator loss: {mean_discriminator_loss}")
                else:
                    print("Pretrained initial state")
                if cur_step % 100 == 0:                
                    show_tensor_images((condition, real, fake), size=(input_dim, target_shape, target_shape))
                    # show_tensor_images(real, size=(real_dim, target_shape, target_shape))
                    # show_tensor_images(fake, size=(real_dim, target_shape, target_shape))
                mean_generator_loss = 0
                mean_discriminator_loss = 0
                # You can change save_model to True if you'd like to save the model
                if save_model:
                    torch.save({'gen': gen.state_dict(),
                        'gen_opt': gen_opt.state_dict(),
                        'disc': disc.state_dict(),
                        'disc_opt': disc_opt.state_dict()
                    }, f"pix2pix_{cur_step}.pth")
            cur_step += 1
            


if __name__ == "__main__":

    adv_criterion = nn.BCEWithLogitsLoss() 
    recon_criterion = nn.L1Loss() 
    lambda_recon = 200
    n_epochs = 20
    input_dim = 3
    real_dim = 3
    display_step = 200
    batch_size = 4
    lr = 0.0002
    target_shape = 256
    device = 'cuda'
    root_path = ''
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = torchvision.datasets.ImageFolder(root_path, transform=transform)
    gen = UNet(input_dim, real_dim).to(device)
    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
    disc = Discriminator(input_dim + real_dim).to(device)
    disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)

    pretrained = False
    gen = gen.apply(weights_init)
    disc = disc.apply(weights_init)

    dataloader = DataLoader(dataset, batch_size=batch_size)
    print('Number of images in training dataset:', len(dataloader.dataset))
    print('Number of batches for training datasets:', len(dataloader))  
    train()
    plt.show()

