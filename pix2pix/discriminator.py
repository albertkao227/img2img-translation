import torch
import torch.nn as nn

class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels):
        super(PatchGANDiscriminator, self).__init__()

        def discriminator_block(in_channels, out_channels, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels * 2, 64, normalization=False),  # No normalization on the first layer
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.Conv2d(512, 1, 4, padding=1)  # Output layer
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)

# Example of creating a PatchGAN Discriminator
in_channels = 3  # Assuming RGB images
discriminator = PatchGANDiscriminator(in_channels)

# Example inputs (real and condition images)
real_image = torch.randn(1, in_channels, 256, 256)  # Real image
condition_image = torch.randn(1, in_channels, 256, 256)  # Condition image (e.g., input for image-to-image translation)

# Forward pass
discriminator_output = discriminator(real_image, condition_image)
print(discriminator_output.shape)  # Shape depends on the input image size