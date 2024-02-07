import torch
import torch.nn as nn

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_dropout=False):
        super(UNetBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False) if down 
            else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True) if down else nn.LeakyReLU(0.2, True)
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
        self.down = down

    def forward(self, x, skip_input=None):
        x = self.block(x)
        if self.down:
            return x
        else:
            x = torch.cat([x, skip_input], 1)
            return self.dropout(x) if self.use_dropout else x

class UNetGenerator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetGenerator, self).__init__()
        # Downsampling part
        self.down1 = UNetBlock(in_channels, 64, down=True, use_dropout=False)
        self.down2 = UNetBlock(64, 128, down=True, use_dropout=False)
        self.down3 = UNetBlock(128, 256, down=True, use_dropout=False)
        self.down4 = UNetBlock(256, 512, down=True, use_dropout=True)
        
        # Upsampling part
        self.up1 = UNetBlock(512, 256, down=False, use_dropout=True)
        self.up2 = UNetBlock(512, 128, down=False, use_dropout=False)
        self.up3 = UNetBlock(256, 64, down=False, use_dropout=False)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        
        u1 = self.up1(d4, d3)
        u2 = self.up2(u1, d2)
        u3 = self.up3(u2, d1)
        
        return self.final(u3)

# Example of creating a U-Net Generator
in_channels = 3  # Input image channels
out_channels = 3  # Output image channels
generator = UNetGenerator(in_channels, out_channels)

# Generate a sample input
sample_input = torch.randn(1, in_channels, 256, 256)  # Example input tensor

# Generate an image
fake_image = generator(sample_input)
print(fake_image.shape)  # Should be torch.Size([1, 3, 256, 256])
