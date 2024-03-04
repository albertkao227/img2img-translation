import torch 
import torch.nn.finctional as F 
from torch import nn 
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.data import DataLoader
import matplotlib.pyplot as plt 


def crop(image, new_shape):
	mid_h = image.shape[2]//2
	mid_w = image.shape[3]//2
    start_h = mid_h - new_shape[2]//2
    end_h = start_h + new_shape[2]
    start_w = mid_w - new_shape[3]//2
    end_w = start_w + new_shape[3]
    cropped_image = image[:,:, start_h:end_h, start_w:end_w]
    return cropped_image 


class ContractingBlock(nn.Module):
	'''
    Two convolutions followed by a max pool operation. 
	'''
	def __init__(self, input_channel):
		super(ContractingBlock, self).__init__()
		self.conv1 = nn.Conv2d(input_channel, 2*input_channel, kernel_size=3)
		self.conv2 = nn.Conv2d(2*input_channel, 2*input_channel, kernel_size=3)
		self.activation = nn.ReLU()
		self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

	def forward(self, x):
		'''
		Given a image tensor, completes a contracting block and returns transformed block.
		'''
		x = self.conv1(x)
		x = self.activation(x)
		x = self.conv2(x)
		x = self.activation(x)
		x = self.maxpool(x)
		return x 


class ExpandingBlock(nn.Module):
    '''
    Performs upsampling, one conv, concatenation two inputs, and two convs. 
    ''' 
    def __init__(self, input_channels):
    	super(ExpandingBlock, self).__init__()
    	self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    	self.conv1 = nn.Conv2d(input_channels, input_channels//2, kernel_size=2)
    	self.conv2 = nn.Conv2d(input_channels, input_channels//2, kernel_size=3)
    	self.conv3 = nn.Conv2d(input_channels//2, input_channels//2, kernel_size=3)
    	self.activation = nn.ReLU() 


    def forward(self, x, skip_con_x):
    	'''
    	x: image tensor of shape batch_size, channel, height, width
    	skip_con_x: image tensor from contracting path for skip connection  
    	'''
        x = self.upsample(x)
        x = self.conv1(x)
        skip_con_x = crop(skip_con_x, x.shape)
        x = torch.cat([x, skip_con_x], axis=1)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.conv3(x)
        x = self.activation(x)
        return x 


class FeatureMapBlock(nn.Module):
	'''
	Final layer for UNet. 
	Map each pixel to pixel with correct output dimension using 1X1 conv.  
	'''
	def __init__(self, input_channels, output_channels):
		super(FeatureMapBlock, self).__init__():
		self.conv = nn.Conv2d(input_channels, output_channels, kerner_size=1)

	def forward(self, x):
		x = self.conv(x)
		return x  


class UNet(nn.Module):
	def __init__(self, input_channels, output_channels, hidden_size=64):
		super(UNet, self).__init__()
		sefl.upfeatures = FeatureMapBlock(input_channels, hidden_channels)
		self.contract1 = ContractingBlock(hidden_channels)
		self.contract2 = ContractingBlock(hidden_channels * 2)
		self.contract3 = ContractingBlock(hidden_channels * 4)
        self.contract4 = ContractingBlock(hidden_channels * 8)
        self.expand1 = ExpandingBlock(hidden_channels * 16)
        self.expand2 = ExpandingBlock(hidden_channels * 8)
        self.expand3 = ExpandingBlock(hidden_channels * 4)
        self.expand4 = ExpandingBlock(hidden_channels * 2)
        self,downfeature = FeatureMapBlock(hidden_channels, output_channels)

    def forward(self, x):
    	x0 = self.upfeatures(x)
    	x1 = self.contract1(x0)
    	x2 = self.contract2(x1)
    	x3 = self.contract3(x2)
    	x4 = self.contract4(x3)
    	x5 = self.expand1(x4, x3)
    	x6 = self.expand2(x5, x2)
    	x7 = self.expand3(x6, x1)
    	x8 = self.expand4(x7, x0)
    	xn = self.contract1(x8)
    	return xn 


def train():
	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    unet = UNet(input_dim, label_dim).to_device(device) 
    unet_opt = torch.optim.Adam(unet.parameters(), lr=lr)
    cur_step = 0 

    for epoch in range(n_epochs):
    	for real, labels in tqdm(dataloader):
    		cur_batch_size = len(real)
    		real = real.to(device)
    		labels = labels.to(device)
    		unet_opt.zero_grad()
    		pred = unet(real)
    		unet_loss = criterion(pred, labels)
    		unet_opt.step()

    		if cur_step % display_step == 0:
    			print('UNet loss: ', unet_loss.item())

    		cur_step += 1


if __name__ == "__main__":
    criterion = nn.BCEWithLogitsLoss()
    n_epochs = 200
    input_dim = 1
    label_dim = 1
    display_step = 20 
    batch_size = 4 
    lr = 0.0002
    initial_shape = 512
    target_shape = 373 
    device = 'cuda'














