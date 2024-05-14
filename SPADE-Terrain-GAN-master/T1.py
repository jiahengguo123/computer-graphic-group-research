import os
import torch
import glob
import numpy as np
import time
import matplotlib as plt
import cv2
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from colorspacious import cspace_convert
import torch.optim as optim
from torchvision import models
from torchvision.transforms import Normalize
from torchvision.utils import save_image
from torchvision.models import vgg19

PATH ='/Users/yaochen/PycharmProjects/computer-graphic-group-research/SPADE-Terrain-GAN-master/archive'

BUFFER_SIZE = 50
BATCH_SIZE = 5
IMG_WIDTH = 256
IMG_HEIGHT = 256


class CustomDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_files = glob.glob(os.path.join(image_dir, '*_i2.png'))
        self.transform = transform
        self.color_map = np.array([
            [17, 141, 215],
            [225, 227, 155],
            [127, 173, 123],
            [185, 122, 87],
            [230, 200, 181],
            [150, 150, 150],
            [193, 190, 175]
        ], dtype=np.float32) / 255.0  # Normalize colors to match PyTorch's [0,1] input range

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img = Image.open(img_path).convert('RGB')
        trf_img = Image.open(img_path.replace('_i2.png', '_t.png')).convert('RGB')
        htf_img = Image.open(img_path.replace('_i2.png', '_h.png')).convert('L')

        if self.transform:
            img = self.transform(img)
            trf_img = self.transform(trf_img)
            htf_img = self.transform(htf_img)

        # Create one-hot encoding
        input_tensor = transforms.functional.to_tensor(img).unsqueeze(0)  # Add batch dimension
        one_hot = torch.zeros((7, img.height, img.width), dtype=torch.float32)
        for i, color in enumerate(self.color_map):
            mask = (input_tensor == torch.tensor(color, dtype=torch.float32).view(3, 1, 1)).all(dim=0)
            one_hot[i][mask] = 1

        return one_hot, transforms.functional.to_tensor(trf_img), transforms.functional.to_tensor(htf_img).unsqueeze(0)


transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

train_dataset = CustomDataset('/Users/yaochen/PycharmProjects/computer-graphic-group-research/SPADE-Terrain-GAN-master/data/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)

test_dataset = CustomDataset('/Users/yaochen/PycharmProjects/computer-graphic-group-research/SPADE-Terrain-GAN-master/data/test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=5, shuffle=False)



def conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1, pad_type='reflect', use_bias=True):
    if pad_type == 'reflect':
        padding_layer = nn.ReflectionPad2d(padding)
    else:
        padding_layer = nn.ZeroPad2d(padding)
    return nn.Sequential(
        padding_layer,
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, bias=use_bias)
    )


class AddNoise(nn.Module):
    def __init__(self, channels):
        super(AddNoise, self).__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        noise = torch.randn_like(x)
        return x + self.weight * noise


class SPADE(nn.Module):
    def __init__(self, norm_nc, label_nc):
        super(SPADE, self).__init__()

        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)

        nhidden = 128
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)

    def forward(self, x, segmap):
        normalized = self.param_free_norm(x)

        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        out = normalized * (1 + gamma) + beta
        return out


class SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, label_nc):
        super(SPADEResnetBlock, self).__init__()
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        self.conv_0 = SPADE(fin, label_nc)
        self.conv_1 = SPADE(fmiddle, label_nc)
        self.conv_s = SPADE(fin, label_nc) if self.learned_shortcut else None

        self.conv_last = conv(fmiddle, fout, kernel_size=3, padding=1, use_bias=False)

    def forward(self, x, seg):
        x_s = self.shortcut(x, seg)

        dx = self.conv_0(x, seg)
        dx = F.leaky_relu(dx, 0.2)
        dx = self.conv_1(dx, seg)
        dx = F.leaky_relu(dx, 0.2)
        dx = self.conv_last(dx)

        out = x_s + dx
        return out

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(x, seg)
            x_s = self.conv_last(x_s)
        else:
            x_s = x
        return x_s


class Generator(nn.Module):
    def __init__(self, output_channels=4, num_classes=7, channel=512):
        super(Generator, self).__init__()
        self.fc = nn.Linear(16, channel * 16)  # Adjust this size as necessary
        self.head = SPADEResnetBlock(channel, channel, num_classes)
        self.middle_blocks = nn.ModuleList([
            SPADEResnetBlock(channel, channel, num_classes) for _ in range(2)
        ])
        self.up_blocks = nn.ModuleList([
            SPADEResnetBlock(channel, channel // 2, num_classes),
            SPADEResnetBlock(channel // 2, channel // 4, num_classes),
            SPADEResnetBlock(channel // 4, channel // 8, num_classes),
            SPADEResnetBlock(channel // 8, channel // 16, num_classes)
        ])
        self.final_conv = nn.Conv2d(channel // 16, output_channels, kernel_size=1)

    def forward(self, segmap, noise):
        batch_size, _, height, width = segmap.size()
        x = self.fc(noise).view(batch_size, -1, 1, 1)
        x = F.interpolate(x, size=(height // 32, width // 32))

        x = self.head(x, segmap)
        for block in self.middle_blocks:
            x = block(x, segmap)

        for block in self.up_blocks:
            x = F.interpolate(x, scale_factor=2)
            x = block(x, segmap)

        x = self.final_conv(F.leaky_relu(x, 0.2))
        return torch.tanh(x)


class ShadowGen(nn.Module):
    def __init__(self):
        super(ShadowGen, self).__init__()
        self.conv1 = conv(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = conv(32, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, inputs):
        depth = inputs[:, 3:4, :, :]  # Assuming depth channel is last
        diff_vertical = depth[:, :, 1:, :] - depth[:, :, :-1, :]
        diff_horizontal = depth[:, :, :, 1:] - depth[:, :, :, :-1]

        diff_vertical = F.pad(diff_vertical, (0, 0, 1, 0), mode='replicate')
        diff_horizontal = F.pad(diff_horizontal, (1, 0, 0, 0), mode='replicate')

        h = torch.cat([diff_vertical, diff_horizontal], dim=1)
        h = F.leaky_relu(self.conv1(h))
        h = self.conv2(h)
        out = torch.tanh(inputs + h)

        return out

def lab_preprocess(lab):
    L_chan, a_chan, b_chan = lab[:, 0, :, :], lab[:, 1, :, :], lab[:, 2, :, :]
    # Normalize L channel to range [0, 1]
    L_chan = (L_chan + 1) / 2 * 100
    # a and b channels are typically in the range [-110, 110]
    a_chan = a_chan * 110
    b_chan = b_chan * 110
    return torch.stack([L_chan, a_chan, b_chan], dim=1)

def lab_postprocess(lab):
    L_chan, a_chan, b_chan = lab[:, 0, :, :], lab[:, 1, :, :], lab[:, 2, :, :]
    L_chan = L_chan / 100 * 2 - 1
    a_chan = a_chan / 110
    b_chan = b_chan / 110
    return torch.stack([L_chan, a_chan, b_chan], dim=1)

def lab_to_rgb(lab, device='cpu'):
    # Assumes lab is a torch tensor of shape [batch_size, 3, H, W]
    # and its values are in the range [0, 100] for L and [-128, 127] for a and b
    lab_np = lab.permute(0, 2, 3, 1).cpu().numpy()
    rgb_np = cspace_convert(lab_np, "CIELab", "sRGB1")
    rgb = torch.tensor(rgb_np, dtype=torch.float32, device=device).permute(0, 3, 1, 2)
    # Clip to ensure the values are valid
    rgb = torch.clamp(rgb, 0.0, 1.0)
    return rgb

def rgb_to_lab(rgb, device='cpu'):
    # Assumes rgb is a torch tensor of shape [batch_size, 3, H, W]
    # and its values are in the range [0, 1]
    rgb_np = rgb.permute(0, 2, 3, 1).cpu().numpy()
    lab_np = cspace_convert(rgb_np, "sRGB1", "CIELab")
    lab = torch.tensor(lab_np, dtype=torch.float32, device=device).permute(0, 3, 1, 2)
    return lab


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.downsample = nn.AvgPool2d(4, stride=4, padding=0, ceil_mode=False)

        # Initial convolution block
        self.init_conv = nn.Sequential(
            nn.Conv2d(11, 128, kernel_size=3, stride=2, padding=1),
            nn.SELU(inplace=True)
        )

        # Downsampling blocks
        self.down_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                nn.SELU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.SELU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                nn.SELU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.SELU(inplace=True)
            )
        ])

        # Final prediction layer
        self.final_conv = nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, segmap, image):
        # Concatenate image and segmentation map
        combined_input = torch.cat([image, segmap], dim=1)

        # Apply initial convolution
        x = self.init_conv(combined_input)

        # Apply downsampling blocks
        for block in self.down_blocks:
            x = block(x)

        # Final prediction
        return self.final_conv(x)


def discriminator_loss(real_logits, fake_logits):
    real_loss = F.relu(1.0 - real_logits).mean()
    fake_loss = F.relu(1.0 + fake_logits).mean()
    return real_loss + fake_loss

def generator_loss(fake_logits):
    return -fake_logits.mean()


class VGGLoss(nn.Module):
    def __init__(self, layer_names):
        super(VGGLoss, self).__init__()
        self.vgg = models.vgg19(pretrained=True).features
        self.layer_names = layer_names
        for param in self.vgg.parameters():
            param.requires_grad = False  # Freeze VGG parameters

    def forward(self, fake, real):
        loss = 0
        x_fake, x_real = fake, real
        for name, layer in self.vgg._modules.items():
            x_fake = layer(x_fake)
            x_real = layer(x_real)
            if name in self.layer_names:
                loss += torch.mean(torch.abs(x_fake - x_real))
        return loss

style_layers = ['2', '7', '12', '21', '30']  # Corresponds to conv layers in VGG19
vgg_loss = VGGLoss(style_layers)


generator = Generator() # Assume Generator() is defined elsewhere
shadow_generator =ShadowGen()
discriminator = Discriminator()  # Assume Discriminator() is defined elsewhere

generator_optimizer = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.999))
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=1e-3, betas=(0.5, 0.999))


# Save checkpoint
def save_checkpoint(state, filename="checkpoint.pth.tar"):
    torch.save(state, filename)

# Load checkpoint
def load_checkpoint(checkpoint_path, model, optimizer):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


def generate_images(model, test_input, tar, htar, orig_inp, name, device='cuda'):
    model.eval()  # Set model to evaluation mode

    # Convert data to PyTorch tensors and send to appropriate device
    test_input = test_input.to(device)
    tar = tar.to(device)
    htar = htar.to(device)
    orig_inp = orig_inp.to(device)

    # Generate prediction
    with torch.no_grad():
        prediction = model(test_input)

    # Post-process outputs (assuming the model outputs in a similar LAB format)
    prediction = torch.tanh(prediction)  # Apply tanh activation
    terrain = lab_to_rgb(prediction[:, :3, :, :])  # Convert LAB to RGB

    # Further processing if using a shadow generator, assuming shadow_generator is defined
    # shadow_output = shadow_generator(prediction)
    # prediction = torch.cat((terrain, shadow_output[:, 3:, :, :]), dim=1)

    # Normalize images for saving to disk
    prediction = (prediction + 1) / 2  # Normalize to [0, 1]
    tar = (tar + 1) / 2
    orig_inp = (orig_inp + 1) / 2
    htar = htar.float() / 257

    # Concatenate images for saving
    images = torch.cat([orig_inp, tar, htar, terrain, prediction], dim=3)  # Adjust dimension if needed

    # Save images
    for i in range(images.size(0)):
        save_path = os.path.join('datasets/TERR/outH2', f"{name}-{i}.png")
        save_image(images[i], save_path)

    model.train()  # Set model back to train mode if needed


def train_step(model, shadow_generator, discriminator, generator_optimizer, discriminator_optimizer, vgg_loss,
               input_image, target, device):
    model.train()
    shadow_generator.train()
    discriminator.train()

    input_image, target = input_image.to(device), target.to(device)

    # Generator and shadow generator forward
    gen_output = model(input_image)
    gen_output1 = shadow_generator(gen_output)

    # VGG perceptual loss
    loss_vgg = vgg_loss(gen_output1, target)

    # Compute discriminator outputs and losses
    real_output = discriminator(target)
    fake_output = discriminator(gen_output1.detach())  # detach to avoid training G on these labels
    disc_loss = (torch.mean((real_output - 1) ** 2) + torch.mean(fake_output ** 2)) / 2

    # Generator loss
    gen_loss = torch.mean((fake_output - 1) ** 2) + 0.001 * loss_vgg

    # Update generator
    generator_optimizer.zero_grad()
    gen_loss.backward(retain_graph=True)
    generator_optimizer.step()

    # Update discriminator
    discriminator_optimizer.zero_grad()
    disc_loss.backward()
    discriminator_optimizer.step()

    return disc_loss.item(), gen_loss.item(), loss_vgg.item()


def fit(model, shadow_generator, discriminator, train_loader, epochs, device):
    generator_optimizer = optim.Adam(list(model.parameters()) + list(shadow_generator.parameters()), lr=1e-4, betas=(0.5, 0.999))
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=1e-3, betas=(0.5, 0.999))
    vgg_loss = VGGLoss().to(device)

    for epoch in range(epochs):
        for input_image, target in train_loader:
            disc_loss, gen_loss, vgg_loss_val = train_step(model, shadow_generator, discriminator, generator_optimizer, discriminator_optimizer, vgg_loss, input_image, target, device)
            print(f'Epoch [{epoch+1}/{epochs}], Loss D: {disc_loss}, Loss G: {gen_loss}, VGG Loss: {vgg_loss_val}')

# Assume 'device' is defined (e.g., device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
# Assume 'train_loader' is defined, e.g., train_loader = DataLoader(dataset, batch_size=4, shuffle=True)



EPOCHS = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

fit(generator, shadow_generator, discriminator, train_loader, EPOCHS, device)