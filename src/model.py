import pytorch_lightning as pl
import torch
import torchvision
from torch import nn

from src.entropy_coder import RangeCoder


class InverseResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        self.inv_conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, output_padding=stride - 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act_fn = nn.GELU()
        self.inv_conv2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.upscale = None
        if stride >= 1 or in_channels != out_channels:
            self.upscale = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False, output_padding=stride - 1),
                nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, x):
        out = self.inv_conv1(x)
        out = self.bn1(out)
        out = self.act_fn(out)
        
        out = self.inv_conv2(out)
        out = self.bn2(out)
        
        if self.upscale is not None:
            x = self.upscale(x)
        
        out += x
        return self.act_fn(out)


class SuperInverseResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.inv_rnb1 = InverseResNetBlock(in_channels, in_channels)
        self.inv_rnb2 = InverseResNetBlock(in_channels, out_channels, stride=stride)
    
    def forward(self, x):
        out = self.inv_rnb1(x)
        out = self.inv_rnb2(out)
        return out


class InverseResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(512, 512, kernel_size=7, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(512)
        
        self.sirb1 = SuperInverseResNetBlock(512, 256, stride=2)
        self.sirb2 = SuperInverseResNetBlock(256, 128, stride=2)
        self.sirb3 = SuperInverseResNetBlock(128, 64, stride=2)
        self.sirb4 = SuperInverseResNetBlock(64, 64, stride=1)
        
        self.conv2 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False, output_padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.ConvTranspose2d(64, 3, kernel_size=7, stride=2, padding=3, bias=False, output_padding=1)
        
        self.act_fn = nn.GELU()
    
    def forward(self, x):
        x = x.view(-1, 256, 32, 32)
        # out = self.conv1(x)
        # out = self.bn1(out)
        # out = self.act_fn(out)
        
        # x = self.sirb1(x)
        x = self.sirb2(x)
        x = self.sirb3(x)
        x = self.sirb4(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act_fn(x)

        x = self.conv3(x)
        return x


class ResNet18AE(pl.LightningModule):
    def __init__(self, precision=3, learning_rate=1e-3):
        super().__init__()
        
        # log hyperparameters
        self.save_hyperparameters()
       
        self.encoder = torchvision.models.resnet18(weights='IMAGENET1K_V1')
        # for param in self.encoder.parameters():
        #     param.requires_grad = False
        
        self.encoder.fc = nn.Identity()
        self.encoder.avgpool = nn.Identity()
        self.encoder.layer4 = nn.Identity()
        
        self.decoder = InverseResNet18()
        
        self.precision = precision
        self.features_mean = torch.zeros(262144, dtype=torch.float32, device='cuda')
        self.features_std = torch.ones(262144, dtype=torch.float32, device='cuda')
        
        self.learning_rate = learning_rate
        self.loss = nn.MSELoss()
    
    def encode(self, x):
        x = self.encoder(x)
        x = torch.clamp(x, -1, 1)
        
        if self.training:
            with torch.no_grad():
                if self.on_gpu and not self.features_mean.is_cuda:
                    self.features_mean = self.features_mean.cuda()
                    self.features_std = self.features_std.cuda()
                self.features_mean = self.features_mean * 0.9 + x.mean(axis=0) * 0.1
                self.features_std = self.features_std * 0.9 + x.std(axis=0) * 0.1
        # #     x = x + (torch.rand_like(x) - 0.5)  / 10**self.precision
        else:
            rc = RangeCoder(self.features_mean.cpu(), self.features_std.cpu(), self.precision)
            x = rc.encode_batch(x.cpu())
        
        return x
    
    def encode_to_file(self, x, path):
        x = self.encoder(x)
        x = torch.clamp(x, -1, 1)
        rc = RangeCoder(self.features_mean.cpu(), self.features_std.cpu(), self.precision)
        x = rc.encode_batch(x.cpu())
        rc.put_to_binary(x[0], path)
    
    def decode_from_file(self, path):
        rc = RangeCoder(self.features_mean.cpu(), self.features_std.cpu(), self.precision)
        x = rc.get_from_binary(path)
        x = rc.decode_batch([x])
        if self.on_gpu:
            x = x.cuda()
        x = self.decoder(x)
        return x
    
    def decode(self, x):
        if not self.training:
            rc = RangeCoder(self.features_mean.cpu(), self.features_std.cpu(), self.precision)
            x = rc.decode_batch(x)
            if self.on_gpu:
                x = x.cuda()

        x = self.decoder(x)
        return x
    
    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x
    
    def training_step(self, batch, batch_idx):
        image = batch
        reconstructed_image = self(image)
        
        loss = self.loss(reconstructed_image, image)        
        self.log('train_mse_loss', loss, on_step=True, on_epoch=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        image = batch
        reconstructed_image = self(image)
        
        loss = self.loss(reconstructed_image, image)
        self.log('val_mse_loss', loss, on_step=True, on_epoch=True, logger=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        image = batch
        reconstructed_image = self(image)

        loss = self.loss(reconstructed_image, image)
        self.log('test_mse_loss', loss, on_step=True, on_epoch=True, logger=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
