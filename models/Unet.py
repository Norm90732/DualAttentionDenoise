import torch.nn as nn
from .layerModules import EncodingBlock, BottleneckBlock, DecoderBlock

#Works for both 3 color and 1 color images.
class BasicUnet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicUnet, self).__init__()
        #In channel scales
        self.Encoding1 = EncodingBlock(in_channels, 64,kernel_size=3, stride=1, padding=1)
        self.Encoding2 = EncodingBlock(64,128,kernel_size=3, stride=1, padding=1)
        self.Encoding3 = EncodingBlock(128,256,kernel_size=3, stride=1, padding=1)
        self.Encoding4 = EncodingBlock(256,512,kernel_size=3, stride=1, padding=1)
        #Bottle Neck Block
        self.Bottleneck1 = BottleneckBlock(512,1024,kernel_size=3, stride=1, padding=1)
        #Decoding Block that scales down to output channel
        self.Decoding1 = DecoderBlock(1024,512,kernel_size=2, stride=2, padding=0)
        self.Decoding2 = DecoderBlock(512,256,kernel_size=2, stride=2, padding=0)
        self.Decoding3 = DecoderBlock(256,128,kernel_size=2, stride=2, padding=0)
        self.Decoding4 = DecoderBlock(128,64,kernel_size=2, stride=2, padding=0)
        self.finalConv = nn.Conv2d(64,out_channels,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        #Encoding Block
        x,skip1 = self.Encoding1(x)
        x,skip2 = self.Encoding2(x)
        x,skip3 = self.Encoding3(x)
        x,skip4 = self.Encoding4(x)
        #Bottleneck/Latent Feature Space
        x = self.Bottleneck1(x)
        #Decoding Block
        x = self.Decoding1(x,skip4)
        x = self.Decoding2(x,skip3)
        x = self.Decoding3(x,skip2)
        x = self.Decoding4(x,skip1)
        x = self.finalConv(x)
        return x

