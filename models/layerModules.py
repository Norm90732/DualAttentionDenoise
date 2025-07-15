import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvolutionalBlock(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size, stride, padding,transpose,act=True):
        super(ConvolutionalBlock, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.transpose = transpose
        if transpose == True:
            layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride),]
        else:
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride),]
        if act== True:
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)

class EncodingBlock(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size, stride, padding):
        super(EncodingBlock, self).__init__()
        #Encoder Block
        self.convBlock1 = ConvolutionalBlock(in_channels, out_channels,kernel_size, stride, padding,transpose=False,act=True)
        self.convBlock2 = ConvolutionalBlock(out_channels, out_channels,kernel_size, stride, padding,transpose=False,act=True)
        self.p1 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.convBlock1(x)
        x = self.convBlock2(x)
        skip = x
        x = self.p1(x)
        return x, skip


class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size, stride, padding):
        super(BottleneckBlock, self).__init__()
        self.convBlock1 = ConvolutionalBlock(in_channels, out_channels,kernel_size, stride, padding,transpose=False,act=True)
        self.convBlock2 = ConvolutionalBlock(out_channels, out_channels,kernel_size, stride, padding,transpose=False,act=True)

    def forward(self, x):
        x = self.convBlock1(x)
        x = self.convBlock2(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size, stride, padding):
        super(DecoderBlock, self).__init__()
        self.convBlockup1 = ConvolutionalBlock(in_channels, out_channels, kernel_size, stride, padding, transpose=True)
        self.convBlock2 = ConvolutionalBlock(out_channels*2, out_channels, 3, 1, 1, transpose=False)
        self.convBlock3 = ConvolutionalBlock(out_channels, out_channels, 3, 1, 1, transpose=False)

    def forward(self,x,skip):
        x = self.convBlockup1(x)
        x = torch.cat([x,skip],dim=1)
        x = self.convBlock2(x)
        x = self.convBlock3(x)
        return x


##Creating Attention Spatial Attention Block
class SpatialAttentionBlock(nn.Module):
    def __init__(self,kernel_size, stride, padding):
        super(SpatialAttentionBlock, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        #Maxpool and AVG Pool, Concat, then Convolution, Then sigmoid to compress feature map to [0,1]
        #Has to be done channel pooling not spatial pooling
        self.conv1 = nn.Conv2d(2, 1, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        avgChannelPool = torch.mean(x,dim=1,keepdim=True)
        maxChannelPool = torch.max(x,dim=1,keepdim=True)[0]
        concat = torch.cat([maxChannelPool,avgChannelPool],dim=1)
        attention = self.conv1(concat)
        attention = self.sigmoid(attention)
        #hadamard product between attention and feature
        result = x * attention
        return result


class ChannelAttentionBlock(nn.Module):
    def __init__(self, in_channel,kernel_size, stride, padding,reduction1):
        super(ChannelAttentionBlock, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.in_channel = in_channel
        self.reduction1 = reduction1
        reductionFactor = int((2*in_channel/reduction1))
        self.linear1 = nn.Linear(2*in_channel,reductionFactor) #Concatinated Channels (2C) is compressed to 2C/reducation (some reducation factor is used to compress the neurons on the output)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(reductionFactor,in_channel)
        self.sigmoid = nn.Sigmoid()


    def forward(self,x):
        #Global pooling to get Chanel,1,1, Then Passed into Two linear layers, then scale attention map with hadamard product
        maxPoolGlobal = F.adaptive_max_pool2d(x,(1,1))
        avgPoolGlobal = F.adaptive_avg_pool2d(x,(1,1))
        concat = torch.cat([maxPoolGlobal,avgPoolGlobal],dim=1)
        concat = torch.flatten(concat,1)
        attention = self.linear1(concat)
        attention = self.relu(attention)
        attention = self.linear2(attention)
        attention = torch.reshape(attention,[x.shape[0],x.shape[1],1,1])
        attention = self.sigmoid(attention)
        result = x * attention
        return result


#Creating DualAttention Block (DAnet)
class DualAttentionBlock(nn.Module):
    def __init__(self,in_channel,reduction1,kernel_size, stride, padding,sumOrElement:bool):
        super(DualAttentionBlock, self).__init__()
        self.channelAttention = ChannelAttentionBlock(in_channel, kernel_size, stride, padding, reduction1)
        self.spatialAttention = SpatialAttentionBlock(kernel_size, stride, padding)
        self.sumOrElement = sumOrElement

    def forward(self,x):
        #channelAttention then Spatial Attention to create a dual attention block on one feature map input
        attentionChannel = self.channelAttention(x)
        attentionSpatial = self.spatialAttention(x)
        if (self.sumOrElement == True):
            sumElement = attentionChannel + attentionSpatial
        else:
            sumElement = attentionChannel * attentionSpatial

        return x + sumElement #residual dual attention block
    #this works by determining the most important features in the channel space, then in the spatial space, then since the dimensions
    #are the same for attention, you can add them together and the original feature map will highlight which values are more important.
    #This way it becomes useful for using on skip connections for concatenating the abstracted feature maps for reconstruction in the decoder







