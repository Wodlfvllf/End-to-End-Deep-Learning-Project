from utils_c1 import *

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        
        x = self.conv(x)
        
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16):
        super(ChannelGate, self).__init__()
        
        # Number of input channels to image
        self.gate_channels = gate_channels
        
        #MLP layer
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )

    def forward(self, x):
        
        #Avg_pool
        avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        channel_att_avg = self.mlp( avg_pool )

        #max_pool
        max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        channel_att_max = self.mlp( max_pool )

        #Element wise sum
        channel_att_sum = channel_att_max + channel_att_avg

        #scaling output of channel attention to match dimensions with input
        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        
        return x * scale


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
        
    def forward(self, x):
        
        # Applying Average Pooling and Maxpooling layer and concatenating
        x_compress = self.compress(x)
        
        # Applying Convolution operation on concatenated inputs
        x_out = self.spatial(x_compress)
        
        # Applying Sigmoid to attention mask
        scale = F.sigmoid(x_out) # broadcasting
        
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=1, no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio)
        self.no_spatial=no_spatial
        
        if not no_spatial:
            self.SpatialGate = SpatialGate()
            
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int,out_channels: int,stride: int = 1,expansion: int = 1,downsample: nn.Module = None) -> None:
        super(BasicBlock, self).__init__()
        self.expansion = expansion
        self.downsample = downsample
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=3, padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels*self.expansion)
        
    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return  out
    
class ResNet(nn.Module):
    def __init__(
        self, 
        img_channels: int,
        num_layers: int,
        block: Type[BasicBlock],
        cbam: Type[BasicBlock],
        num_classes: int  = 2
    ) -> None:
        super(ResNet, self).__init__()
        if num_layers == 15:
            # The following `layers` list defines the number of `BasicBlock` 
            # to use to build the network and how many basic blocks to stack
            # together.
            layers = [2, 2, 2, 2]
        self.expansion = 1
        self.in_channels = 64
        
        # All ResNets (18 to 152) contain a Conv2d => BN => ReLU for the first
        # three layers. Here, kernel size is 7.
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=self.in_channels,
            kernel_size=7, 
            stride=2,
            padding=3,
            bias=False
        )
        self.conv2 = nn.Conv2d(
            in_channels=256,
            out_channels=64,
            kernel_size=3, 
            stride=2,
            padding=3,
            bias=False
        )
        
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.cbam_1 = self.channel_block_attention(cbam)
        
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.cbam_2 = self.channel_block_attention(cbam)
        
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.cbam_3 = self.channel_block_attention(cbam)
        
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64*self.expansion, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p = 0.2)
        
    def _make_layer(
        self, 
        block: Type[BasicBlock],
        out_channels: int,
        blocks: int,
        stride: int = 1
    ) -> nn.Sequential:
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels, 
                    out_channels*self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False 
                ),
                nn.BatchNorm2d(out_channels * self.expansion),
            )
        layers = []
        layers.append(
            block(
                self.in_channels, out_channels, stride, self.expansion, downsample
            )
        )
        self.in_channels = out_channels * self.expansion
        for i in range(1, blocks):
            layers.append(block(
                self.in_channels,
                out_channels,
                expansion=self.expansion
            ))
        return nn.Sequential(*layers)
    
    def channel_block_attention(self, cbam):
        return cbam(self.in_channels)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.dropout(x)
        x = self.relu(x)
        
    
        x = self.layer2(x)
        x = self.dropout(x)
        
        x = self.layer3(x)
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x
    
class Resnet_Ensemble(nn.Module):
    def __init__(self,img_channels:int,
                 num_layers: int,
                 block: Type[BasicBlock],
                 cbam: Type[CBAM],
                 resnet,
                 num_classes: int  = 2):
        
        super(Resnet_Ensemble, self).__init__()
        self.img_channels = img_channels
        self.num_layers = num_layers
        self.expansion = 1
        self.resnet_time = resnet(img_channels, num_layers, block, cbam)
        self.resnet_energy = resnet(img_channels, num_layers, block, cbam)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64*self.expansion, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p = 0.2)
        self.relu = nn.ReLU()
    
    def __make_layer(self, block):
        return 
    def forward(self, x):
        x_t = x[:,1,:,:]
        # print(x_t.shape)
        x_t = x_t.reshape((x_t.shape[0],1,x_t.shape[2], x_t.shape[2]))
        x_e = x[:,0,:,:]
        x_e = x_e.reshape((x_e.shape[0],1,x_e.shape[2], x_e.shape[2]))
        
        op_T = self.resnet_time(x_t)
        op_E = self.resnet_energy(x_e)
        
        op =  op_T+op_E
        op = self.relu(op)
        op = self.fc(op)
        op = self.sigmoid(op)
        return op
            