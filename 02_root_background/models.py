"""Same U-Net / SegFormer architectures as random_tiles_v1, for fine-tuning."""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18
from transformers import SegformerConfig, SegformerForSemanticSegmentation

class Conv(nn.Sequential):
    def __init__(self,a,b):
        super().__init__(nn.Conv2d(a,b,3,padding=1,bias=False),nn.GroupNorm(4,b),nn.ReLU(),
                         nn.Conv2d(b,b,3,padding=1,bias=False),nn.GroupNorm(4,b),nn.ReLU())

class RootUNet(nn.Module):
    """ResNet18 U-Net, same native RGB skip as v9; fresh encoder and decoder."""
    def __init__(self):
        super().__init__(); self.encoder=resnet18(weights=None); self.encoder.fc=nn.Identity()
        self.d3=Conv(768,128); self.d2=Conv(256,64); self.d1=Conv(128,32); self.d0=Conv(96,32)
        self.native=Conv(3,16); self.finish=Conv(48,16); self.head=nn.Conv2d(16,1,1)
        nn.init.constant_(self.head.bias,-3.)
    def forward(self,x):
        e=self.encoder; z=(x-.5)/.25
        s0=e.relu(e.bn1(e.conv1(z))); s1=e.layer1(e.maxpool(s0)); s2=e.layer2(s1); s3=e.layer3(s2); s4=e.layer4(s3)
        def up(a,b): return torch.cat([F.interpolate(a,size=b.shape[-2:],mode='bilinear',align_corners=False),b],1)
        z=self.d3(up(s4,s3)); z=self.d2(up(z,s2)); z=self.d1(up(z,s1)); z=self.d0(up(z,s0))
        return self.head(self.finish(up(z,self.native(x))))

class RootSegFormer(nn.Module):
    """Full MiT-B0 with a native-pixel RGB residual head, randomly initialized."""
    def __init__(self):
        super().__init__()
        conf=SegformerConfig(num_channels=3,num_labels=1,hidden_sizes=[32,64,160,256],depths=[2,2,2,2],
            num_attention_heads=[1,2,5,8],sr_ratios=[8,4,2,1],decoder_hidden_size=256)
        self.net=SegformerForSemanticSegmentation(conf)
        self.refine=nn.Sequential(nn.Conv2d(4,16,3,padding=1,bias=False),nn.GroupNorm(4,16),nn.ReLU(),
            nn.Conv2d(16,16,3,padding=1,bias=False),nn.GroupNorm(4,16),nn.ReLU(),nn.Conv2d(16,1,1))
        nn.init.zeros_(self.refine[-1].weight); nn.init.zeros_(self.refine[-1].bias)
    def forward(self,x):
        logit=F.interpolate(self.net(pixel_values=(x-.5)/.25).logits,size=x.shape[-2:],mode='bilinear',align_corners=False)
        return logit+self.refine(torch.cat([x,logit],1))

def create(name): return RootUNet() if name=='unet' else RootSegFormer()

def erode(x): return torch.minimum(-F.max_pool2d(-x,(3,1),1,(1,0)),-F.max_pool2d(-x,(1,3),1,(0,1)))

def skeleton(x):
    s=F.relu(x-F.max_pool2d(erode(x),3,1,1))
    for _ in range(5):
        x=erode(x); d=F.relu(x-F.max_pool2d(erode(x),3,1,1)); s=s+F.relu(d-s*d)
    return s

def objective(logit,y,v):
    logit=logit.float(); b=F.binary_cross_entropy_with_logits(logit,y,reduction='none')
    pos=b[(y>.5)&(v>.5)]; neg=b[(y<.5)&(v>.5)]
    k=min(neg.numel(),max(1,4*pos.numel()+256*len(y))); hard=neg.topk(k).values
    bce=(pos.sum()+hard.sum())/max(1,pos.numel()+hard.numel())
    p=logit.sigmoid()*v; g=y*v; dice=1-(2*(p*g).sum()+1)/(p.sum()+g.sum()+1)
    sp=skeleton(p); sg=skeleton(g); pr=((sp*g).sum()+1)/(sp.sum()+1); re=((sg*p).sum()+1)/(sg.sum()+1)
    return bce+dice+.15*(1-2*pr*re/(pr+re+1e-7))
