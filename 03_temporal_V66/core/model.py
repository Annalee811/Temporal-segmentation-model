"""Shared native-resolution temporal head for each of the three frozen root models."""
import torch
from torch import nn
from torch.nn import functional as F

class Block(nn.Sequential):
    def __init__(self,a,b):
        super().__init__(nn.Conv2d(a,b,3,padding=1,bias=False),nn.GroupNorm(4,b),nn.SiLU(),nn.Conv2d(b,b,3,padding=1,bias=False),nn.GroupNorm(4,b),nn.SiLU())
class TemporalHead(nn.Module):
    def __init__(self):
        super().__init__();self.a=Block(25,8);self.b=Block(8,16);self.c=Block(16,32);self.middle=Block(32,48)
        self.u2=Block(80,32);self.u1=Block(48,16);self.u0=Block(24,8);self.out=nn.Conv2d(8,3,1);self.boundary=nn.Conv2d(8,1,1)
        nn.init.zeros_(self.boundary.weight);nn.init.zeros_(self.boundary.bias)
    def representation(self,x):
        a=self.a(x);b=self.b(F.avg_pool2d(a,2));c=self.c(F.avg_pool2d(b,2));m=self.middle(F.avg_pool2d(c,2))
        def up(v,s):return torch.cat([F.interpolate(v,size=s.shape[-2:],mode='bilinear',align_corners=False),s],1)
        return self.u0(up(self.u1(up(self.u2(up(m,c)),b)),a))
    def forward(self,x):return self.out(self.representation(x))
    def forward_with_boundary(self,x):
        v=self.representation(x);return self.out(v),self.boundary(v)[:,0]



def loss(logits,gt,weights,boundary_logit=None,junction=None):
    valid=(gt>0)&(gt<255)
    if not valid.any():return logits.sum()*0
    labels=gt.long()-1;labels[~valid]=255
    pixel=F.cross_entropy(logits,labels,weight=weights,ignore_index=255,reduction='none')
    factors=(1+2*junction.float()) if junction is not None else torch.ones_like(pixel)
    ce=(pixel*factors).sum()/(weights[labels.clamp(0,2)]*valid*factors).sum().clamp_min(1)
    prob=logits.softmax(1);d=[]
    for k in [1,2,3]:
        g=(gt==k).float();p=prob[:,k-1]*valid
        if g.sum()>0:d.append(1-(2*(p*g).sum()+1)/(p.sum()+g.sum()+1))
    extra=logits.sum()*0
    if boundary_logit is not None:
        onehot=torch.stack([(gt==k).float() for k in [1,2,3]],1);near=F.max_pool2d(onehot,5,1,2)
        target=((onehot*(near.sum(1,keepdim=True)-near)).sum(1)>0)&valid
        bce=F.binary_cross_entropy_with_logits(boundary_logit,target.float(),reduction='none')
        extra=.2*(bce*valid*(1+3*target)).sum()/(valid*(1+3*target)).sum().clamp_min(1)
    smooth=[]
    for dy,dx in [(0,1),(1,0),(1,1)]:
        a=(slice(None),slice(0,-dy) if dy else slice(None),slice(0,-dx) if dx else slice(None));b=(slice(None),slice(dy,None),slice(dx,None))
        same=(gt[a]==gt[b])&valid[a]&valid[b]
        diff=(prob[(slice(None),slice(None),a[1],a[2])]-prob[(slice(None),slice(None),b[1],b[2])]).square().sum(1)
        smooth.append((diff*same).sum()/same.sum().clamp_min(1))
    from coherence import path_loss
    return ce+.5*torch.stack(d).mean()+extra+.05*torch.stack(smooth).mean()+.3*path_loss(logits,gt)
