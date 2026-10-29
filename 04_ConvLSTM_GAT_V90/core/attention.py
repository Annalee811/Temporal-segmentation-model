from paths import ROOT,WORK
"""Relation-gated GAT; class boundaries and root identity are distinct predictions."""

import torch

from torch import nn

import torch.nn.functional as F

class Layer(nn.Module):
    def __init__(self,dim=64,heads=4):
        super().__init__();self.heads=heads;self.d=dim//heads
        self.project=nn.Linear(dim,dim,bias=False);self.source=nn.Parameter(torch.empty(heads,self.d));self.destination=nn.Parameter(torch.empty(heads,self.d));self.edge=nn.Linear(8,heads,bias=False);self.out=nn.Linear(dim,dim);self.norm=nn.LayerNorm(dim)
        nn.init.xavier_uniform_(self.source);nn.init.xavier_uniform_(self.destination)
    def forward(self,h,edge_index,attrs,gates):
        a,b=edge_index;v=self.project(h).reshape(len(h),self.heads,self.d)
        score=F.leaky_relu((v[a]*self.source).sum(-1)+(v[b]*self.destination).sum(-1)+self.edge(attrs),.2)+torch.log(gates.clamp_min(.03))[:,None]
        index=b[:,None].expand(-1,self.heads);mx=torch.full((len(h),self.heads),-torch.inf,device=h.device);mx.scatter_reduce_(0,index,score.detach(),reduce='amax',include_self=True)
        weight=(score-mx[b]).exp();den=torch.zeros_like(mx).scatter_add_(0,index,weight);weight=weight/den[b].clamp_min(1e-8)
        agg=torch.zeros_like(v);agg.index_add_(0,b,F.dropout(weight,.05,self.training)[:,:,None]*v[a])
        return self.norm(h+F.elu(self.out(agg.reshape(len(h),-1))))
