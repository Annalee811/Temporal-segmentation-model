from paths import ROOT,WORK
"""Causal 2D ConvLSTM + causal GAT; identity and root-period state are learned."""

import numpy as np

import torch

from torch import nn

from torch.nn import functional as F

from attention import Layer

class ConvLSTMCell(nn.Module):
    def __init__(self,channels=8):
        super().__init__();self.channels=channels;self.gates=nn.Conv2d(channels*2,channels*4,3,padding=1)
    def forward(self,x,h,c):
        i,f,o,g=self.gates(torch.cat([x,h],1)).chunk(4,1)
        c=torch.sigmoid(f+1)*c+torch.sigmoid(i)*torch.tanh(g)
        return torch.sigmoid(o)*torch.tanh(c),c

class TemporalGraph(nn.Module):
    def __init__(self,variant):
        super().__init__();self.variant=variant
        self.frame_encoder=nn.Sequential(nn.Conv2d(8,8,3,padding=1),nn.GroupNorm(4,8),nn.ELU()) if variant=='convlstm_gat' else None
        self.memory=ConvLSTMCell() if variant=='convlstm_gat' else None
        self.encoder=nn.Sequential(nn.Linear(40,64),nn.LayerNorm(64),nn.ELU(),nn.Linear(64,64),nn.ELU())
        self.layers=nn.ModuleList([Layer(),Layer()]) if variant=='convlstm_gat' else nn.ModuleList()
        self.identity=nn.Sequential(nn.Linear(64,48),nn.ELU())
        self.connection=nn.Sequential(nn.Linear(104,64),nn.ELU(),nn.Linear(64,1))
        self.pixel=nn.Sequential(nn.Linear(79,48),nn.ELU(),nn.Linear(48,3))
        self.root_state=nn.Linear(64,1)
        nn.init.zeros_(self.pixel[-1].weight);nn.init.zeros_(self.pixel[-1].bias)
        nn.init.zeros_(self.root_state.weight);nn.init.zeros_(self.root_state.bias)
    def forward(self,frames,x,pairs,attrs,phase,centers,extra):
        context=x.new_zeros((len(x),8));maps=[]
        if self.memory is not None:
            amp=frames.device.type=='cpu' and torch.cpu._is_avx512_bf16_supported()
            with torch.autocast(device_type=frames.device.type,dtype=torch.bfloat16,enabled=amp):
                latent=self.frame_encoder(frames);h=torch.zeros_like(latent[:1]);c=h.clone()
                for t in range(4):
                    h,c=self.memory(latent[t:t+1],h,c);maps.append(h.float())
            pieces=[]
            for t in range(4):
                ids=torch.nonzero(phase==t).flatten();grid=torch.stack([centers[ids,0]/2540*2-1,centers[ids,1]/1107*2-1],1)[None,None]
                v=F.grid_sample(maps[t],grid,mode='bilinear',align_corners=True)[0,:,0].T;pieces.append((ids,v))
            context=context.index_copy(0,torch.cat([p[0] for p in pieces]),torch.cat([p[1] for p in pieces]))
        h=self.encoder(torch.cat([x,context,extra],1));a,b=pairs.T;same=phase[a]==phase[b]

        n=torch.arange(len(x),device=x.device);index=torch.cat([pairs.T,pairs[same].flip(1).T,torch.stack([n,n])],1)
        reverse=attrs[same].clone();reverse[:,1:3]*=-1
        edgeattrs=torch.cat([attrs,reverse,attrs.new_zeros((len(x),8))])
        prior=.1+.9*torch.exp(-attrs[:,0].square()/2)*(.25+.75*attrs[:,3].square());prior=torch.where(attrs[:,5]>.5,torch.ones_like(prior),prior)
        gates=torch.cat([prior,prior[same],prior.new_ones(len(x))])
        for layer in self.layers:

            ident=self.identity(h);sym0=attrs.clone();sym0[:,1:3]=sym0[:,1:3].abs()
            affinity=self.connection(torch.cat([(ident[a]-ident[b]).abs(),ident[a]*ident[b],sym0],1))[:,0].sigmoid()
            affinity=torch.where(attrs[:,5]>.5,torch.ones_like(affinity),.05+.95*affinity)
            learned_gates=gates*torch.cat([affinity,affinity[same],affinity.new_ones(len(x))])
            h=layer(h,index,edgeattrs,learned_gates)
        z=self.identity(h);sym=attrs.clone();sym[:,1:3]=sym[:,1:3].abs()
        connection=self.connection(torch.cat([(z[a]-z[b]).abs(),z[a]*z[b],sym],1))[:,0]
        return h,connection,F.normalize(z,dim=1)
    def pixel_logits(self,h,node,pf):return pf[:,:3]+3*torch.tanh(self.pixel(torch.cat([h[node],pf],1)))

def pool_groups(h,nodes,ptr):
    lengths=ptr[1:]-ptr[:-1];index=torch.repeat_interleave(torch.arange(len(lengths),device=h.device),lengths)
    result=h.new_zeros((len(lengths),h.shape[1]));result.index_add_(0,index,h[nodes]);return result/lengths.clamp_min(1)[:,None]

def decode_groups(pairs,attrs,phase,connection,threshold=.5,routing=None):
    if routing is None:raise ValueError('Endpoint routing metadata required; unrestricted KNN union is disabled')
    native=routing['native_group'];junction=routing['is_junction'];count=int(native.max())+1;parent=np.arange(count)
    def find(a):
        while parent[a]!=a:parent[a]=parent[parent[a]];a=parent[a]
        return a
    ea,eb=routing['bridge_a'],routing['bridge_b'];edge=routing['bridge_pair_index'];p=connection[edge]
    logit=lambda v:np.log(np.clip(v,1e-6,1-1e-6)/(1-np.clip(v,1e-6,1-1e-6)))
    weight=logit(p)-logit(threshold)+.75*routing['bridge_alignment']+.25*routing['bridge_facing']-.01*routing['bridge_distance']
    used=set()
    for i in np.argsort(-weight,kind='stable'):
        if p[i]<threshold or weight[i]<=0:continue
        a,b=int(ea[i]),int(eb[i])
        if a in used or b in used:continue
        aa,bb=find(int(routing['end_chain'][a])),find(int(routing['end_chain'][b]))
        if aa==bb:continue
        parent[bb]=aa;used.add(a);used.add(b)
    group=np.array([find(int(g)) for g in native])

    aa,bb=pairs.T;same=phase[aa]==phase[bb];candidates={}
    for i in np.flatnonzero(same&(junction[aa]^junction[bb])):
        j=int(aa[i] if junction[aa[i]] else bb[i]);other=int(bb[i] if junction[aa[i]] else aa[i]);value=float(connection[i]);old=candidates.get(j)
        if old is None or value>old[0]:candidates[j]=(value,other)
    for j,(value,other) in candidates.items():
        if value>=.05:group[j]=group[other]
    _,group=np.unique(group,return_inverse=True)
    return group

def group_state(model,h,group):
    idx=torch.from_numpy(group).to(h.device);n=int(group.max())+1
    values=h.new_zeros((n,h.shape[1]));values.index_add_(0,idx,h);counts=torch.bincount(idx,minlength=n)
    return model.root_state(values/counts.clamp_min(1)[:,None])[:,0]

def hierarchical_prediction(logits,node,group,new_probability,new_threshold=.5):

    roots=group[node];new=new_probability[roots]>=new_threshold
    established=np.where(logits[:,2]>logits[:,0],2,0)
    return np.where(new,1,established).astype(np.int64)
