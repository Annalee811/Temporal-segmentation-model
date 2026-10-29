from paths import ROOT,WORK
"""ConvLSTM/GAT learn connections and local corrections; original U-Net stays immutable."""

import torch

from torch import nn

from backbone import TemporalGraph,decode_groups,group_state

class LocalCorrector(TemporalGraph):
 def __init__(self):
  super().__init__('convlstm_gat');self.local=nn.Sequential(nn.Linear(84,48),nn.ELU(),nn.Linear(48,3));self.edit=nn.Sequential(nn.Linear(84,32),nn.ELU(),nn.Linear(32,1));self.branch=nn.Sequential(nn.Linear(68,24),nn.ELU(),nn.Linear(24,1))
  nn.init.zeros_(self.local[-1].weight);nn.init.zeros_(self.local[-1].bias)

  for p in self.pixel.parameters():p.requires_grad_(False)
 def local_logits(self,h,node,pf):
  x=torch.cat([h[node],pf],1);anchor=pf[:,15:18];base=torch.log(.01+.97*anchor)
  log=base+5*torch.tanh(self.local(x));return log,self.edit(x)[:,0]
 def branch_logits(self,h,extra):return self.branch(torch.cat([h,extra],1))[:,0]
