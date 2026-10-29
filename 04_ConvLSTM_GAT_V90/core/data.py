from paths import ROOT,WORK
import numpy as np

import torch

from paths import LAST as OUT

class Data:
 def __init__(self):
  self.folder=OUT/'data/unet';self.g=dict(np.load(self.folder/'graph.npz'));self.route=dict(np.load(self.folder/'routing.npz'));e=dict(np.load(self.folder/'extra.npz'));self.extra=torch.from_numpy(e['geometry']);self.branch=e['branch_target'];self.frames=torch.from_numpy(np.load(self.folder/'frames.npz')['frames'].astype(np.float32));self.x=torch.from_numpy(self.g['x']);self.pairs=torch.from_numpy(self.g['pairs']);self.attrs=torch.from_numpy(self.g['attrs']);self.phase=torch.tensor(self.g['phase'],dtype=torch.int64);self.centers=torch.tensor(self.g['centers'],dtype=torch.float32)
  self.sets={s:dict(np.load(self.folder/('local_'+s+'.npz'))) for s in ['train','val','test']};self.tensor={s:{k:torch.from_numpy(v) for k,v in d.items()} for s,d in self.sets.items()};self.same=self.g['phase'][self.g['pairs'][:,0]]==self.g['phase'][self.g['pairs'][:,1]];rs=dict(np.load(self.folder/'root_state.npz'));self.rs={k:torch.from_numpy(v) for k,v in rs.items()}
 def forward(self,model,frames=None,x=None):return model(self.frames if frames is None else frames,self.x if x is None else x,self.pairs,self.attrs,self.phase,self.centers,self.extra)
