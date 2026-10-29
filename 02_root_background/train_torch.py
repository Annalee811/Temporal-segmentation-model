"""Fresh initialization, full epochs, gradient accumulation, recoverable state."""
import os
from pathlib import Path
import numpy as np
import torch
from models import create,objective
from engine import entry


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp')
    torch.save(value,tmp);os.replace(str(tmp),str(path))


class TorchBackend:
    suffix='.pt';resume_suffix='.pt'
    def __init__(self,args,protocol):
        assert args.model in ['unet','segformer']
        torch.set_num_threads(args.threads);torch.manual_seed(protocol['seed'])
        self.model=create(args.model)
        self.optimizer=torch.optim.AdamW(self.model.parameters(),lr=protocol['initial_lr'],weight_decay=1e-4)
        self.optimizer.zero_grad(set_to_none=True);self.accum=protocol['gradient_accumulation']
    def set_lr(self,rate):
        for group in self.optimizer.param_groups:group['lr']=rate
    def tensors(self,x,y,v):
        return torch.from_numpy(x.transpose(0,3,1,2).copy()),torch.from_numpy(y[:,None]),torch.from_numpy(v[:,None])
    def train_batch(self,x,y,v):
        self.model.train();x,y,v=self.tensors(x,y,v)
        with torch.autocast(device_type='cpu',dtype=torch.bfloat16):z=self.model(x)
        loss=objective(z,y,v);(loss/self.accum).backward();return float(loss.detach())
    def step(self):
        norm=torch.nn.utils.clip_grad_norm_(self.model.parameters(),5.)
        assert torch.isfinite(norm),'Nonfinite gradient norm'
        self.optimizer.step();self.optimizer.zero_grad(set_to_none=True)
    def evaluate(self,x,y,v):
        self.model.eval();x,y,v=self.tensors(x,y,v)
        with torch.inference_mode():
            z=self.model(x);loss=objective(z,y,v)
            return z.sigmoid()[:,0].numpy(),float(loss)
    def predict(self,x):
        self.model.eval()
        with torch.inference_mode():return self.model(torch.from_numpy(x.transpose(0,3,1,2).copy())).sigmoid()[:,0].numpy()
    def save_model(self,path):save(path,dict(state_dict=self.model.state_dict()))
    def load_model(self,path):self.model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['state_dict'])
    def save_resume(self,path,meta):
        save(path,dict(state_dict=self.model.state_dict(),optimizer=self.optimizer.state_dict(),torch_rng=torch.get_rng_state(),metadata=meta))
    def restore(self,path):
        state=torch.load(path,map_location='cpu',weights_only=False)
        self.model.load_state_dict(state['state_dict']);self.optimizer.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['torch_rng']);return state['metadata']
    def close(self):pass


if __name__=='__main__':entry(TorchBackend)
