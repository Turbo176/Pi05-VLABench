"""Keep AdamW updates in FP32 while the forward model keeps its original dtypes."""
import torch


class FP32MasterAdamW:
    FORMAT = 'fp32-master-adamw-v1'

    def __init__(self, parameters, **kwargs):
        self.model_parameters = list(parameters)
        self.optimizer_parameters = [
            torch.nn.Parameter(p.detach().float().clone()) if p.dtype != torch.float32 else p
            for p in self.model_parameters]
        self.master_indices = [i for i, (p, m) in enumerate(zip(self.model_parameters, self.optimizer_parameters))
                               if p is not m]
        self.inner = torch.optim.AdamW(self.optimizer_parameters, **kwargs)
        self.steps = 0

    @property
    def state(self):
        return self.inner.state

    def zero_grad(self, set_to_none=True):
        self.inner.zero_grad(set_to_none=set_to_none)
        for i in self.master_indices:
            p = self.model_parameters[i]
            if set_to_none:
                p.grad = None
            elif p.grad is not None:
                p.grad.zero_()

    @torch.no_grad()
    def step(self):
        for i in self.master_indices:
            p, m = self.model_parameters[i], self.optimizer_parameters[i]
            m.grad = None if p.grad is None else p.grad.detach().float()
        self.inner.step()
        for i in self.master_indices:
            self.model_parameters[i].copy_(self.optimizer_parameters[i])
        self.steps += 1

    def state_dict(self):
        result = self.inner.state_dict()
        result.update(master_format=self.FORMAT, master_indices=self.master_indices,
                      master_weights=[self.optimizer_parameters[i].detach().cpu().clone()
                                      for i in self.master_indices], master_steps=self.steps)
        return result

    @torch.no_grad()
    def load_state_dict(self, saved):
        if saved.get('master_format') != self.FORMAT or saved['master_indices'] != self.master_indices:
            raise ValueError('Different FP32 master optimizer profile')
        if len(saved['master_weights']) != len(self.master_indices):
            raise ValueError('Missing FP32 master parameters')
        self.inner.load_state_dict({key: saved[key] for key in ('state', 'param_groups')})
        for i, weight in zip(self.master_indices, saved['master_weights'], strict=True):
            p = self.optimizer_parameters[i]
            if p.shape != weight.shape or weight.dtype != torch.float32:
                raise ValueError('Unexpected master weight shape or precision')
            p.copy_(weight)
            self.model_parameters[i].copy_(p)
        self.steps = saved['master_steps']
