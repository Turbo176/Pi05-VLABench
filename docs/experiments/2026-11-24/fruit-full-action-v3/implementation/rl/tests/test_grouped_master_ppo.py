import copy
import numpy as np
import torch
import test_fruit_ppo as fixtures
from train_fruit_ppo import optimize,parameter_groups,optimizer_cpu_snapshot
from fp32_master_adamw import FP32MasterAdamW

class GroupedMasterTests(fixtures.OptimizerTests):
    def setUp(self):
        super().setUp()
        self.model.action_head.to(dtype=torch.bfloat16)
        head=self.model.action_head
        head.forward=lambda features:torch.nn.functional.linear(features,head.weight.float(),head.bias.float())
        actor,critic=parameter_groups(self.model)
        self.optimizer=FP32MasterAdamW([dict(params=actor,lr=.001),dict(params=critic,lr=.01)],weight_decay=0)
        for transition in self.transitions:
            with torch.no_grad():logp,value=self.model.replay(transition)
            transition.update(old_logp=float(logp),value=float(value))

    def test_warmup_preserves_actor_master_weights_and_has_no_actor_moments(self):
        actor,critic=parameter_groups(self.model)
        masters=[(p,m) for p,m in zip(self.optimizer.model_parameters,self.optimizer.optimizer_parameters) if any(p is a for a in actor)]
        before=[m.detach().clone() for p,m in masters]
        optimize(self.model,self.optimizer,self.transitions,self.model.replay,np.random.default_rng(1),critic_only=True)
        for (p,m),expected in zip(masters,before):
            self.assertTrue(torch.equal(m,expected))
            self.assertNotIn(m,self.optimizer.state)

    def test_post_kl_rollback_restores_master_residuals_and_step_counters(self):
        optimize(self.model,self.optimizer,self.transitions,self.model.replay,np.random.default_rng(1),critic_only=True)
        self.optimizer.param_groups[0]['lr']=1.
        before=optimizer_cpu_snapshot(self.optimizer)
        weights=self.snapshot()
        history=optimize(self.model,self.optimizer,self.transitions,self.model.replay,np.random.default_rng(1),minibatch=8)
        self.assertTrue(history[0]['poststep_reverted_for_kl'])
        restored=self.optimizer.state_dict()
        self.assertEqual(restored['master_steps'],before['master_steps'])
        for x,y in zip(restored['master_weights'],before['master_weights']):torch.testing.assert_close(x,y,atol=0,rtol=0)
        for name,value in self.model.named_parameters():torch.testing.assert_close(value,weights[name],atol=0,rtol=0)
        for index,fields in before['state'].items():
            for key,value in fields.items():torch.testing.assert_close(restored['state'][index][key],value,atol=0,rtol=0)
