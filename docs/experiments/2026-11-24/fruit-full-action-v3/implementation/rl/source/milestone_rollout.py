"""Observe physical controls for reward only; keep the policy input untouched."""
from contextlib import contextmanager
import numpy as np
from milestones import FruitMilestones


class FruitRewardObserver:
    def __init__(self, env):
        name = env.task.target_entity
        if isinstance(name, list):
            name, = name
        self.target = env.task.entities[name]
        self.tracker = FruitMilestones(float(self.target.get_xpos(env.physics)[2]))
        self.env = env

    @contextmanager
    def observe(self):
        from select_fruit import success
        original = self.env.step
        def step(action):
            timestep = original(action)
            self.tracker.tick(height=float(self.target.get_xpos(self.env.physics)[2]),
                              target_grasped=bool(self.target.is_grasped(self.env.physics, self.env.robot)),
                              closed_command=bool(np.asarray(action)[-1] < .03),
                              official_success=success(self.env), terminal=timestep.last())
            return timestep
        self.env.step = step
        try:
            yield self
        finally:
            self.env.step = original

    def finish_chunk(self, *, executed_total, solved, terminal):
        if self.tracker.control != executed_total:
            raise RuntimeError('Physical-control and PPO-chunk counts differ')
        if bool(solved) != self.tracker.success_awarded:
            raise RuntimeError('Official success and observed reward event differ')
        if terminal:
            self.tracker.closed = True
        return dict(self.tracker.consume(), terminal=bool(terminal or solved))
