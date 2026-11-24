"""Reward-only target events, never policy observations or official success changes."""
import math

class FruitMilestones:
    version='grasp-lift-completion-v3'
    def __init__(self,initial_height,*,lift_height=.04,persistence=3):
        if not math.isfinite(initial_height) or not lift_height>0 or persistence<1:
            raise ValueError('Invalid reward thresholds')
        self.initial_height=initial_height
        self.lift_height=lift_height
        self.persistence=persistence
        self.streak=0
        self.lift_awarded=False
        self.success_awarded=False
        self.closed=False
        self.pending=[]
        self.control=0
    def tick(self,*,height,target_grasped,closed_command,official_success,terminal=False):
        if self.closed:
            raise ValueError('Cannot accumulate rewards after a terminal state')
        if not math.isfinite(height):
            raise ValueError('Nonfinite target height')
        self.control+=1
        lifted=bool(target_grasped and closed_command and height-self.initial_height>=self.lift_height)
        self.streak=self.streak+1 if lifted else 0
        if not self.lift_awarded and self.streak>=self.persistence:
            self.lift_awarded=True
            self.pending.append(dict(event='target_grasped_sustained_lift',control=self.control,reward=.2))
        if official_success and not self.success_awarded:
            self.success_awarded=True
            self.pending.append(dict(event='official_success',control=self.control,reward=.8))
        self.closed=bool(terminal or official_success)
    def consume(self):
        events,self.pending=self.pending,[]
        return dict(reward=sum(x['reward'] for x in events),
                    official_reward=sum(x['reward'] for x in events if x['event']=='official_success'),
                    progress_reward=sum(x['reward'] for x in events if x['event']!='official_success'),
                    events=events,reward_version=self.version,
                    lift_milestone_awarded=self.lift_awarded,official_success_awarded=self.success_awarded)
