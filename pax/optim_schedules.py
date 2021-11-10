from typing import List


def step_lr(step_size: int, gamma: int = 0.1, initial_lr: float = 1.0):
    """Like torch.optim.lr_scheduler.StepLR"""

    def schedule(step: int) -> float:
        return initial_lr * gamma ** (step // step_size)

    return schedule


def multistep_lr(milestones: List[int], gamma=0.1, initial_lr=1.0):
    """Like torch.optim.lr_scheduler.MultiStepLR"""

    def schedule(step: int) -> float:
        lr = initial_lr
        for milestone in milestones:
            if step >= milestone:
                lr *= gamma
        return lr

    return schedule
