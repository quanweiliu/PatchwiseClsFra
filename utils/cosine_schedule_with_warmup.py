import math
from torch.optim.lr_scheduler import LambdaLR

def cosine_warmup_scheduler_epoch(optimizer, warmup_epochs: int, total_epochs: int, last_epoch: int = -1):
    assert total_epochs > 0
    warmup_epochs = int(warmup_epochs)

    def lr_lambda(epoch_idx: int):
        # epoch_idx: 0,1,2,... 由 scheduler.step() 递增
        if warmup_epochs > 0 and epoch_idx < warmup_epochs:
            # 让第1个epoch就有非零lr： (epoch_idx+1)/warmup_epochs
            return float(epoch_idx + 1) / float(warmup_epochs)

        # cosine decay from 1 -> 0 over remaining epochs
        if total_epochs <= warmup_epochs:
            return 1.0  # 极端情况：全部都是warmup/或warmup>=total

        progress = float(epoch_idx - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)



def cosine_warmup_scheduler_epoch_cycles(optimizer, warmup_epochs, total_epochs, num_cycles=0.5, last_epoch=-1):
    def lr_lambda(epoch_idx):
        if warmup_epochs > 0 and epoch_idx < warmup_epochs:
            return (epoch_idx + 1) / max(1, warmup_epochs)
        
        progress = (epoch_idx - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return max(0.0, 0.5 * (1.0 + math.cos(2.0 * math.pi * num_cycles * progress)))
    
    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)
