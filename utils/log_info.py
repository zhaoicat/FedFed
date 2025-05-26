# from torch.utils.tensorboard import SummaryWriter
import logging

# 临时禁用tensorboard功能
class DummySummaryWriter:
    def __init__(self, *args, **kwargs):
        pass
    
    def add_scalar(self, *args, **kwargs):
        pass
    
    def add_histogram(self, *args, **kwargs):
        pass
    
    def close(self):
        pass

SummaryWriter = DummySummaryWriter

# 全局writer变量
writer = None

def init_writer(log_dir=None):
    """初始化tensorboard writer"""
    global writer
    if writer is None:
        writer = SummaryWriter(log_dir=log_dir)

def log_info(type:str, name:str, info, step=None, record_tool='wandb', wandb_record=False):
    '''
    type: the info type mainly include: image, scalar (tensorboard may include hist, scalars)
    name: replace the info name displayed in wandb or tensorboard
    info: info to record
    '''
    global writer
    
    if record_tool == 'wandb':
        import wandb
    if type == 'image':
        if record_tool == 'tensorboard':
            if writer is None:
                init_writer()
            writer.add_image(name, info, step)
        if record_tool == 'wandb' and wandb_record:
            wandb.log({name: wandb.Image(info)})

    if type == 'scalar':
        if record_tool == 'tensorboard':
            if writer is None:
                init_writer()
            writer.add_scalar(name, info, step)
        if record_tool == 'wandb'and wandb_record:
            wandb.log({name:info})
    if type == 'histogram':
        if record_tool == 'tensorboard':
            if writer is None:
                init_writer()
            writer.add_histogram(name, info, step)