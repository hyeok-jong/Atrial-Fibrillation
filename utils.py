import wandb
import os
import datetime
import torch

path = './dir/sub_dir/tmp1'

os.makedirs(path, exist_ok=True)




def init_wandb(args):
    wandb.init(
        entity=args.wandb_entity, 
        project=args.wandb_project,
        name=args.short,
        config=args,
    )
    wandb.run.save()
    return wandb.config


def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

def batch_acc(output, target):
    correct = 0
    batch_size = output.shape[0]
    for i in range(batch_size):
        if torch.argmax(output[i]).item() == target[i].item():
            correct += 1
    return correct/batch_size

def save_model(model, optimizer, args, epoch, save_path):
    print(f'==> Saving {save_path}...')
    state = {
        'args': args,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_path)
    del state