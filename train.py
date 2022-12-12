import torch
import wandb
import os
from tqdm import tqdm


from models import *
from ecg_data import ECG_dataset
from parser import set_parser
from utils import init_wandb, batch_acc, save_model






if __name__ == '__main__':

    args = set_parser()
    init_wandb(args)
    save_path = f'./{args.exper}'
    os.makedirs(save_path, exist_ok=True)

    # set loader
    ecg_dataset = ECG_dataset(length = args.length, pickle_dir = args.data_pickle_dir)
    ecg_loader = torch.utils.data.DataLoader(ecg_dataset, batch_size = args.batch_size, shuffle = True,
                                        num_workers = 20, pin_memory = True, drop_last = True)

    # set model
    if args.model == 'MLP':
        model = mlp_model(
                            num_blocks = 8, 
                            sequen_size = args.length, 
                            hidden_features = 1024, 
                            classes = 4, 
                            drop_out = 0.5
                            ).cuda()
    # set optimizer and loss
    if args.opt == 'adam':
        loss = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    
    loss_list = list()
    acc_list = list()

    for epoch in tqdm(range(1, args.epochs + 1)):
        '''for 1-epoch training'''
        one_epoch_loss = []
        one_epoch_acc = []

        for data in ecg_loader:
            '''for 1-batch training'''
            input = data[0].cuda()
            target = data[1].cuda().squeeze().long()

            output = model(input)

            loss_batch = loss(output, target)
            acc_batch = batch_acc(output, target)

            optimizer.zero_grad()
            loss_batch.backward()
            optimizer.step()
            one_epoch_loss.append(loss_batch.item())
            one_epoch_acc.append(acc_batch)
        result = {
        'loss': sum(one_epoch_loss) / len(one_epoch_loss),
        'accuracy': sum(one_epoch_acc) / len(one_epoch_acc)
            }
        wandb.log(result, step = epoch)

        if (epoch%6 == 0) or (epoch == args.epochs):
            save_model(model, optimizer, args, epoch, save_path + f'/{epoch}.pt')
    wandb.finish()
            