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
    train_ratio = 0.9
    train_len = int(len(ecg_dataset)*train_ratio)
    valid_len = len(ecg_dataset) - train_len
    train_dataset, valid_dataset = torch.utils.data.random_split(ecg_dataset, [train_len, valid_len])


    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True,
                                        num_workers = 20, pin_memory = True, drop_last = True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size = args.batch_size, shuffle = True,
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
    elif args.model == 'RNN':
        input_size = 1
        hidden_size = 40
        num_layers = 2
        batch_size = args.batch_size
        h0 = torch.zeros(num_layers, batch_size, hidden_size).cuda()
        model = rnn_model(  input_size = input_size,
                            hidden_size = hidden_size, 
                            num_layers = num_layers, 
                            fc_in_features = hidden_size*args.length).cuda()
    elif args.model == 'LSTM':
        input_size = 1
        hidden_size = 40
        num_layers = 2
        batch_size = args.batch_size
        h0 = torch.zeros(num_layers, batch_size, hidden_size).cuda()
        c0 = torch.zeros(num_layers, batch_size, hidden_size).cuda()
        model = lstm_model(  input_size = input_size,
                            hidden_size = hidden_size, 
                            num_layers = num_layers, 
                            fc_in_features = hidden_size*args.length).cuda()

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
        model.train()

        for data in train_loader:
            '''for 1-batch training'''
            input = data[0].cuda()
            target = data[1].cuda().squeeze().long()

            if args.model == 'MLP':
                output = model(input)
            elif args.model in ['RNN']:
                output = model(h0, input)
            elif args.model in ['LSTM']:
                output = model(h0, c0, input)

            loss_batch = loss(output, target)
            acc_batch = batch_acc(output, target)

            optimizer.zero_grad()
            loss_batch.backward()
            optimizer.step()
            one_epoch_loss.append(loss_batch.item())
            one_epoch_acc.append(acc_batch)
        result = {
        'train loss': sum(one_epoch_loss) / len(one_epoch_loss),
        'train accuracy': sum(one_epoch_acc) / len(one_epoch_acc)
            }
        wandb.log(result, step = epoch)

        if (epoch%6 == 0) or (epoch == args.epochs):
            save_model(model, optimizer, args, epoch, save_path + f'/{epoch}.pt')
        
        '''for 1-batch validation'''
        model.eval()
        one_epoch_loss = []
        one_epoch_acc = []
        with torch.no_grad():
            for data in valid_loader:
                input = data[0].cuda()
                target = data[1].cuda().squeeze().long()
                if args.model == 'MLP':
                    output = model(input)
                elif args.model in ['RNN']:
                    output = model(h0, input)
                elif args.model in ['LSTM']:
                    output = model(h0, c0, input)
                loss_batch = loss(output, target)
                acc_batch = batch_acc(output, target)

                one_epoch_loss.append(loss_batch.item())
                one_epoch_acc.append(acc_batch)
            result = {
            'valid loss': sum(one_epoch_loss) / len(one_epoch_loss),
            'valid accuracy': sum(one_epoch_acc) / len(one_epoch_acc)
                }
            wandb.log(result, step = epoch)


    wandb.finish()
            