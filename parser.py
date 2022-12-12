import argparse
import datetime

def set_parser():
    parser = argparse.ArgumentParser('argument for ECG')
    parser.add_argument('--exper', type = str, default = 'test')
    parser.add_argument('--wandb_entity', type=str, default='hyeokjong',
                        help='Wandb ID')
    parser.add_argument('--wandb_project', type=str, default=None,
                        help='Project name')
    parser.add_argument('--short', type=str, default=None,
                        help='short name')

    parser.add_argument('--data_pickle_dir', type = str, default = './texts/train_dict.pickle')

    parser.add_argument('--opt', type = str, default = 'adam')
    parser.add_argument('--lr', type = float, default = '0.01')
    parser.add_argument('--batch_size', type = int, default = '512')
    parser.add_argument('--length', type = int, default = 3000)
    parser.add_argument('--epochs', type = int, default = 30)


    parser.add_argument('--model', type = str, help = 'LSTM, RNN, MLP', default = 'MLP')

    args = parser.parse_args()

    if args.wandb_project == None:
        args.wandb_project = f'[ECG]'
    if args.short == None:
        args.short = f'[Training][{args.model}][{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]'

    return args
