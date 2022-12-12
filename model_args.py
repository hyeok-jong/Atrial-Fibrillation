import torch

from models import *



def create_model(args):
    if args.model == 'MLP':
        model = mlp_model(
                            num_blocks = 8, 
                            sequen_size = args.length, 
                            hidden_features = 1024, 
                            classes = 4, 
                            drop_out = 0.5
                            ).cuda()
        c0 = h0 = None
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
        c0 = None
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
    return model, h0, c0