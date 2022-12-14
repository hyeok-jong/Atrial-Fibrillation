import torch

class rnn_model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, fc_in_features):
        super().__init__()
        self.rnn = torch.nn.RNN(
            input_size = input_size, 
            hidden_size = hidden_size, 
            num_layers = num_layers, 
            batch_first = True,
            nonlinearity = 'tanh',
            bidirectional = False)
            
        self.classifier = torch.nn.Sequential(
            *[  torch.nn.Linear(in_features = fc_in_features, out_features = 512),
                torch.nn.LayerNorm(512),
                torch.nn.ReLU(),
                torch.nn.Dropout(p = 0.5),
                torch.nn.Linear(in_features = 512, out_features = 4),
                torch.nn.LayerNorm(4)
                ]
        )

    def forward(self, h0, input_batch):
        
        features, _ = self.rnn(input_batch.unsqueeze(dim = 2), h0)
        features = torch.flatten(features, start_dim = 1)
        logits = self.classifier(features)
        return logits