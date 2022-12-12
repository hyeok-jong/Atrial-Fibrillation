import torch

class mlp_model(torch.nn.Module):
    def __init__(self, num_blocks, sequen_size, hidden_features, classes, drop_out):
        super().__init__()
        self.layers = self.make_blocks(num_blocks, sequen_size, hidden_features, classes, drop_out)
        self.hidden_features = hidden_features
        self.classes = classes
        self.drop_out = drop_out

    def make_blocks(self, num_blocks, sequen_size, hidden_features, classes, drop_out):
        layers = [  torch.nn.Linear(in_features = sequen_size, out_features = hidden_features),
                    torch.nn.LayerNorm(hidden_features),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(p = drop_out)]
        for _ in range(num_blocks-2):
            layers += [ torch.nn.Linear(in_features = hidden_features, out_features = hidden_features),
                        torch.nn.LayerNorm(hidden_features),
                        torch.nn.ReLU(),
                        torch.nn.Dropout(p = drop_out)]
        layers += [ torch.nn.Linear(in_features = hidden_features, out_features = classes),
                    torch.nn.LayerNorm(classes)]
        return torch.nn.Sequential(*layers)

    def forward(self, input_batch):
        return self.layers(input_batch)

