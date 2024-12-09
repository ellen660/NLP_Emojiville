import torch.nn as nn

class LinearProbe(nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super(LinearProbe, self).__init__()
        self.linear = nn.Linear(embedding_dim, num_classes)
    
    def forward(self, x):
        return self.linear(x)