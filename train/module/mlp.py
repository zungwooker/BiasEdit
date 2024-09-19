''' Modified from https://github.com/alinlab/LfF/blob/master/module/mlp.py'''
import torch
import torch.nn as nn
 
class MLP(nn.Module):
    def __init__(self, num_classes = 10):
        super(MLP, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(3*28*28, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 16),
            nn.ReLU()
        )
        self.classifier = nn.Linear(16, num_classes)
        self.proj = nn.Sequential(
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
        )


    def forward(self, x, return_feat=False):
        x = x.view(x.size(0), -1) / 255
        x = self.feature(x)
        final_x = self.classifier(x)

        if return_feat:
            return final_x, x
        else:
            return final_x
