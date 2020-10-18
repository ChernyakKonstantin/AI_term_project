import torch
import torch.nn as nn
import numpy as np

class Net(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
                                  nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=1, padding=1),
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=2,stride=2),
                                  nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=2,stride=2),
                                  nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                                  # nn.MaxPool2d(kernel_size=2,stride=2),
                                  nn.ReLU()
                                 )
        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
                                nn.Linear(conv_out_size * 2, 256),
                                nn.ReLU(),
                                nn.Linear(256, 256),
                                nn.ReLU(),
                                nn.Linear(256, n_actions)
                               )
        
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    
    def forward(self, x1, x2):
        conv_out_1 = self.conv(x1).view(x1.size()[0], -1)
        conv_out_2 = self.conv(x2).view(x2.size()[0], -1)
        conv_out = conv_out = torch.cat((conv_out_1, conv_out_2), dim=1) #Unite to one vector
        return self.fc(conv_out)
