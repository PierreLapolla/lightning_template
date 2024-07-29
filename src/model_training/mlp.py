import torch.nn as nn

from .base_model import BaseModel


class MLP(BaseModel):
    def __init__(self):
        super(MLP, self).__init__()

        self.input_size = 28 * 28
        self.output_size = 10

        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.output_size),
        )

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        x = x.view(batch_size, -1)
        return self.layers(x)
