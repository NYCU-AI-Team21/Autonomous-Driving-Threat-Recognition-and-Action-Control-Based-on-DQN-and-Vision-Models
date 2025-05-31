import torch.nn as nn
import torch
'''
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

    def forward(self, x):
        return self.layers(x)
    
'''

class DQN(nn.Module):
    def __init__(self, action_size):
        super(DQN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), 
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  
            nn.ReLU()
        )

        # 計算 conv 輸出 feature map 的維度
        self.flatten_size = 64 * 7 * 7

        self.fc_layers = nn.Sequential(
            nn.Linear(self.flatten_size, 512),
            nn.ReLU(),
            nn.Linear(512, action_size)
        )

    def forward(self, x):
        x = x / 255.0  # 歸一化到 0-1
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

# class DQN(nn.Module):
#     def __init__(self, action_size, input_shape=(3,84,84)):
#         super().__init__()
#         self.conv_layers = nn.Sequential(
#             nn.Conv2d(3, 32, 8, 4),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, 4, 2),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, 3, 1),
#             nn.ReLU(),
#         )
#         # 动态推 feature 数量
#         with torch.no_grad():
#             dummy = torch.zeros(1, *input_shape)
#             conv_out = self.conv_layers(dummy)
#             n_features = conv_out.view(1, -1).size(1)

#         self.fc_layers = nn.Sequential(
#             nn.Linear(n_features, 512),
#             nn.ReLU(),
#             nn.Linear(512, action_size)
#         )

#     def forward(self, x):
#         x = self.conv_layers(x)
#         x = x.view(x.size(0), -1)
#         return self.fc_layers(x)
