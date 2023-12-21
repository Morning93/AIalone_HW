import torch
import torch.nn as nn 

# AI 모델 설계도 만들기 (class)
# init, forward 구현하기 
class myMLP(nn.Module):
    def __init__(self, args): 
        super().__init__()
        self.image_size = args.image_size 
        self.hidden_size = args.hidden_size
        self.num_class = args.num_class
        self.mlp1 = nn.Linear(in_features=self.image_size*self.image_size, out_features=self.hidden_size)
        self.mlp2 = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size)
        self.mlp3 = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size)
        self.mlp4 = nn.Linear(in_features=self.hidden_size, out_features=self.num_class)

    def forward(self, x): # x : [batch_size, height, width]
        batch_size = x.shape[0]
        x = torch.reshape(x, (batch_size, self.image_size * self.image_size)) # x : [batch_size, 784]
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.mlp3(x)
        x = self.mlp4(x) # x : [batch_size, 10]
        return x 