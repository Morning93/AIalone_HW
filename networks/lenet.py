import torch
import torch.nn as nn 

# AI 모델 설계도 만들기 (class)
# init, forward 구현하기 
class lenet(nn.Module):
    def __init__(self, args): 
        super().__init__()
        num_class = args.num_class
        self.Conv1 = nn.Sequential(
                    nn.Conv2d(in_channels = 3, out_channels = 6, 
                              kernel_size = 5, stride = 1, padding = 0),
                    nn.BatchNorm2d(num_featrues = 6),
                    nn.ReLU(),
        ) # list에 넣지 말고 그냥 콤마로!!, ReLU는 inpur parameter 없음
        self.Pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.Conv2 = nn.Sequential(
                    nn.Conv2d(in_channels = 6, out_channels = 16, 
                              kernel_size = 5, stride = 1, padding = 0),
                    nn.BatchNorm2d(16),
                    nn.ReLU(),
        )
        self.Pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.FC1 = nn.Sequential(
            nn.Linear(in_features = 400, out_features = 120,),
            nn.ReLU(),
        ) #ReLU 빼먹지 않기!!
        self.FC2 = nn.Sequential(
            nn.Linear(in_features = 120, out_features = 84),
            nn.ReLU(),
        )
        self.FC3 = nn.Sequential(
            nn.Linear(in_features = 84, out_features = num_class),
            nn.ReLU(),
        )

    def forward(self, x): # x : [batch_size, height, width]
        batch_size = x.shape[0]

        x = self.Conv1(x)
        x = self.Pool1(x)
        x = self.Conv2(x)
        x = self.Pool2(x) # x : [batch_size, 10]

        x = torch.reshape(x, (batch_size, 400))

        x = self.FC1(x)
        x = self.FC2(x)
        x = self.FC3(x)
        
        return x 