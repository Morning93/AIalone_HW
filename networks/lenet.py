import torch
import torch.nn as nn 

# AI 모델 설계도 만들기 (class)
# init, forward 구현하기 
class LeNet(nn.Module):
    def __init__(self, args): 
        super().__init__()
        num_class = args.num_class
        self.conv1 = nn.Sequential(
                    nn.Conv2d(in_channels = 3, out_channels = 6, 
                              kernel_size = 5, stride = 1, padding = 0),
                    nn.BatchNorm2d(num_featrues = 6),
                    nn.ReLU(),
        ) # list에 넣지 말고 그냥 콤마로!!, ReLU는 inpur parameter 없음
        self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv2 = nn.Sequential(
                    nn.Conv2d(in_channels = 6, out_channels = 16, 
                              kernel_size = 5, stride = 1, padding = 0),
                    nn.BatchNorm2d(16),
                    nn.ReLU(),
        )
        self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
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

        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x) # x : [batch_size, 10]

        x = torch.reshape(x, (batch_size, 400))

        x = self.FC1(x)
        x = self.FC2(x)
        x = self.FC3(x)
        
        return x 
    

class LeNet_inj(nn.Module):
    def __init__(self, args): 
        super().__init__()
        num_class = args.num_class
        self.conv1 = nn.Sequential(
                    nn.Conv2d(in_channels = 3, out_channels = 6, 
                              kernel_size = 5, stride = 1, padding = 0),
                    nn.BatchNorm2d(num_featrues = 6),
                    nn.ReLU(),
        ) 
        self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.inj_Linear1 = nn.Sequential(
            nn.Linear(1176, 2048),
            nn.ReLU(),
        )

        self.inj_Linear2 = nn.Sequential(
            nn.Linear(2048, 1176),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
                    nn.Conv2d(in_channels = 6, out_channels = 16, 
                              kernel_size = 5, stride = 1, padding = 0),
                    nn.BatchNorm2d(16),
                    nn.ReLU(),
        )
        self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
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

        
        return x 
    
class LeNet_multiconv(nn.Module):
    def __init__(self, args, num_conv1 = 4, num_conv2 = 3): 
        super().__init__()
        num_class = args.num_class
        
        self.conv1 = []
        for i in range(num_conv1):
            conv1_in_channel = 3 if i == 0 else 6
            conv1_padding = 0 if i == (num_conv1 - 1) else 2
            module = nn.Sequential(
                nn.Conv2d(in_channels = conv1_in_channel, out_channels = 6, 
                              kernel_size = 5, stride = 1, padding = conv1_padding),
                    nn.BatchNorm2d(num_featrues = 6),
                    nn.ReLU(),
            )
            self.conv1.append(module)
        self.conv1 = nn.ModuleList(self.conv1)

        self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.conv2 = []
        for i in range(num_conv2):
            conv2_in_channel = 6 if i == 0 else 16
            conv2_padding = 0 if i == (num_conv1 - 1) else 2
            module = nn.Sequential(
                nn.Conv2d(in_channels = conv2_in_channel, out_channels = 6, 
                              kernel_size = 5, stride = 1, padding = conv2_padding),
                    nn.BatchNorm2d(num_featrues = 6),
                    nn.ReLU(),
            )
            self.conv2.append(module)
        self.conv2 = nn.ModuleList(self.conv2)
        
        self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
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
        for  module in self.conv1:
            x = module(x)
        x = self.pool1(x)

        for module in self.conv2:
            x = module(x)
        x = self.pool2(x)

        x = torch.reshape(x, (batch_size, -1))
        x = self.FC1(x)
        x = self.FC2(x)
        x = self.FC3(x)
        
        return x 
    
class LeNet_incep(nn.Module):
    def __init__(self, args): 
        super().__init__()
        num_class = args.num_class

        self.conv1_1 = nn.Sequential(
                    nn.Conv2d(in_channels = 3, out_channels = 6, 
                              kernel_size = 1, stride = 1, padding = 0),
                    nn.BatchNorm2d(num_featrues = 6),
                    nn.ReLU(),
        ) 

        self.conv1_2 = nn.Sequential(
                    nn.Conv2d(in_channels = 3, out_channels = 6, 
                              kernel_size = 3, stride = 1, padding = 1),
                    nn.BatchNorm2d(num_featrues = 6),
                    nn.ReLU(),
        ) 

        self.conv1_3 = nn.Sequential(
                    nn.Conv2d(in_channels = 3, out_channels = 6, 
                              kernel_size = 5, stride = 1, padding = 2),
                    nn.BatchNorm2d(num_featrues = 6),
                    nn.ReLU(),
        ) 

        self.conv1 = nn.Sequential(
                    nn.Conv2d(in_channels = 18, out_channels = 6, 
                              kernel_size = 5, stride = 1, padding = 0),
                    nn.BatchNorm2d(num_featrues = 6),
                    nn.ReLU(),
        ) 

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, 
                      kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.FC1 = nn.Sequential(
            nn.Linear(in_features=400, out_features=120),
            nn.ReLU(),
        )
        self.FC2 = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(),
        )
        self.FC3 =  nn.Linear(in_features=84, out_features=num_class)

    def forward(self, x): # x : [batch_size, height, width]
        batch_size = x.shape[0]

        x1 = self.conv1_1(x)
        x2 = self.conv1_2(x)
        x3 = self.conv1_3(x)

        x = torch.cat((x1, x2, x3), dim = 1)

        x = self.conv1(x)
        x = self.pool1(x)    
        x = self.conv2(x)
        x = self.pool2(x)
        
        x = torch.reshape(x, (batch_size, -1))
        x = self.FC1(x)
        x = self.FC2(x)
        x = self.FC3(x)
        
        return x 
    
class LeNet_nh(nn.Module):
    def __init__(self, args): 
        super().__init__()
        num_class = args.num_class
        self.conv1_1 = nn.Sequential(
                    nn.Conv2d(in_channels = 3, out_channels = 6, 
                              kernel_size = 1, stride = 1, padding = 0),
                    nn.BatchNorm2d(num_featrues = 6),
                    nn.ReLU(),
        ) 
        self.conv1_1_Linear1 = nn.Sequential(
                    nn.Linear(32*32*6, 32*32*3),
                    nn.ReLU(),
        )
        self.conv1_1_Linear2 = nn.Sequential(
                    nn.Linear(32*32*3, 32*32*6),
                    nn.ReLU(),
        )

        self.conv1_2 = nn.Sequential(
                    nn.Conv2d(in_channels = 3, out_channels = 6, 
                              kernel_size = 3, stride = 1, padding = 1),
                    nn.BatchNorm2d(num_featrues = 6),
                    nn.ReLU(),
        ) 
        self.conv1_2_Linear1 = nn.Sequential(
                    nn.Linear(32*32*6, 32*32*3),
                    nn.ReLU(),
        )
        self.conv1_2_Linear2 = nn.Sequential(
                    nn.Linear(32*32*3, 32*32*6),
                    nn.ReLU(),
        )

        self.conv1_3 = nn.Sequential(
                    nn.Conv2d(in_channels = 3, out_channels = 6, 
                              kernel_size = 5, stride = 1, padding = 2),
                    nn.BatchNorm2d(num_featrues = 6),
                    nn.ReLU(),
        ) 
        self.conv1_3_Linear1 = nn.Sequential(
                    nn.Linear(32*32*6, 32*32*3),
                    nn.ReLU(),
        )
        self.conv1_3_Linear2 = nn.Sequential(
                    nn.Linear(32*32*3, 32*32*6),
                    nn.ReLU(),
        )
        self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv2 = nn.Sequential(
                    nn.Conv2d(in_channels = 6, out_channels = 16, 
                              kernel_size = 5, stride = 1, padding = 0),
                    nn.BatchNorm2d(16),
                    nn.ReLU(),
        )
        self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.conv3 = nn.Sequential(
                    nn.Conv2d(in_channels = 16, out_channels = 32, 
                              kernel_size = 5, stride = 1, padding = 0),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
        )
        self.pool3 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.FC0 = nn.Sequential(
            nn.Linear(in_features = 800, out_features = 400,),
            nn.ReLU(),
        )

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

        x1 = self.conv1_1(x)
        _, c, h, w = x1.shape
        x1 = torch.reshape(x1, (batch_size, -1))
        x1 = self.conv1_1_Linear1(x1)
        x1 = self.conv1_1_Linear2(x1)
        x1 = torch.reshape(x1, (batch_size, c, h, w))

        x2 = self.conv1_2(x)
        _, c, h, w = x2.shape
        x2 = torch.reshape(x2, (batch_size, -1))
        x2 = self.conv1_2_Linear1(x2)
        x2 = self.conv1_2_Linear2(x2)
        x2 = torch.reshape(x2, (batch_size, c, h, w))

        x3 = self.conv1_3(x)
        _, c, h, w = x3.shape
        x3 = torch.reshape(x3, (batch_size, -1))
        x3 = self.conv1_3_Linear1(x3)
        x3 = self.conv1_3_Linear2(x3)
        x3 = torch.reshape(x3, (batch_size, c, h, w))

        x = torch.cat((x1, x2, x3), dim = 1)

        x = self.pool1(x)

        x = self.conv2(x)
        x = self.pool2(x) # x : [batch_size, 10]

        x = self.conv3(x)
        x = self.pool3(x)

        x = torch.reshape(x, (batch_size, 800))

        x = self.FC0(x)
        x = self.FC1(x)
        x = self.FC2(x)
        x = self.FC3(x)
        
        return x 