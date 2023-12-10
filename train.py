# 필요한 패키지를 import 
'''
import os 
import torch.nn as nn 
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.transforms import Resize
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor
'''
import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from utils.HWparser import parse_train_args
from utils.utils import make_result_folder, make_save_folder, save_hparam, evaluate, save_model #, evaluate_per_class
from utils.get_modules import get_dataloader, get_model

def main():
    args = parse_train_args()
    args.device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    # device는 실행할 때 마다 달라지니까 parser.py에 박제하지 않고, 코드 실행 시 마다 새로 추가
    # 여기까지 하고 parser에 넣었던 변수들 앞에 전부 args. 붙여주기!
    '''
    hyper-parameter 선언 
    image_size = 28 
    batch_size = 100 
    hidden_size = 500 
    num_class = 10 
    lr = 0.001
    epoch = 3
    results_folder_path = 'results'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    '''

    # 전체 저장 폴더를 담고 있는 최상위 results folder를 만들고 그 경로를 받아옴 
    make_result_folder(args)
    '''
    # if not os.path.exists(args.results_folder_path) : 
    #     os.makedirs(args.results_folder_path)
    '''
    # 저장할 데이터를 품고있을 상위폴더의 위치를 찾아내야함 
    make_save_folder(args)
    '''
    # folder_contents = os.listdir(args.results_folder_path) + ['-1']
    # max_folder_name = max([int(f) for f in folder_contents])
    # new_folder_name = str(max_folder_name + 1).zfill(2)
    # save_path = os.path.join(args.results_folder_path, new_folder_name)
    # os.makedirs(save_path)
    '''
    # 선언한 hparam을 저장 
    save_hparam(args)
    '''
    # with open(os.path.join(save_path, 'hparam.txt'), 'w') as f: 
    #     f.write('28|100|500|10|0.001|5|results')
    '''
    # 데이터 불러오기 
    # dataset 만들기 & 전처리하는 코드도 같이 작성 
    '''
    # transform = Compose([Resize((args.image_size, 
    #                              args.image_size)), 
    #                      ToTensor()])
    '''
    '''
    # train_val_dataset = MNIST(root='../../data', train=True, download=True, transform=transform)
    # train_dataset, val_dataset = random_split(train_val_dataset, 
    #                                           [50000, 10000], 
    #                                           torch.Generator().manual_seed(42))
    # test_dataset = MNIST(root='../../data', train=False, download=True, transform=transform)
    '''
    # dataloader 만들기 
    train_loader, val_loader, test_loader = get_dataloader(args)
    '''
    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    # val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    # test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    '''
    '''
    # AI 모델 설계도 만들기 (class)
    # init, forward 구현하기 
    # class myMLP(nn.Module):
    #     def __init__(self, image_size, hidden_size, num_class): 
    #         super().__init__()
    #         self.image_size = image_size 
    #         self.mlp1 = nn.Linear(in_features=image_size*image_size, out_features=hidden_size)
    #         self.mlp2 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
    #         self.mlp3 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
    #         self.mlp4 = nn.Linear(in_features=hidden_size, out_features=num_class)

    #     def forward(self, x): # x : [batch_size, height, width]
    #         batch_size = x.shape[0]
    #         x = torch.reshape(x, (batch_size, self.image_size * self.image_size)) # x : [batch_size, 784]
    #         x = self.mlp1(x)
    #         x = self.mlp2(x)
    #         x = self.mlp3(x)
    #         x = self.mlp4(x) # x : [batch_size, 10]
    #         return x 
    '''
    '''
    # 평가 함수 구현 (입력: model, dataloader) 
    # def evaluate(model, dataloader, device): 
    #     with torch.no_grad() :
    #         model.eval() 
    #         corrects, totals = 0, 0
    #         # dataloader를 바탕으로 for문을 돌면서 : 
    #         for image, label in dataloader: 
    #             # 데이터와 정답을 받아서 
    #             image, label = image.to(device), label.to(device) 

    #             # 모델에 입력을 넣고 출력을 생성, 출력 : [0.1, 0.05, 0.05, 0.70, 0.02, 0.02, 0.02, 0.02, 0.01, 0.01]
    #             output = model(image)
    #             # 출력물을 실제 정답과 비교 가능한 숫자로 변경 
    #             output_index = torch.argmax(output, dim=1)
    #             # 출력과 실제 정답을 비교 (4, 3) -> correct 
    #             corrects += torch.sum(label == output_index).item()
    #             totals += image.shape[0]
    #     acc = corrects / totals 
    #     model.train()
    #     # return acc (correts / totals)
    #     return acc

    # def evaluate_per_class(model, dataloader, device, total_num_class=10): 
    #     with torch.no_grad() :
    #         model.eval() 
    #         corrects, totals = torch.zeros(total_num_class), torch.zeros(total_num_class)
    #         for image, label in dataloader: 
    #             image, label = image.to(device), label.to(device) 
    #             output = model(image)
    #             output_index = torch.argmax(output, dim=1)
    #             # # 들어온 정답 데이터를 기준으로 for문 
    #             # for idx, lbl in enumerate(label) : 
    #             #     lbl = lbl.item() 
    #             #     totals[lbl] += 1 
    #             #     if output_index[idx].item() == lbl : 
    #             #         corrects[lbl] += 1
    #             # 클래스 정보를 바탕으로 for문 
    #             for _class in range(total_num_class): 
    #                 totals[_class] += (label == _class).sum().item() 
    #                 corrects[_class] += ((label == _class) * (output_index == _class)).sum().item()

    #     acc = corrects / totals 
    #     model.train()
    #     return acc # 10짜리 벡터 텐서의 형태 
    '''
    # AI 모델 객체 생성 (과정에서 hyper-parameter가 사용)
    model = get_model(args)
    '''
    # model = myMLP(image_size=args.image_size, 
    #               hidden_size=args.hidden_size, 
    #               num_class=args.num_class).to(args.device) 
    '''
    # Loss 객체 만들고 
    criteria = CrossEntropyLoss()
    # Optimizer 객체도 만들고 
    optimizer = Adam(model.parameters(), lr=args.lr)

    # -------- 준비단계 -------
    # -------- 학습단계 -------

    best = -1
    # for loop를 기반으로 학습이 시작됨 
    for ep in range(args.epoch): 
        # [epoch]을 학습하기 위해 batch 단위로 데이터를 가져와야 함 
        # 이 과정이 for loop로 진행 
        for idx, (image, label) in enumerate(train_loader):
            # dataloader가 넘겨주는 데이터를 받아서 
            image = image.to(args.device) 
            label = label.to(args.device) 

            # AI 모델에게 넘겨주고 
            output = model(image)
            # 출력물을 기반으로 Loss를 구하고 
            loss = criteria(output, label)
            # Loss를 바탕으로 Optimize를 진행  
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # 특정 조건을 제시해서, 그 조건이 만족한다면 학습의 중간 과정을 확인 
            if idx % 100 == 0 :
                # 평가를 진행 
                acc = evaluate(model, val_loader, args.device) 
                # acc_per_class = evaluate_per_class(model, val_loader, device)
                # 보고 싶은 수치 확인 (Loss, 평가 결과 값, 이미지와 같은 meta-data)
                print(f'Epoch : {ep}/{args.epoch}, step : {idx}, Loss : {loss.item():.3f}')
                # 만약 평가 결과가 나쁘지 않으면 
                if best < acc : 
                    print(f'이전보다 성능이 좋아짐 {best} -> {acc}')
                    best = acc 
                    # 모델을 저장 
                    save_model(args)
                    '''
                    torch.save(model.state_dict(), 
                            os.path.join(args.save_path, 'best_model.ckpt'))
                    '''


    final_acc = evaluate(model, test_loader, args.device) 
    print(f'최종 test data에 해당하는 평가 결과는 {final_acc:.3f}입니다')

if __name__ == '__main__':
    main()