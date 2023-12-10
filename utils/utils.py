import os
import torch
import json


def make_result_folder(args):
    if not os.path.exists(args.results_folder_path) : 
        os.makedirs(args.results_folder_path)

def make_save_folder(args):
    folder_contents = os.listdir(args.results_folder_path) + ['-1']
    max_folder_name = max([int(f) for f in folder_contents])
    new_folder_name = str(max_folder_name + 1).zfill(2)
    save_path = os.path.join(args.results_folder_path, new_folder_name)
    os.makedirs(save_path)
    args.save_path = save_path

def evaluate(model, dataloader, device): 
    with torch.no_grad() :
        model.eval() 
        corrects, totals = 0, 0
        # dataloader를 바탕으로 for문을 돌면서 : 
        for image, label in dataloader: 
            # 데이터와 정답을 받아서 
            image, label = image.to(device), label.to(device) 

            # 모델에 입력을 넣고 출력을 생성, 출력 : [0.1, 0.05, 0.05, 0.70, 0.02, 0.02, 0.02, 0.02, 0.01, 0.01]
            output = model(image)
            # 출력물을 실제 정답과 비교 가능한 숫자로 변경 
            output_index = torch.argmax(output, dim=1)
            # 출력과 실제 정답을 비교 (4, 3) -> correct 
            corrects += torch.sum(label == output_index).item()
            totals += image.shape[0]
    acc = corrects / totals 
    model.train()
    # return acc (correts / totals)
    return acc

def evaluate_per_class(model, dataloader, device, total_num_class=10): 
    with torch.no_grad() :
        model.eval() 
        corrects, totals = torch.zeros(total_num_class), torch.zeros(total_num_class)
        for image, label in dataloader: 
            image, label = image.to(device), label.to(device) 
            output = model(image)
            output_index = torch.argmax(output, dim=1)
            # # 들어온 정답 데이터를 기준으로 for문 
            # for idx, lbl in enumerate(label) : 
            #     lbl = lbl.item() 
            #     totals[lbl] += 1 
            #     if output_index[idx].item() == lbl : 
            #         corrects[lbl] += 1
            # 클래스 정보를 바탕으로 for문 
            for _class in range(total_num_class): 
                totals[_class] += (label == _class).sum().item() 
                corrects[_class] += ((label == _class) * (output_index == _class)).sum().item()

    acc = corrects / totals 
    model.train()
    return acc # 10짜리 벡터 텐서의 형태 

def save_hparam(args):
    dict_args = vars(args).copy() # device를 떼고 저장해야 하는데 원본을 건드리면 안되니까 copy!
    del dict_args['device']
    with open(os.path.join(args.save_path, 'hparam.json'), 'w') as f:
        json.dump(dict_args, f, indent = 4)

def save_model(args, model):
    torch.save(model.state_dict(), 
               os.path.join(args.save_path, 'best_model.ckpt'))
    
def get_ckpt_path(args):
    ckpt_path = os.path.join(args.load_folder, 'best_model.ckpt')
    return ckpt_path

def get_hparam_path(args):
    hparam_path = os.path.join(args.load_folder, 'hparam.txt')
    return hparam_path

