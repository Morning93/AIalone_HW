import argparse

# hyper-parameter 선언 
def parse_train_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int, default=28, help='image data size for training and inferencing')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--hidden_size', type=int, default=500)
    parser.add_argument('--num_class', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epoch', type=int, default=3)
    parser.add_argument('--results_folder_path', type=str, default='results')

    # 사용하는 데이터셋에 관한 argument 추가
    parser.add_argument('--model_type', type = str, default = 'mlp', 
                        choices = ['mlp', 'lenet', 'lenet_inj', 'lenet_multiconv', 'lenet_incep', 'lenet_nh'])
    parser.add_argument('--data', type = str, default = 'MNIST', choices = ['MNIST', 'CIFAR'])
    
    args = parser.parse_args() 
    return args

def parse_infer_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_image_path', type=str, help='추론을 원하는 데이터의 경로')
    parser.add_argument('--load_folder', nargs='+',  help='학습된 모델을 담고 있는 폴더의 경로')
    args = parser.parse_args() 
    return args
