import argparse

def parse_train_args():
    parser = argparse.ArgumentParser() # parser 객체 만들기
    # parser에 argument들 추가
    parser.add_argument('--image_size', type = int, default = 28,
                        help = 'input imags size')
    parser.add_argument('--batch_size', type = int, default = 100)
    parser.add_argument('--hidden_size', type = int, default = 500,
                        help = 'hidden layer의 output_features')
    parser.add_argument('--num_class', type = int, default = 10)
    parser.add_argument('--lr', type = float, default = 0.001,
                        help = 'learning rate')
    parser.add_argument('--epoch', type = int, default = 3)
    parser.add_argument('--results_folder_path', type = str, default = 'results',
                        help = '학습 결과를 저장 할 최상위 폴더 경로')

    #args 변수에 파싱된 arguments들 연결!
    args = parser.parse_args()
    return args


def parse_infer_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_image_path', type= str, default = 'mnist_test.png', help = 'infer용 데이터 경로')
    parser.add_argument('--load_folder', nargs = '+', type= str, help = '학습 결과를 담고있는 폴더 경로')

    args = parser.parse_args()
    return args