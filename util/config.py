import argparse

parser = argparse.ArgumentParser(description="Process")
parser.add_argument('--data_dir', type=str, default='./data')
parser.add_argument('--model', type=str, choices=['transformer_resnet'],default='transformer_resnet')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--hidden_size', type=int, default=8)

parser.add_argument('--task', type=str,choices=['status','type','type_cl'], default="type")  #只用到type中的4类,type_cl是剩下的两类
parser.add_argument('--train_or_test',type=str,choices=['train','test'],default='train')
parser.add_argument('--gpu',type=int,default=1)
parser.add_argument('--d_size', default=-1, type=int, help='T')

args = parser.parse_args()
