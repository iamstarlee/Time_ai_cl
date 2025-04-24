from torch.utils.data import DataLoader, TensorDataset,Dataset
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from util.config import args
import torch
from PIL import Image
import torchvision.transforms as transforms
import random
def transform_image():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# def load_and_transform_image(image_path, transform):
#     # 加载图像
#     image = Image.open(image_path).convert('RGB')
#     # 应用变换
#     transformed_image = transform(image)
#     # 移动到指定设备
#     # transformed_image = transformed_image.to(device)
#     return transformed_image



class MyDataset(Dataset):
    def __init__(self, dataset, images, transform=None, device='cuda:0'):
        self.dataset = dataset
        self.images = images
        self.transform = transform
        self.device = device  # 指定设备

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        time_series, label = self.dataset[idx]
        image_path = self.images[idx]
        image = Image.open(image_path).convert('RGB')


        # 应用转换
        if self.transform:
            image = self.transform(image)

        # 将张量移动到指定设备
        # time_series = time_series.cuda(self.device)
        # label = label.cuda(self.device)
        # image = image.cuda(self.device)

        return time_series, label,image
        # return time_series, label, tobrch.Tensor([1])

def read_data_status(path):
    df = pd.read_excel(path, sheet_name='样本集')
    # print(df.iloc[:15,:])

    data_raw_x=df.iloc[:,2:10]
    data_raw_y=df.iloc[:,10]

    data_x=data_raw_x.values
    data_y=data_raw_y.values

    X_max=data_raw_x.max().values
    X_min=data_raw_x.min().values

    data_x=(data_x-X_min)/X_max

    n=0
    fl=0
    length=len(data_x)
    X=[]
    label=[]

    images=[]


    t1 = 0
    t2 = 0
    t3 = 0

    le = (len(data_x) if args.d_size <= 0 else args.d_size)

    while n + 10 <= le:
        # X.append(data_x[n:n + 10, :])
        if data_y[n] == '正常':
            a = random.random()
            if a <= 0.5:
                X.append(data_x[n:n + 10, :])

                label.append(0)
                images.append('./data/images/{}.jpg'.format(fl))
                t1 += 1


        elif data_y[n] == '异常':

            a = random.randint(1, 3)
            for i in range(a):
                X.append(data_x[n:n + 10, :])
                label.append(1)
                images.append('./data/images/{}.jpg'.format(fl))
                t2 += 1


        else:
            a = random.randint(4, 6)
            for i in range(a):
                X.append(data_x[n:n + 10, :])
                label.append(2)
                images.append('./data/images/{}.jpg'.format(fl))
                t3 += 1

        fl+=1
        n=n+10
    # print(X)
    X=np.array(X)
    label=np.array(label)

    X_train, X_test, y_train, y_test,img_train,img_test = train_test_split(X, label, images, test_size=0.3, random_state=42)

    train_data = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))

    train_data = MyDataset(train_data, img_train, transform=transform_image())


    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)


    test_data = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))
    test_data = MyDataset(test_data, img_test, transform=transform_image())
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    # train_loader,test_loader=load_data(X,label)
    return train_loader,test_loader



    # print(data_x)
def read_data_type(path):
    df = pd.read_excel(path, sheet_name='样本集')
    # print(df.iloc[:15,:])

    data_raw_x=df.iloc[:,2:10]
    data_raw_y=df.iloc[:,11]

    data_x=data_raw_x.values
    data_y=data_raw_y.values

    X_max=data_raw_x.max().values
    X_min=data_raw_x.min().values

    data_x=(data_x-X_min)/X_max

    n=0
    length=len(data_x)
    X=[]
    label=[]
    images=[]

    t1 = 0
    t2 = 0
    t3 = 0
    t4 = 0
    t5 = 0
    t6 = 0

    i = 0
    le = (len(data_x) if args.d_size <= 0 else args.d_size)
    while n + 10 <= le:

        if data_y[n] == '无':

            a = random.random()
            if a < 0.2:
                X.append(data_x[n:n + 10, :])
                label.append(0)
                images.append('./data/images/{}.jpg'.format(i))
                t1 += 1


        elif data_y[n] == '高温过热':

            for l in range(2):
                X.append(data_x[n:n + 10, :])
                label.append(1)
                images.append('./data/images/{}.jpg'.format(i))
                t2 += 1

        elif data_y[n] == '过热缺陷':

            for l in range(36):
                X.append(data_x[n:n + 10, :])
                label.append(2)
                images.append('./data/images/{}.jpg'.format(i))
                t3 += 1


        elif data_y[n] == '局部放电':

            # for l in range(2):

            X.append(data_x[n:n + 10, :])
            label.append(3)
            images.append('./data/images/{}.jpg'.format(i))
            t4 += 1

        elif data_y[n] == '受潮故障':
        
            for l in range(68):
                X.append(data_x[n:n + 10, :])
                label.append(4)
                images.append('./data/images/{}.jpg'.format(i))
                t5 += 1
        
        
        elif data_y[n] == '悬浮放电':
            for l in range(2):
                X.append(data_x[n:n + 10, :])
                label.append(5)
                images.append('./data/images/{}.jpg'.format(i))
                t6 += 1
        
        # label.append(fl)
        # images.append('./data/images/{}.jpg'.format(i))
        n = n + 10
        i = i + 1
    # print(label)
    X=np.array(X)
    label=np.array(label)
    # device = torch.device(args.gpu if torch.cuda.is_available() else 'cpu')
    # print(label.shape)
    # print(args.batch_size)
    X_train, X_test, y_train, y_test,img_train,img_test = train_test_split(X, label, images, test_size=0.3, random_state=42)

    train_data = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))

    train_data = MyDataset(train_data, img_train, transform=transform_image())


    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)


    test_data = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))
    test_data = MyDataset(test_data, img_test, transform=transform_image())
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    # train_loader,test_loader=load_data(X,label)
    return train_loader,test_loader

def read_data_type_cl(path):
    df = pd.read_excel(path, sheet_name='样本集')
    # print(df.iloc[:15,:])

    data_raw_x=df.iloc[:,2:10]
    data_raw_y=df.iloc[:,11]

    data_x=data_raw_x.values
    data_y=data_raw_y.values

    X_max=data_raw_x.max().values
    X_min=data_raw_x.min().values

    data_x=(data_x-X_min)/X_max

    n=0
    length=len(data_x)
    X=[]
    label=[]
    images=[]

    t1 = 0
    t2 = 0
    t3 = 0
    t4 = 0
    t5 = 0
    t6 = 0

    i = 0
    le = (len(data_x) if args.d_size <= 0 else args.d_size)
    while n + 10 <= le:

        # if data_y[n] == '无':
        #
        #     a = random.random()
        #     if a < 0.2:
        #         X.append(data_x[n:n + 10, :])
        #         label.append(0)
        #         images.append('./data/images/{}.jpg'.format(i))
        #         t1 += 1
        #
        #
        # elif data_y[n] == '高温过热':
        #
        #     for l in range(2):
        #         X.append(data_x[n:n + 10, :])
        #         label.append(1)
        #         images.append('./data/images/{}.jpg'.format(i))
        #         t2 += 1
        #
        # elif data_y[n] == '过热缺陷':
        #
        #     for l in range(36):
        #         X.append(data_x[n:n + 10, :])
        #         label.append(2)
        #         images.append('./data/images/{}.jpg'.format(i))
        #         t3 += 1
        #
        #
        # elif data_y[n] == '局部放电':
        #
        #     # for l in range(2):
        #
        #     X.append(data_x[n:n + 10, :])
        #     label.append(3)
        #     images.append('./data/images/{}.jpg'.format(i))
        #     t4 += 1

        if data_y[n] == '受潮故障':

            for l in range(68):
                X.append(data_x[n:n + 10, :])
                label.append(4)
                images.append('./data/images/{}.jpg'.format(i))
                t5 += 1


        if data_y[n] == '悬浮放电':
            for l in range(2):
                X.append(data_x[n:n + 10, :])
                label.append(5)
                images.append('./data/images/{}.jpg'.format(i))
                t6 += 1
        # label.append(fl)
        # images.append('./data/images/{}.jpg'.format(i))
        n = n + 10
        i = i + 1
    # print(label)
    X=np.array(X)
    label=np.array(label)
    # device = torch.device(args.gpu if torch.cuda.is_available() else 'cpu')
    # print(label.shape)
    # print(args.batch_size)
    X_train, X_test, y_train, y_test,img_train,img_test = train_test_split(X, label, images, test_size=0.3, random_state=42)

    train_data = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))

    train_data = MyDataset(train_data, img_train, transform=transform_image())


    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)


    test_data = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))
    test_data = MyDataset(test_data, img_test, transform=transform_image())
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    # train_loader,test_loader=load_data(X,label)
    return train_loader,test_loader

def read_data_pre(path):
    df = pd.read_excel(path, sheet_name='样本集')
    # print(df.iloc[:15,:])

    data_raw_x = df.iloc[:, 2:10]
    # data_raw_y = df.iloc[:, 11]

    data_x = data_raw_x.values
    # data_y = data_raw_y.values

    X_max = data_raw_x.max().values
    X_min = data_raw_x.min().values

    data_x = (data_x - X_min) / X_max

    le = (len(data_x) if args.d_size <= 0 else args.d_size)

    n = 0
    # length = len(data_x)
    X = []
    label = []
    images = []
    fl=0

    while n + 10 <= le and n+60<=le:
        X.append(data_x[n:n + 10, :])
        label_1=[]
        k=20
        for i in range(5):

            label_1.append(data_x[n+k+10*i-1])

        label.append(label_1)
        images.append('./data/images/{}.jpg'.format(fl))
        fl+=1
        n+=10

    X = np.array(X)
    label = np.array(label)
    # device = torch.device(args.gpu if torch.cuda.is_available() else 'cpu')
    # print(label.shape)
    # print(args.batch_size)
    X_train, X_test, y_train, y_test, img_train, img_test = train_test_split(X, label, images, test_size=0.3,
                                                                             random_state=42)

    train_data = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))

    train_data = MyDataset(train_data, img_train, transform=transform_image())

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    test_data = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))
    test_data = MyDataset(test_data, img_test, transform=transform_image())
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    # train_loader,test_loader=load_data(X,label)
    return train_loader, test_loader,X_max,X_min

