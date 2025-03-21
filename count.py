import numpy as np
import pandas as pd
from util.config import args
import random

def read_status():
    df = pd.read_excel('./data/case.xlsx', sheet_name='样本集')
    # print(df.iloc[:15,:])

    data_raw_x = df.iloc[:, 2:10]
    data_raw_y = df.iloc[:, 10]

    data_x = data_raw_x.values
    data_y = data_raw_y.values

    X_max = data_raw_x.max().values
    X_min = data_raw_x.min().values

    data_x = (data_x - X_min) / X_max

    n = 0
    fl = 0
    l=0
    length = len(data_x)
    X = []
    label = []

    images = []

    t1=0
    t2=0
    t3=0

    le = (len(data_x) if args.d_size <= 0 else args.d_size)

    while n + 10 <= le:
        # X.append(data_x[n:n + 10, :])
        if data_y[n]=='正常':
            a=random.random()
            if a<=0.5:
                X.append(data_x[n:n + 10, :])

                label.append(0)
                images.append('./data/images/{}.jpg'.format(fl))
                t1+=1


        elif data_y[n]=='异常':

            a=random.randint(1,3)
            for i in range(a):
                X.append(data_x[n:n + 10, :])
                label.append(1)
                images.append('./data/images/{}.jpg'.format(fl))
                t2+=1


        else:
            a=random.randint(4,6)
            for i in range(a):
                X.append(data_x[n:n + 10, :])
                label.append(2)
                images.append('./data/images/{}.jpg'.format(fl))
                t3+=1


        # label.append(l)
        # label.append(0 if data_y[n] == '正常' else 1 if data_y[n] == '异常' else 2)
        # transformer=transform_image()

        # img=load_and_transform_image('./data/images/{}.jpg'.format(fl), transformer)


        fl += 1
        n = n + 10

    # print(fl)
    print(t1,t2,t3,t1+t2+t3)

def read_type():
    df = pd.read_excel('./data/case.xlsx')
    data_raw_x = df.iloc[:, 2:10]
    data_raw_y = df.iloc[:, 11]

    data_x = data_raw_x.values
    data_y = data_raw_y.values

    X_max = data_raw_x.max().values
    X_min = data_raw_x.min().values

    data_x = (data_x - X_min) / X_max

    n = 0
    length = len(data_x)
    X = []
    label = []
    images = []

    t1=0
    t2=0
    t3=0
    t4=0
    t5=0
    t6=0

    i = 0
    le = (len(data_x) if args.d_size <= 0 else args.d_size)
    while n + 10 <= le:

        if data_y[n] == '无':

            a=random.random()
            if a<0.2:
                X.append(data_x[n:n + 10, :])
                label.append(0)
                images.append('./data/images/{}.jpg'.format(i))
                t1+=1


        elif data_y[n] == '高温过热':

            for l in range(2):
                X.append(data_x[n:n + 10, :])
                label.append(1)
                images.append('./data/images/{}.jpg'.format(i))
                t2+=1

        elif data_y[n] == '过热缺陷':

            for l in range(36):
                X.append(data_x[n:n + 10, :])
                label.append(2)
                images.append('./data/images/{}.jpg'.format(i))
                t3+=1


        elif data_y[n] == '局部放电':

            # for l in range(2):

            X.append(data_x[n:n + 10, :])
            label.append(3)
            images.append('./data/images/{}.jpg'.format(i))
            t4+=1

        elif data_y[n] == '受潮故障':

            for l in range(68):
                X.append(data_x[n:n + 10, :])
                label.append(4)
                images.append('./data/images/{}.jpg'.format(i))
                t5+=1


        elif data_y[n] == '悬浮放电':
            for l in range(2):
                X.append(data_x[n:n + 10, :])
                label.append(5)
                images.append('./data/images/{}.jpg'.format(i))
                t6+=1
        # label.append(fl)
        # images.append('./data/images/{}.jpg'.format(i))
        n = n + 10
        i = i + 1
    print(i)
    print(t1,t2,t3,t4,t5,t6,t1+t2+t3+t4+t5+t6)

def read_pre():
    df = pd.read_excel('./data/case.xlsx')
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
    fl = 0

    while n + 10 <= le and n + 60 <= le:
        X.append(data_x[n:n + 10, :])
        label_1 = []
        k = 20
        for i in range(5):
            label_1.append(data_x[n + k + 10 * i - 1])

        label.append(label_1)
        images.append('./data/images/{}.jpg'.format(fl))
        fl+=1
        n=n + 10
    print(fl)



read_pre()
