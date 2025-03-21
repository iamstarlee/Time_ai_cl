import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# from model import Class_no_I
from model import transformer_resnet
from sklearn import metrics
from util.config import args
from util.DataLoad_images import read_data_status,read_data_type,read_data_pre,read_data_type_cl
import os
import time
# from util.trans_to_snn import replace_activation_by_floor,replace_activation_by_neuron,reset_net,replace_maxpool2d_by_avgpool2d
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import math
def train(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCEWithLogitsLoss()
    criterion = nn.CrossEntropyLoss()
    epochs=args.epochs
    if args.task == 'status':
        train_loader,test_loader=read_data_status('./data/case.xlsx')
    elif args.task == 'type':
        train_loader,test_loader=read_data_type('./data/case.xlsx')
    elif args.task == 'type_cl':
        train_loader,test_loader=read_data_type_cl('./data/case.xlsx')
    else:
        train_loader,test_loader, X_max, X_min=read_data_pre('./data/case.xlsx')
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for epoch in range(epochs):
        model.train()
        run_loss=0.0
        for X,label,img in train_loader:
            # label=label.long()
            # X.to(device)
            # label.to(device)
            # img=img.to(device)


            X=X.cuda(args.gpu)
            label=label.cuda(args.gpu)
            img=img.cuda(args.gpu)
            ret=model.run_on_batch(X,img,label,optimizer)

            # ret=model(X,img,label)
            # torch.cuda.synchronize()
            #
            # loss = criterion(ret, label)
            # torch.cuda.synchronize()
            #
            # loss.backward()
            # torch.cuda.synchronize()
            # optimizer.step()
            # torch.cuda.synchronize()
            # optimizer.zero_grad()
            # #
            # run_loss += loss.item() * X.size(0)
            # optimizer.zero_grad()
            run_loss+=ret["loss"].item()*X.size(0)
        epoch_loss = run_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}")

    os.makedirs('./result/{}/{}/'.format(args.model,args.task), exist_ok=True)
    save_path='./result/{}/{}/model.pth'.format(args.model,args.task)

    torch.save(model.state_dict(),save_path)

    if args.task in ['status','type']:
        test(model, test_loader)
    else:

        test_pre(model,test_loader,X_max,X_min)

def test_(model):
    if args.task == 'status':
        train_loader,test_loader=read_data_status('./data/case.xlsx')
        test(model, test_loader)
    elif args.task == 'type':
        train_loader,test_loader=read_data_type('./data/case.xlsx')
        test(model, test_loader)
    elif args.task == 'type_cl':
        train_loader, test_loader = read_data_type_cl('./data/case.xlsx')
        test(model, test_loader)
    else:
        train_loader, test_loader, X_max, X_min = read_data_pre('./data/case.xlsx')
        test_pre(model, test_loader, X_max, X_min)
    # test(model,test_loader)


# def test(model,test_loader):
def test_pre(model,test_loader,X_max,X_min):

    correct = 0
    total = 0
    batch_num=0
    start_time = time.time()
    # with torch.no_grad():
    #
    #     for X,label,img in test_loader:
    #         label=label.long()
    #
    #         ret=model.run_on_batch(X,img,label,None)
    #
    #
    #
    #         _, predicted = torch.max(ret['label'], dim=1)
    #         total += label.size(0)
    #         correct += (predicted == label).sum().item()
    #         batch_num += 1

    # tot = torch.zeros(args.t).cuda(args.gpu)
    length = 0


    model = model.cuda(args.gpu)
    model.eval()
    # valuate
    with torch.no_grad():
        for X,label,img in test_loader:
            X = X.cuda(args.gpu)
            # label=label.long()
            label=label.cuda(args.gpu)
            img = img.cuda(args.gpu)
            out = model.run_on_batch(X, img, label, None)["label"]
            out = out.cpu().numpy()
            # out=out*X_max+X_min
            label=label.cpu().numpy()
            # label=label*X_max+X_min

            total+= np.sum(np.fabs(out-label))

            x,y,z=out.shape
            length+=x*y*z

            batch_num+=1

            # batch_num+=1
            # spikes = 0
            # length += len(label)
            # X = X.cuda(args.gpu)
            # label=label.long()
            # label = label.cuda(args.gpu)
            # img = img.cuda(args.gpu)
            #
            # for t in range(args.t):
            #     out = model.run_on_batch(X,img,label,None)["label"]
            #     spikes += out
            #     tot[t] += (label == spikes.max(1)[1]).sum()
            # reset_net(model)



            # print(predicted.cpu().numpy())
            # print(label.cpu().numpy())
        end_time = time.time()
        mae=total/length
        # accuracy = np.array((tot/length).cpu())[-1]
        print(f"Dataset:{args.task}\nModel:{args.model}\nTask:{args.train_or_test}")
        if torch.cuda.is_available():
            print(f"Device:{args.gpu}")
        else:
            print(f"Device:cpu")
        print(f"MAE on test set: {mae}")
        batch_time=(end_time-start_time)/float(batch_num)
        print(f"Batch time: {batch_time:.4f}s")


def test(model,test_loader):

    correct = 0
    total = 0
    batch_num=0
    start_time = time.time()
    with torch.no_grad():

        for X,label,img in test_loader:
            label=label.long()

            ret=model.run_on_batch(X,img,label,None)



            _, predicted = torch.max(ret['label'], dim=1)
            total += label.size(0)
            correct += (predicted.cuda() == label.cuda()).sum().item()
            batch_num += 1

    # tot = torch.zeros(args.t).cuda(args.gpu)
    # length = 0
    #
    # # model=replace_activation_by_neuron(model)
    # # model.images_encoder=replace_activation_by_neuron(model.images_encoder)
    # # model.transformer_encoder=replace_activation_by_neuron(model.transformer_encoder)
    # model = model.cuda(args.gpu)
    # model.eval()
    # # valuate
    # with torch.no_grad():
    #     for X,label,img in test_loader:
    #         batch_num+=1
    #         spikes = 0
    #         length += len(label)
    #         X = X.cuda(args.gpu)
    #         label=label.long()
    #         label = label.cuda(args.gpu)
    #         img = img.cuda(args.gpu)
    #
    #         for t in range(args.t):
    #             out = model.run_on_batch(X,img,label,None)["label"]
    #             spikes += out
    #             tot[t] += (label == spikes.max(1)[1]).sum()
    #         reset_net(model)



            # print(predicted.cpu().numpy())
            # print(label.cpu().numpy())
        end_time = time.time()
        # accuracy = np.array((tot/length).cpu())[-1]
        accuracy=correct/total
        print(f"Dataset:{args.task}\nModel:{args.model}\nTask:{args.train_or_test}")
        if torch.cuda.is_available():
            print(f"Device:{args.gpu}")
        else:
            print(f"Device:cpu")
        print(f"Accuracy on test set: {accuracy:.2%}")
        batch_time=(end_time-start_time)/float(batch_num)
        print(f"Batch time: {batch_time:.4f}s")

        # path='./result/{}/{}/model_all.pt'.format(args.model,args.task)
        # torch.save(model,path)



def run():
    if args.task=='status':
        class_num=3
    elif args.task=='type':
        class_num=6
    elif args.task=='type_cl':
        class_num=2
    else:
        class_num=5*8
    model_=globals()[args.model].Model(args.hidden_size,1,class_num,ddp=False)
    total_params = sum(p.numel() for p in model_.parameters() if p.requires_grad)
    print('Total params is {}'.format(total_params))




    if args.train_or_test=='train':


        if torch.cuda.is_available():
            model_ = model_.cuda(args.gpu)

        train(model_)

    else:
        save_path = './result/{}/{}/model.pth'.format(args.model, args.task)
        if torch.cuda.is_available():

            device = torch.device(args.gpu)
            model_.load_state_dict(torch.load(save_path,map_location=device))
            # model_ = replace_activation_by_neuron(model_)
        else:

            model_.load_state_dict(torch.load(save_path, map_location='cpu'))
            # model_= torch.load(save_path)
        test_(model_)



if __name__ == '__main__':
    run()

