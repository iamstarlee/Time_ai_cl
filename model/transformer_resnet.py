import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import math
from util.config import args
# from util.trans_to_snn import replace_activation_by_floor,replace_activation_by_neuron,reset_net
from torchvision import models

from model.VGG import vgg16

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        # pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class MLP(nn.Module):
    def __init__(self,model_dim,image_dim=1,class_num=3):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(model_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, class_num)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

class Model(nn.Module):
    def __init__(self,model_dim,image_dim=1, class_num=3,ddp=False):
        super(Model, self).__init__()
        self.ddp=ddp
        self.model_type = 'Transformer'
        feature_size=8
        self.model_dim=model_dim
        self.image_dim=image_dim
        self.hidden_size=64
        self.embedding=nn.Linear(feature_size, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=8, dropout=0.5)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3)
        # self.transformer_encoder = replace_activation_by_floor(self.transformer_encoder, t=args.l)
        self.decoder = nn.Linear(model_dim,model_dim)
        self.prediction=nn.LSTM(8,64,batch_first=True)
        self.init_weights()

        # self.images_encoder = models.vgg16(pretrained=False)
        # self.images_encoder.load_state_dict(torch.load("./model/vgg16-397923af.pth"))
        self.images_encoder = vgg16(num_classes=1000)
        self.images_encoder.classifier[-1] = nn.Identity()
        # self.images_encoder=replace_activation_by_floor(self.images_encoder,t=args.l)
        # self.en1 = nn.Linear(2048 * 4, 256)
        self.en1 = nn.Linear(4096, image_dim)

        # self.images_encoder = models.resnet152(pretrained=False)
        #
        # self.images_encoder.load_state_dict(torch.load("./model/resnet152.pth"))
        # # self.images_encoder.load_state_dict(torch.load("./resnet152-b121ed2d.pth"))
        # self.images_encoder.fc = nn.Identity()
        #
        # self.images_encoder = replace_activation_by_floor(self.images_encoder, t=args.l)
        # # self.en1 = nn.Linear(2048 * 4, 256)
        # self.en1 = nn.Linear(2048,128)

        self.out = MLP(model_dim,image_dim,class_num)
        self.out1=nn.Linear(64+model_dim+image_dim,class_num)
        # self.out1 = nn.Linear(64 + model_dim , class_num)

        self.criterion = nn.CrossEntropyLoss()
        self.loss=nn.MSELoss()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self,src,img,labels):



        src1 = self.embedding(src.cuda(args.gpu))
        src1=src1.reshape(src.shape[1],src.shape[0],-1)

        src2 = self.pos_encoder(src1)

        output = self.transformer_encoder(src2)
        output = output.reshape(output.shape[1],output.shape[0],-1)

        output=output.mean(dim=1)
        output=self.decoder(output)

        label_img = self.images_encoder(img)
        label_img = self.en1(label_img)
        h=torch.zeros(src.size(0),64).cuda(args.gpu)
        c=torch.zeros(src.size(0),64).cuda(args.gpu)

        out,(h,c)=self.prediction(src)


        # output1=torch.cat((label_img,output),dim=1)
        output1=torch.cat((output,h.squeeze(),label_img),dim=1)
        # output1 = torch.cat((output, h.squeeze()), dim=1)
        # label_pre = self.out(output1)
        label_pre=self.out1(output1)


        if self.ddp:
            return label_pre
        else:

            if args.task in ['status','type']:
                labels=labels.long()
                loss = self.criterion(label_pre.cuda(), labels.cuda())

            else:
                b,t,f=labels.shape
                label_pre=label_pre.reshape(b,t,-1)
                loss = self.loss(label_pre, labels)






            return {"label":label_pre, "loss": loss}



        # return label_pre

    def run_on_batch(self, data,img, labels, optimizer):
        ret = self(data.cuda(args.gpu),img.cuda(args.gpu), labels.cuda(args.gpu))
        if optimizer is not None:
            optimizer.zero_grad()
            ret["loss"].backward()
            optimizer.step()

        return ret

# if __name__ == '__main__':
#     model = Model(1)
#
#     # Count the number of parameters
#     num_params = sum(p.numel() for p in model.parameters())
#
#     print(f'Total number of parameters: {num_params}')


