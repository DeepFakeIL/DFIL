import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import argparse
import os
import cv2
from network.models import model_selection
from dataset.transform import xception_default_data_transforms
from dataset.mydataset import MyDataset
import pandas as pd
import math

from torchsummary import summary

def main():
    args = parse.parse_args()
    test_list = args.test_list
    batch_size = args.batch_size
    model_path = args.model_path
    torch.backends.cudnn.benchmark=True
    test_dataset = MyDataset(txt_path=test_list, transform=xception_default_data_transforms['test'], get_feature = True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=8)
    test_dataset_size = len(test_dataset)
    corrects = 0
    acc = 0
    #model = torchvision.models.densenet121(num_classes=2)
    # model = model_selection(modelname='SupCON', num_out_classes=2, dropout=0.5)
    model = model_selection(modelname='xception', num_out_classes=2, dropout=0.5)
    model.load_state_dict(torch.load(model_path))
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model = model.cuda()
    model.eval()
    print(dir(model.model))
    
    # summary(model,(3, 299, 299),batch_size=1,device="cuda")

    sum = 0
    model.model.last_linear = nn.Sequential()
    feature = []
    with torch.no_grad():
        for (image, labels) in test_loader:
            image = image.cuda()
            #print(image.size())
            labels = labels.cuda()
            outputs,_ = model(image)
            print(outputs.shape)
            feature.append(outputs)
            #exit()
        #     _, preds = torch.max(outputs.data, 1)
        #     prob = nn.functional.softmax(outputs.data,dim=1)
    
        #     # calculate the difficult of sample 

        #     corrects += torch.sum(preds == labels.data).to(torch.float32)
        #     print('Iteration Acc {:.4f}'.format(torch.sum(preds == labels.data).to(torch.float32)/batch_size))
        # acc = corrects / test_dataset_size
        # print('Test Acc: {:.4f}'.format(acc))
    print(feature)
    a = feature[0]
    for i in range(len(feature)):
        if i == 0:
            continue
        a = torch.cat([a,feature[i]],dim = 0)
    torch.save(a, "20230510_800_sampes.pt")
    b = torch.load("20230510_800_sampes.pt")
    print(b.shape)



if __name__ == '__main__':

    parse = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parse.add_argument('--batch_size', '-bz', type=int, default=28)
    #parse.add_argument('--test_list', '-tl', type=str, default='./data_list/Deepfakes_c0_test.txt')
    parse.add_argument('--test_list', '-tl', type=str, default='FF++_DFDCP_DFD_CDF2.txt')
    #parse.add_argument('--model_path', '-mp', type=str, default='./pretrained_model/df_c0_best.pkl')
    parse.add_argument('--model_path', '-mp', type=str, default='20230414_Task4_CDF_with_KD_MFD_SCL_Memory/best.pkl')
    
    main()

    print('Hello world!!!')