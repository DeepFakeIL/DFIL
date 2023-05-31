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
import glob
from torch.nn import functional as F
import torch.nn.functional as F

def main():
    args = parse.parse_args()
    test_list = args.test_list
    batch_size = args.batch_size
    model_path = args.model_path
    torch.backends.cudnn.benchmark=True
    test_dataset = MyDataset(txt_path=test_list, transform=xception_default_data_transforms['test'],get_feature=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=8)
    test_dataset_size = len(test_dataset)

    print('the number of test image is ', test_dataset_size)
    corrects = 0
    acc = 0
    #model = torchvision.models.densenet121(num_classes=2)
    model = model_selection(modelname='xception', num_out_classes=2, dropout=0.5)
    model.load_state_dict(torch.load(model_path))
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model = model.cuda()
    model.eval()


    file_image_dir = test_list
    res = [] 

    image_label = []
    image_confidence = []
    image_margin = []
    image_entropy = []
    image_pred = []
    dis = []
    fh = open(file_image_dir, 'r')

    # for line in fh:     # False
    #     line = line.rstrip()
    #     words = line.split(',')

    #     image_file = words[1]
    #     path = os.path.join(image_file,'*.png')
    #     image_filenames = sorted(glob.glob(path))

    #     for i in image_filenames:
    #         res.append(i)
    #         image_label.append(int(words[0]))


    for line in fh:   # True
        line = line.rstrip()
        words = line.split(',')
        res.append(words[1])
        image_label.append(int(words[0]))
    print(len(image_label))
    print(len(res))
    # exit()

    feature_mean_0 = None
    feature_mean_1 = None
    num_0 = 0
    num_1 = 0
    with torch.no_grad():
        for (image, labels) in test_loader:
            image = image.cuda()
            labels = labels.cuda()
            outputs,fc_features = model(image)
            fc_features = F.adaptive_avg_pool2d(fc_features, (1, 1)) 
            fc_features = fc_features.view(fc_features.size(0), -1)
            for i in range(len(labels)):
                if labels[i] == 0:
                    num_0 += 1
                    if feature_mean_0 == None:
                        feature_mean_0 = fc_features[i]
                    else:
                        feature_mean_0 = feature_mean_0 + fc_features[i]

                else:
                    num_1 += 1
                    if feature_mean_1 == None:
                        feature_mean_1 = fc_features[i]
                    else:
                        feature_mean_1 = feature_mean_1 + fc_features[i]

    # print(feature_mean)
    feature_mean_0 = feature_mean_0/num_0
    feature_mean_0=torch.unsqueeze(feature_mean_0,0)

    feature_mean_1 = feature_mean_1/num_1
    feature_mean_1=torch.unsqueeze(feature_mean_1,0)
    # print(num_0,num_1)
    # print(feature_mean_0)
    # print(feature_mean_1)
    # print(feature_mean_0.size())
    # print(feature_mean_1.size())
    # exit()
    # print(feature_mean)
    # print(feature_mean.size())
    # print(feature_list.shape)

    sum = 0

    bug = 0
    with torch.no_grad():
        for (image, labels) in test_loader:
            image = image.cuda()
            labels = labels.cuda()
            outputs,fc_features = model(image)
            fc_features = F.adaptive_avg_pool2d(fc_features, (1, 1)) 
            fc_features = fc_features.view(fc_features.size(0), -1)
            _, preds = torch.max(outputs.data, 1)
            prob = nn.functional.softmax(outputs.data,dim=1)
            # calculate the difficult of sample 
            for i in range(len(image)):
                bug += 1
                print(bug)
                if outputs[i][0].item() <= -40:
                    prob[i][0] = 1e-40
                if outputs[i][1].item() <= -40:
                    print("yes!!")
                    prob[i][1] = 1e-40

                image_confidence.append(max(prob[i][0].item(),prob[i][1].item()))
                image_margin.append(max(prob[i][0].item(),prob[i][1].item()) - min(prob[i][0].item(),prob[i][1].item()))
                print(outputs[i][0].item(),outputs[i][1].item())
                print(prob[i][0].item(),prob[i][1].item())
                image_entropy.append(- (prob[i][0].item() * math.log(prob[i][0].item()) + prob[i][1].item() * math.log(prob[i][1].item())) )
                if(prob[i][0]>prob[i][1]):
                    image_pred.append(0)
                else:
                    image_pred.append(1)
                
                now_features=torch.unsqueeze(fc_features[i],0)
                # print(feature_mean.size())
                # print(now_features.size())
                # exit()
                if labels[i] == 0:
                    distance = F.pairwise_distance(feature_mean_0, now_features, p=2)
                else :
                    distance = F.pairwise_distance(feature_mean_1, now_features, p=2)
                dis.append(distance)
                
            corrects += torch.sum(preds == labels.data).to(torch.float32)
            print('Iteration Acc {:.4f}'.format(torch.sum(preds == labels.data).to(torch.float32)/batch_size))
        acc = corrects / test_dataset_size
        print('Test Acc: {:.4f}'.format(acc))
    print(len(test_loader))
    print(len(res))
    print(len(image_confidence))
    print(len(image_margin))
    print(len(image_entropy))
    print(len(image_pred))
    print(len(image_label))
    dict = {'image_info': res, 
            'image_confidence': image_confidence, 
            'image_margin': image_margin, 
            'image_entropy':image_entropy, 
            'image_pred':image_pred,
            'image_label':image_label,
            'dis2mean':dis,
            }
    df = pd.DataFrame(dict)
 
    #保存 dataframe
    df.to_csv('20230510_Task3_DFD_by_all(T=10)_img_info.csv')



if __name__ == '__main__':
    parse = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parse.add_argument('--batch_size', '-bz', type=int, default=256)
    #parse.add_argument('--test_list', '-tl', type=str, default='./data_list/Deepfakes_c0_test.txt')
    parse.add_argument('--test_list', '-tl', type=str, default='2500_real_and_2500_fake_train.txt')
    #parse.add_argument('--model_path', '-mp', type=str, default='./pretrained_model/df_c0_best.pkl')
    parse.add_argument('--model_path', '-mp', type=str, default='20230510_Task3_DFD_SCL_KD(T10)_FD1_Memory/best.pkl')
    
    main()

    print('Hello world!!!')