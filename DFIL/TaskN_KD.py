import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import argparse
import os
import cv2
import random
import numpy as np
from tqdm import tqdm
from network.models import model_selection
from network.mesonet import Meso4, MesoInception4
from dataset.transform import xception_default_data_transforms
from dataset.mydataset import MyDataset
from torch.nn import functional as F
from SupConLoss import SupConLoss


def loss_fn_kd(y, labels, teacher_scores, T, alpha):
    return nn.KLDivLoss()(F.log_softmax(y/T,dim=1), F.log_softmax(teacher_scores/T,dim=1)) * (T*T * 2.0 * alpha) + F.cross_entropy(y, labels) * (1. - alpha)

def loss_fn_kd_2(outputs, labels, teacher_outputs, KD_T=20, KD_alpha=0.5):
    KD_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs/KD_T,dim=1),
                             F.softmax(teacher_outputs/KD_T,dim=1) * KD_alpha*KD_T*KD_T) +\
        F.cross_entropy(outputs, labels) * (1. - KD_alpha)
    return KD_loss

def loss_FD(Student_feature, Teacher_feature):
    Student_feature = F.adaptive_avg_pool2d(Student_feature, (1, 1)) 
    Student_feature = Student_feature.view(Student_feature.size(0), -1)
    Student_feature = F.normalize(Student_feature, dim=1)

    Teacher_feature = F.adaptive_avg_pool2d(Teacher_feature, (1, 1)) 
    Teacher_feature = Teacher_feature.view(Teacher_feature.size(0), -1)
    Teacher_feature = F.normalize(Teacher_feature, dim=1)

    # loss = torch.nn.functional.mse_loss(Student_feature, Teacher_feature, reduction="none")
    # return loss.sum()

    loss = torch.nn.functional.mse_loss(Student_feature, Teacher_feature, reduction="mean")
    return loss

def loss_ConSup(fc_features,labels):
    criterion_supcon = SupConLoss()
    fc_features = F.adaptive_avg_pool2d(fc_features, (1, 1)) 
    fc_features = fc_features.view(fc_features.size(0), -1)
    loss = criterion_supcon(fc_features,labels)

    return loss
def main():
    args = parse.parse_args()
    add_memory = args.add_memory
    name = args.name
    continue_train = args.continue_train
    train_list = args.train_list
    val_list = args.val_list
    memory_list = args.memory_list
    epoches = args.epoches
    batch_size = args.batch_size
    model_name = args.model_name
    model_path = args.model_path
    output_path = os.path.join('./output', name)
    teacher_model_path = args.teacher_model_path
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    torch.backends.cudnn.benchmark=True


    train_dataset = MyDataset(txt_path=train_list, transform=xception_default_data_transforms['train'],get_feature = True)
    val_dataset = MyDataset(txt_path=val_list, transform=xception_default_data_transforms['val'],get_feature = True)
    if add_memory:
        memory_dataset = MyDataset(txt_path=memory_list, transform=xception_default_data_transforms['train'],get_feature = True)
        train_dataset = train_dataset + memory_dataset
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size, shuffle=True, drop_last=False, num_workers=8)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=8)
    train_dataset_size = len(train_dataset)
    val_dataset_size = len(val_dataset)

    print(train_dataset_size)
    print(len(train_loader))
    #exit()

    model = model_selection(modelname='xception', num_out_classes=2, dropout=0.5)  # load studnet model
    if continue_train:
        print('continue train path:',model_path)
        print('train_list path:',train_list)
        print('val_list path:',val_list)
        model.load_state_dict(torch.load(model_path))
    model = model.cuda()

    teacher_model = model_selection(modelname='xception', num_out_classes=2, dropout=0.5) # load teacher_model 
    model.load_state_dict(torch.load(teacher_model_path))
    teacher_model = teacher_model.cuda()
    teacher_model.eval()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005, betas=(0.9, 0.999), eps=1e-08, weight_decay = 1e-6)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    model = nn.DataParallel(model)
    teacher_model = nn.DataParallel(teacher_model)
    best_model_wts = model.state_dict()
    best_acc = 0.0
    iteration = 0

    for epoch in tqdm(range(epoches)):
        print('Epoch {}/{}'.format(epoch+1, epoches))
        print('-'*10)
        model.train()
        train_loss = 0.0
        train_corrects = 0.0
        val_loss = 0.0
        val_corrects = 0.0

        data = iter(train_loader)
        idx = 0
        for (image, labels) in train_loader:
            iter_loss = 0.0
            iter_corrects = 0.0

            image = image.cuda()
            labels = labels.cuda()
            optimizer.zero_grad()
            outputs,feature = model(image)
            _, preds = torch.max(outputs.data, 1)
            
            teacher_outputs,teacher_featue = teacher_model(image)
            loss_ce = criterion(outputs, labels)   # calculate CE-Loss

            loss_fd = 0.01 * loss_FD(feature,teacher_featue) # calculate FD_loss
            
            loss_kd = loss_fn_kd(outputs, labels, teacher_outputs, T=20.0, alpha=0.3)  # calculate KD_Loss
            
            # loss_consup = loss_ConSup(feature,labels)   # calculate Consup_loss
            loss_consup = 0.1 * loss_ConSup(feature,labels)   # calculate Consup_loss
            print(loss_ce.item() ,loss_fd.item() ,loss_kd.item(),loss_consup.item())

            loss = loss_ce + loss_fd + loss_kd + loss_consup 
            # loss = loss_ce  + loss_kd + loss_consup 
            # loss = loss_ce  + loss_consup
            loss.backward()
            optimizer.step()
            iter_loss = loss.data.item()
            iter_corrects = torch.sum(preds == labels.data).to(torch.float32)
            train_loss += iter_loss
            train_corrects += iter_corrects
            iteration += 1
            if not (iteration % 20):
                print('iteration {} train loss: {:.8f} Acc: {:.8f}'.format(iteration, iter_loss / batch_size, iter_corrects / batch_size))
        epoch_loss = train_loss / train_dataset_size
        epoch_acc = train_corrects / train_dataset_size
        print('epoch train loss: {:.8f} Acc: {:.8f}'.format(epoch_loss, epoch_acc))

        model.eval()
        with torch.no_grad():
            for (image, labels) in val_loader:
                image = image.cuda()
                labels = labels.cuda()
                outputs,fc_feature = model(image)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                val_loss += loss.data.item()
                val_corrects += torch.sum(preds == labels.data).to(torch.float32)
            epoch_loss = val_loss / val_dataset_size
            epoch_acc = val_corrects / val_dataset_size
            print('epoch val loss: {:.8f} Acc: {:.8f}'.format(epoch_loss, epoch_acc))
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
        scheduler.step()
        #if not (epoch % 40):
        torch.save(model.module.state_dict(), os.path.join(output_path, str(epoch) + '_' + model_name))
    print('Best val Acc: {:.4f}'.format(best_acc))
    model.load_state_dict(best_model_wts)
    torch.save(model.module.state_dict(), os.path.join(output_path, "best.pkl"))
    


if __name__ == '__main__':
    parse = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parse.add_argument('--name', '-n', type=str, default='20230425_Task2_DFDCP_with_FD')
    
    parse.add_argument('--train_list', '-tl' , type=str, default = '2500_real_and_2500_fake_train.txt')
    
    parse.add_argument('--memory_list', '-ml' , type=str, default = '2020230424_Task2_DFDC_with_1consup_and_01_fd_Memory_DFDCP_img_info_Memory copy.txt')
   
    parse.add_argument('--val_list', '-vl' , type=str, default = '2500_real_and_2500_fake_train.txt')
   
    parse.add_argument('--batch_size', '-bz', type=int, default=64)

    parse.add_argument('--epoches', '-e', type=int, default='10')

    parse.add_argument('--model_name', '-mn', type=str, default='demo.pkl')    

    parse.add_argument('--continue_train', type=bool, default=True)

    parse.add_argument('--add_memory', type=bool, default=True)
    
    parse.add_argument('--model_path', '-mp', type=str, default='20230424_Task2_DFDC_with_1consup_and_01_fd/best.pkl')
    
    parse.add_argument('--teacher_model_path', '-tmp', type=str, default='20230424_Task2_DFDC_with_1consup_and_01_fd/best.pkl')
    
    main()




