import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import argparse
import os
import cv2
from tqdm import tqdm
from network.models import model_selection
from network.mesonet import Meso4, MesoInception4
from dataset.transform import xception_default_data_transforms
from dataset.mydataset import MyDataset
from SupConLoss import SupConLoss
import torch.nn.functional as F


def main():
    args = parse.parse_args()
    name = args.name
    continue_train = args.continue_train
    train_list = args.train_list
    val_list = args.val_list
    epoches = args.epoches
    batch_size = args.batch_size
    model_name = args.model_name
    model_path = args.model_path
    output_path = os.path.join('./output', name)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    torch.backends.cudnn.benchmark=True
    train_dataset = MyDataset(txt_path=train_list, transform=xception_default_data_transforms['train'],get_feature = False)
    val_dataset = MyDataset(txt_path=val_list, transform=xception_default_data_transforms['val'],get_feature = False)
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size, shuffle=True, drop_last=False, num_workers=8)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=8)
    
    
    train_dataset_size = len(train_dataset)
    val_dataset_size = len(val_dataset)

    print(len(train_loader))
    #exit()

    model = model_selection(modelname='xception', num_out_classes=2, dropout=0.5)

    if continue_train:
        print('continue train path:',model_path)
        print('train_list path:',train_list)
        print('val_list path:',val_list)
        model.load_state_dict(torch.load(model_path))

    model = model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005, betas=(0.9, 0.999), eps=1e-08, weight_decay = 1e-6)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    criterion_supcon = SupConLoss()

    model = nn.DataParallel(model)
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
        for (image, labels) in train_loader:
            iter_loss = 0.0
            iter_corrects = 0.0
            image = image.cuda()
            labels = labels.cuda()
            # print(labels)
            # exit()
            optimizer.zero_grad()
            outputs,fc_features = model(image)
            _, preds = torch.max(outputs.data, 1)

            #fc_features = model.module.model.features(image)
            fc_features = F.adaptive_avg_pool2d(fc_features, (1, 1)) 
            fc_features = fc_features.view(fc_features.size(0), -1)
            # fc_features = fc_features.unsqueeze(dim = 1)
            # fc_features = F.normalize(fc_features, dim=2)

            loss1 = criterion(outputs, labels)

            loss2 = criterion_supcon(fc_features,labels)
            
            loss = loss1 + loss2
            # print(fc_features.shape)
            # print(labels.shape)
            # print('loss1:',loss1)
            # print('loss2:',loss2)

            #loss = loss1 + loss2
            loss.backward()
            optimizer.step()
            iter_loss = loss.data.item()
            train_loss += iter_loss
            iter_corrects = torch.sum(preds == labels.data).to(torch.float32)
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

                outputs,features = model(image)
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
    parse.add_argument('--name', '-n', type=str, default='20230413_train_model_with_sup_and_ce')
    
    # parse.add_argument('--train_list', '-tl' , type=str, default = './data_list/FaceSwap_c0_train.txt')
    # parse.add_argument('--val_list', '-vl' , type=str, default = './data_list/FaceSwap_c0_val.txt')
    parse.add_argument('--train_list', '-tl' , type=str, default = 'balance_train.txt')
    parse.add_argument('--val_list', '-vl' , type=str, default = 'val.txt')
    
    parse.add_argument('--batch_size', '-bz', type=int, default=160)
    parse.add_argument('--epoches', '-e', type=int, default='20')

    # parse.add_argument('--model_name', '-mn', type=str, default='fs_c0_299.pkl')
    parse.add_argument('--model_name', '-mn', type=str, default='demo.pkl')    

    parse.add_argument('--continue_train', type=bool, default=False)

    # parse.add_argument('--model_path', '-mp', type=str, default='./output/df_xception_c0_299/1_df_c0_299.pkl')
    parse.add_argument('--model_path', '-mp', type=str, default='using_FF++_oversample_train_model_with_SupCon_without_CE/29_demo.pkl')
    
    main()
