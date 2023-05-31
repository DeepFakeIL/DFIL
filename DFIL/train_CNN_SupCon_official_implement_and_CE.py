
from __future__ import print_function
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


import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss



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
            fc_features = fc_features.unsqueeze(dim = 1)
            fc_features = F.normalize(fc_features, dim=2)

            loss1 = criterion(outputs, labels)

            loss2 = criterion_supcon(fc_features,labels)
            
            loss = loss2 + loss1
            # print(fc_features.shape)
            # print(labels.shape)
            print('loss1:',loss1)
            print('loss2:',loss2)

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
    parse.add_argument('--name', '-n', type=str, default='using_FF++_oversample_train_model_with_Official_SupCon_and_CE')
    
    # parse.add_argument('--train_list', '-tl' , type=str, default = './data_list/FaceSwap_c0_train.txt')
    # parse.add_argument('--val_list', '-vl' , type=str, default = './data_list/FaceSwap_c0_val.txt')
    parse.add_argument('--train_list', '-tl' , type=str, default = 'oversample_train.txt')
    parse.add_argument('--val_list', '-vl' , type=str, default = 'val.txt')
    
    parse.add_argument('--batch_size', '-bz', type=int, default=64)
    parse.add_argument('--epoches', '-e', type=int, default='20')

    # parse.add_argument('--model_name', '-mn', type=str, default='fs_c0_299.pkl')
    parse.add_argument('--model_name', '-mn', type=str, default='demo.pkl')    

    parse.add_argument('--continue_train', type=bool, default=False)

    # parse.add_argument('--model_path', '-mp', type=str, default='./output/df_xception_c0_299/1_df_c0_299.pkl')
    parse.add_argument('--model_path', '-mp', type=str, default='using_FF++_oversample_train_model_with_SupCon_without_CE/29_demo.pkl')
    
    main()
