#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from torch.optim.lr_scheduler import MultiStepLR
import argparse
import numpy as np
import torch
from torchvision import transforms, models
import torch.optim as optim
import torch.nn as nn
import copy
import os.path
import pandas as pd
import os
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from cityscapes import Cityscapes

parser = argparse.ArgumentParser(description='PyTorch Training Script')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=150, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=25, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--resume', type=bool, default=True, metavar='N',
                    help='resume from the last weights')

NumCell = 256  # number of cells
NumClass = 3  # number of classes
cell_list = [[[512, 320], [576, 352]], [[576, 320], [640, 352]], [[640, 320], [704, 352]], [[704, 320], [768, 352]],
             [[768, 320], [832, 352]], [[832, 320], [896, 352]], [[896, 320], [960, 352]], [[960, 320], [1024, 352]],
             [[1024, 320], [1088, 352]], [[1088, 320], [1152, 352]], [[1152, 320], [1216, 352]],
             [[1216, 320], [1280, 352]], [[1280, 320], [1344, 352]], [[1344, 320], [1408, 352]],
             [[1408, 320], [1472, 352]], [[1472, 320], [1536, 352]], [[512, 352], [576, 384]], [[576, 352], [640, 384]],
             [[640, 352], [704, 384]], [[704, 352], [768, 384]], [[768, 352], [832, 384]], [[832, 352], [896, 384]],
             [[896, 352], [960, 384]], [[960, 352], [1024, 384]], [[1024, 352], [1088, 384]],
             [[1088, 352], [1152, 384]], [[1152, 352], [1216, 384]], [[1216, 352], [1280, 384]],
             [[1280, 352], [1344, 384]], [[1344, 352], [1408, 384]], [[1408, 352], [1472, 384]],
             [[1472, 352], [1536, 384]], [[512, 384], [576, 416]], [[576, 384], [640, 416]], [[640, 384], [704, 416]],
             [[704, 384], [768, 416]], [[768, 384], [832, 416]], [[832, 384], [896, 416]], [[896, 384], [960, 416]],
             [[960, 384], [1024, 416]], [[1024, 384], [1088, 416]], [[1088, 384], [1152, 416]],
             [[1152, 384], [1216, 416]], [[1216, 384], [1280, 416]], [[1280, 384], [1344, 416]],
             [[1344, 384], [1408, 416]], [[1408, 384], [1472, 416]], [[1472, 384], [1536, 416]],
             [[512, 416], [576, 448]], [[576, 416], [640, 448]], [[640, 416], [704, 448]], [[704, 416], [768, 448]],
             [[768, 416], [832, 448]], [[832, 416], [896, 448]], [[896, 416], [960, 448]], [[960, 416], [1024, 448]],
             [[1024, 416], [1088, 448]], [[1088, 416], [1152, 448]], [[1152, 416], [1216, 448]],
             [[1216, 416], [1280, 448]], [[1280, 416], [1344, 448]], [[1344, 416], [1408, 448]],
             [[1408, 416], [1472, 448]], [[1472, 416], [1536, 448]], [[512, 448], [576, 480]], [[576, 448], [640, 480]],
             [[640, 448], [704, 480]], [[704, 448], [768, 480]], [[768, 448], [832, 480]], [[832, 448], [896, 480]],
             [[896, 448], [960, 480]], [[960, 448], [1024, 480]], [[1024, 448], [1088, 480]],
             [[1088, 448], [1152, 480]], [[1152, 448], [1216, 480]], [[1216, 448], [1280, 480]],
             [[1280, 448], [1344, 480]], [[1344, 448], [1408, 480]], [[1408, 448], [1472, 480]],
             [[1472, 448], [1536, 480]], [[512, 480], [576, 512]], [[576, 480], [640, 512]], [[640, 480], [704, 512]],
             [[704, 480], [768, 512]], [[768, 480], [832, 512]], [[832, 480], [896, 512]], [[896, 480], [960, 512]],
             [[960, 480], [1024, 512]], [[1024, 480], [1088, 512]], [[1088, 480], [1152, 512]],
             [[1152, 480], [1216, 512]], [[1216, 480], [1280, 512]], [[1280, 480], [1344, 512]],
             [[1344, 480], [1408, 512]], [[1408, 480], [1472, 512]], [[1472, 480], [1536, 512]],
             [[512, 512], [576, 544]], [[576, 512], [640, 544]], [[640, 512], [704, 544]], [[704, 512], [768, 544]],
             [[768, 512], [832, 544]], [[832, 512], [896, 544]], [[896, 512], [960, 544]], [[960, 512], [1024, 544]],
             [[1024, 512], [1088, 544]], [[1088, 512], [1152, 544]], [[1152, 512], [1216, 544]],
             [[1216, 512], [1280, 544]], [[1280, 512], [1344, 544]], [[1344, 512], [1408, 544]],
             [[1408, 512], [1472, 544]], [[1472, 512], [1536, 544]], [[512, 544], [576, 576]], [[576, 544], [640, 576]],
             [[640, 544], [704, 576]], [[704, 544], [768, 576]], [[768, 544], [832, 576]], [[832, 544], [896, 576]],
             [[896, 544], [960, 576]], [[960, 544], [1024, 576]], [[1024, 544], [1088, 576]],
             [[1088, 544], [1152, 576]], [[1152, 544], [1216, 576]], [[1216, 544], [1280, 576]],
             [[1280, 544], [1344, 576]], [[1344, 544], [1408, 576]], [[1408, 544], [1472, 576]],
             [[1472, 544], [1536, 576]], [[512, 576], [576, 608]], [[576, 576], [640, 608]], [[640, 576], [704, 608]],
             [[704, 576], [768, 608]], [[768, 576], [832, 608]], [[832, 576], [896, 608]], [[896, 576], [960, 608]],
             [[960, 576], [1024, 608]], [[1024, 576], [1088, 608]], [[1088, 576], [1152, 608]],
             [[1152, 576], [1216, 608]], [[1216, 576], [1280, 608]], [[1280, 576], [1344, 608]],
             [[1344, 576], [1408, 608]], [[1408, 576], [1472, 608]], [[1472, 576], [1536, 608]],
             [[512, 608], [576, 640]], [[576, 608], [640, 640]], [[640, 608], [704, 640]], [[704, 608], [768, 640]],
             [[768, 608], [832, 640]], [[832, 608], [896, 640]], [[896, 608], [960, 640]], [[960, 608], [1024, 640]],
             [[1024, 608], [1088, 640]], [[1088, 608], [1152, 640]], [[1152, 608], [1216, 640]],
             [[1216, 608], [1280, 640]], [[1280, 608], [1344, 640]], [[1344, 608], [1408, 640]],
             [[1408, 608], [1472, 640]], [[1472, 608], [1536, 640]], [[0, 640], [128, 704]], [[128, 640], [256, 704]],
             [[256, 640], [384, 704]], [[384, 640], [512, 704]], [[512, 640], [640, 704]], [[640, 640], [768, 704]],
             [[768, 640], [896, 704]], [[896, 640], [1024, 704]], [[1024, 640], [1152, 704]],
             [[1152, 640], [1280, 704]], [[1280, 640], [1408, 704]], [[1408, 640], [1536, 704]],
             [[1536, 640], [1664, 704]], [[1664, 640], [1792, 704]], [[1792, 640], [1920, 704]],
             [[1920, 640], [2048, 704]], [[0, 704], [128, 768]], [[128, 704], [256, 768]], [[256, 704], [384, 768]],
             [[384, 704], [512, 768]], [[512, 704], [640, 768]], [[640, 704], [768, 768]], [[768, 704], [896, 768]],
             [[896, 704], [1024, 768]], [[1024, 704], [1152, 768]], [[1152, 704], [1280, 768]],
             [[1280, 704], [1408, 768]], [[1408, 704], [1536, 768]], [[1536, 704], [1664, 768]],
             [[1664, 704], [1792, 768]], [[1792, 704], [1920, 768]], [[1920, 704], [2048, 768]], [[0, 768], [128, 832]],
             [[128, 768], [256, 832]], [[256, 768], [384, 832]], [[384, 768], [512, 832]], [[512, 768], [640, 832]],
             [[640, 768], [768, 832]], [[768, 768], [896, 832]], [[896, 768], [1024, 832]], [[1024, 768], [1152, 832]],
             [[1152, 768], [1280, 832]], [[1280, 768], [1408, 832]], [[1408, 768], [1536, 832]],
             [[1536, 768], [1664, 832]], [[1664, 768], [1792, 832]], [[1792, 768], [1920, 832]],
             [[1920, 768], [2048, 832]], [[0, 832], [128, 896]], [[128, 832], [256, 896]], [[256, 832], [384, 896]],
             [[384, 832], [512, 896]], [[512, 832], [640, 896]], [[640, 832], [768, 896]], [[768, 832], [896, 896]],
             [[896, 832], [1024, 896]], [[1024, 832], [1152, 896]], [[1152, 832], [1280, 896]],
             [[1280, 832], [1408, 896]], [[1408, 832], [1536, 896]], [[1536, 832], [1664, 896]],
             [[1664, 832], [1792, 896]], [[1792, 832], [1920, 896]], [[1920, 832], [2048, 896]], [[0, 896], [128, 960]],
             [[128, 896], [256, 960]], [[256, 896], [384, 960]], [[384, 896], [512, 960]], [[512, 896], [640, 960]],
             [[640, 896], [768, 960]], [[768, 896], [896, 960]], [[896, 896], [1024, 960]], [[1024, 896], [1152, 960]],
             [[1152, 896], [1280, 960]], [[1280, 896], [1408, 960]], [[1408, 896], [1536, 960]],
             [[1536, 896], [1664, 960]], [[1664, 896], [1792, 960]], [[1792, 896], [1920, 960]],
             [[1920, 896], [2048, 960]], [[0, 960], [128, 1024]], [[128, 960], [256, 1024]], [[256, 960], [384, 1024]],
             [[384, 960], [512, 1024]], [[512, 960], [640, 1024]], [[640, 960], [768, 1024]], [[768, 960], [896, 1024]],
             [[896, 960], [1024, 1024]], [[1024, 960], [1152, 1024]], [[1152, 960], [1280, 1024]],
             [[1280, 960], [1408, 1024]], [[1408, 960], [1536, 1024]], [[1536, 960], [1664, 1024]],
             [[1664, 960], [1792, 1024]], [[1792, 960], [1920, 1024]], [[1920, 960], [2048, 1024]]]
interested_classes = [(11, 12), (13, 14, 15, 16, 17, 18),
                      (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 25, 21, 20, 19), (0, 23, 22, 24)]

save_name = 'cityscapes_mobilenet'  # name of the model
model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)  # load the model
model.classifier[1] = nn.Linear(1280, NumCell * (NumClass + 1))
optimizer = optim.Adam(model.parameters(), lr=0.001)  # optimizer and learning rate
torch.cuda.empty_cache()
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

train_trans = transforms.Compose(([

    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    transforms.ToTensor()  # divides by 255
]))
val_test_trans = transforms.Compose(([
    transforms.Resize((224, 224)),
    transforms.ToTensor()  # divides by 255
]))
root = 'Datasets/Cityscapes'
train_dataset = Cityscapes(root, cell_list=cell_list, interested_classes=interested_classes, split='train',
                           target_type='semantic', transform=train_trans)
val_dataset = Cityscapes(root, cell_list=cell_list, interested_classes=interested_classes, split='val',
                         target_type='semantic', transform=val_test_trans)
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=True, num_workers=8)
valid_loader = torch.utils.data.DataLoader(val_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=False, num_workers=8)

if args.cuda:
    model.cuda()

criterion = nn.BCEWithLogitsLoss()
scheduler = MultiStepLR(optimizer, milestones=[100, 125], gamma=0.1)


def pred_acc(original, predicted):
    pred = torch.round(predicted).detach().numpy().astype(np.int64)
    orig = original.detach().numpy()
    pred = np.reshape(pred, (NumCell * (NumClass + 1), 1)).flatten()
    orig = np.reshape(orig, (NumCell * (NumClass + 1), 1)).flatten()
    num = 0
    enum = 0
    normal = np.asarray([0] * NumClass + [1])
    for cell in range(0, (NumCell * (NumClass + 1)), NumClass + 1):
        if (orig[cell:cell + NumClass + 1] == pred[cell:cell + NumClass + 1]).all():
            num = num + 1
        else:
            if not (orig[cell:cell + NumClass + 1] == normal).all() and not (
                    pred[cell:cell + NumClass + 1] == normal).all():
                enum = enum + 1
    return num / NumCell, (num + enum) / NumCell


def train(epoch, model):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        target = target.type_as(output)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.10f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


best_correct = -999


def evaluate(model):
    model.eval()
    running_loss = []
    running_acc = []
    running_binary = []
    global best_correct
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(train_loader):
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            target = target.type_as(output)
            loss = criterion(output, target)
            output = torch.sigmoid(output)
            acc_all = []
            acc_binary = []
            for each_image, d in enumerate(output):
                all_acc, b_acc = pred_acc(torch.Tensor.cpu(target[each_image]), torch.Tensor.cpu(d))
                acc_all.append(all_acc)
                acc_binary.append(b_acc)
            running_loss.append(loss.item())
            running_acc.append(np.asarray(acc_all).mean())
            running_binary.append(np.asarray(acc_binary).mean())
    total_batch_loss = np.asarray(running_loss).mean()
    total_batch_acc = np.asarray(running_acc).mean()
    total_batch_binary = np.asarray(running_binary).mean()
    print(
        '\n Train_loader set: total_batch_loss: {:.4f}, total imgs: {} , Acc: ({:.4f}%), Binary ACC: ({:.4f}%)\n'.format(
            total_batch_loss, len(train_loader.dataset), total_batch_acc, total_batch_binary))
    now_correct = total_batch_acc
    if best_correct < now_correct:
        best_correct = now_correct
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(best_model_wts,
                   os.path.join(os.getcwd(), save_name + ".pth.tar"))
        print("New weight!")
    return total_batch_loss, total_batch_acc


def test(model):
    model.eval()
    running_loss = []
    running_acc = []
    running_binary = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(valid_loader):
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            target = target.type_as(output)
            loss = criterion(output, target)
            output = torch.sigmoid(output)
            acc_all = []
            acc_binary = []
            for each_image, d in enumerate(output):
                all_acc, b_acc = pred_acc(torch.Tensor.cpu(target[each_image]), torch.Tensor.cpu(d))
                acc_all.append(all_acc)
                acc_binary.append(b_acc)
            running_loss.append(loss.item())
            running_acc.append(np.asarray(acc_all).mean())
            running_binary.append(np.asarray(acc_binary).mean())
    total_batch_loss = np.asarray(running_loss).mean()
    total_batch_acc = np.asarray(running_acc).mean()
    total_batch_binary = np.asarray(running_binary).mean()
    print('\nTest set: total_batch_loss: {:.4f}, total imgs: {} , Acc: ({:.4f}%), Binary Acc: ({:.4f}%)\n'.format(
        total_batch_loss, len(valid_loader.dataset), total_batch_acc, total_batch_binary))

    return total_batch_loss, total_batch_acc


if __name__ == '__main__':
    import datetime
    start_time = datetime.datetime.now()
    print(save_name)
    all_train_loss = []
    all_train_acc = []
    all_test_loss = []
    all_test_acc = []
    for epoch in range(1, args.epochs + 1):
        train(epoch, model)
        train_loss, train_acc = evaluate(model)
        test_loss, test_acc = test(model)
        all_train_acc.append(train_acc)
        all_train_loss.append(train_loss)
        all_test_loss.append(test_loss)
        all_test_acc.append(test_acc)
        scheduler.step()
    list_res = []
    for i in range(len(all_train_loss)):
        list_res.append([all_train_loss[i], all_train_acc[i], all_test_loss[i], all_test_acc[i]])

    column_name = ['train_loss', 'train_acc', 'test_loss', 'test_acc']
    csv_name = save_name + '.csv'
    xml_df = pd.DataFrame(list_res, columns=column_name)
    xml_df.to_csv(csv_name, index=None)
    end_time = datetime.datetime.now()
    print('\nTime taken: {}\n'.format(end_time - start_time))
