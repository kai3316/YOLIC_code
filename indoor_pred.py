#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools

import cv2
from torchvision.models import mobilenet_v2
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import random
from PIL import Image
import argparse
import numpy as np
import torch
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
import torch.nn as nn
import os.path
import matplotlib.pyplot as plt
import os

parser = argparse.ArgumentParser(description='PyTorch Training Script')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test_batch', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 64)')
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
torch.cuda.empty_cache()
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

NumCell = 30  # number of cells
NumClass = 6  # number of classes except background class

polygonList = [[0, 0, 151, 0, 52, 54, 0, 54], [151, 0, 254, 0, 191, 54, 52, 54], [254, 0, 359, 0, 334, 54, 191, 54],
               [359, 0, 424, 0, 424, 54, 334, 54], [489, 0, 424, 0, 424, 54, 514, 54],
               [594, 0, 489, 0, 514, 54, 657, 54], [697, 0, 594, 0, 657, 54, 796, 54],
               [848, 0, 697, 0, 796, 54, 848, 54], [0, 54, 52, 54, 0, 82], [52, 54, 191, 54, 133, 102, 0, 102, 0, 82],
               [191, 54, 334, 54, 312, 102, 133, 102], [334, 54, 424, 54, 424, 102, 312, 102],
               [514, 54, 424, 54, 424, 102, 536, 102], [657, 54, 514, 54, 536, 102, 715, 102],
               [796, 54, 657, 54, 715, 102, 848, 102, 848, 82], [848, 54, 796, 54, 848, 82], [0, 102, 133, 102, 0, 212],
               [133, 102, 312, 102, 277, 178, 266, 179, 255, 180, 244, 181, 234, 182, 223, 183, 213, 184, 203, 185, 193,
                186, 183, 187, 173, 188, 163, 189, 153, 190, 145, 191, 137, 192, 129, 193, 121, 194, 113, 195, 106, 196,
                99, 197, 93, 198, 86, 199, 80, 200, 74, 201, 68, 202, 62, 203, 57, 204, 51, 205, 46, 206, 41, 207, 36,
                208, 31, 209, 27, 210, 22, 211, 17, 212, 13, 213, 8, 214, 3, 215, 0, 216, 0, 212],
               [312, 102, 424, 102, 424, 178, 277, 178], [536, 102, 424, 102, 424, 178, 571, 178],
               [715, 102, 536, 102, 571, 178, 582, 179, 593, 180, 604, 181, 614, 182, 625, 183, 635, 184, 645, 185, 655,
                186, 665, 187, 675, 188, 685, 189, 695, 190, 703, 191, 711, 192, 719, 193, 727, 194, 735, 195, 742, 196,
                749, 197, 755, 198, 762, 199, 768, 200, 774, 201, 780, 202, 786, 203, 791, 204, 797, 205, 802, 206, 807,
                207, 812, 208, 817, 209, 821, 210, 826, 211, 831, 212, 835, 213, 840, 214, 845, 215, 848, 216, 848,
                212], [848, 102, 715, 102, 848, 212],
               [277, 178, 266, 179, 255, 180, 244, 181, 234, 182, 223, 183, 213, 184, 203, 185, 193, 186, 183, 187, 173,
                188, 163, 189, 153, 190, 145, 191, 137, 192, 129, 193, 121, 194, 113, 195, 106, 196, 99, 197, 93, 198,
                86, 199, 80, 200, 74, 201, 68, 202, 62, 203, 57, 204, 51, 205, 46, 206, 41, 207, 36, 208, 31, 209, 27,
                210, 22, 211, 17, 212, 13, 213, 8, 214, 3, 215, 0, 216, 0, 338, 203, 338],
               [277, 178, 424, 178, 424, 338, 203, 338], [571, 178, 424, 178, 424, 338, 645, 338],
               [571, 178, 582, 179, 593, 180, 604, 181, 614, 182, 625, 183, 635, 184, 645, 185, 655, 186, 665, 187, 675,
                188, 685, 189, 695, 190, 703, 191, 711, 192, 719, 193, 727, 194, 735, 195, 742, 196, 749, 197, 755, 198,
                762, 199, 768, 200, 774, 201, 780, 202, 786, 203, 791, 204, 797, 205, 802, 206, 807, 207, 812, 208, 817,
                209, 821, 210, 826, 211, 831, 212, 835, 213, 840, 214, 845, 215, 848, 216, 848, 338, 645, 338],
               [0, 338, 203, 338, 137, 480, 0, 480], [203, 338, 424, 338, 424, 480, 137, 480],
               [645, 338, 424, 338, 424, 480, 711, 480], [848, 338, 645, 338, 711, 480, 848, 480]]
model = mobilenet_v2()  # load the model
model.classifier[1] = nn.Linear(1280, NumCell * (NumClass + 1))
model.load_state_dict(torch.load("./weights/mobilenet_indoor.pth.tar"))

save_name = 'mobilenet_indoor'  # name of the model
class_names = ["Sofa", "Wall", "Pillar", "People", "Door", "Others", "Road", "Background"]
binary_class_names = ["Risk", "Road"]



val_test_trans = transforms.Compose(([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # divides by 255
]))


class MultiLabelRGBataSet(torch.utils.data.Dataset):
    def __init__(self, imgspath, imgslist, annotationpath):
        self.imgslist = imgslist
        self.imgspath = imgspath
        self.annotationpath = annotationpath

    def __len__(self):
        return len(self.imgslist)

    def __getitem__(self, index):
        ipath = os.path.join(self.imgspath, self.imgslist[index])
        img = cv2.imread(ipath)
        (filename, extension) = os.path.splitext(ipath)
        filename = os.path.basename(filename)
        annotation = os.path.join(self.annotationpath, filename + ".txt")
        label = np.loadtxt(annotation, dtype=np.int64)
        label = torch.tensor(label, dtype=torch.float32)
        return img, label, filename


img_dir = 'images'
label_dir = 'labels'
img_list = os.listdir(img_dir)
train_img, Val_Test = train_test_split(img_list, test_size=0.3, random_state=2)
val_img, test_img = train_test_split(Val_Test, test_size=0.6666, random_state=2)

testSet = MultiLabelRGBataSet(img_dir, test_img, label_dir)

def polygon_center(points):
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    length = len(points)
    area = 0.5 * sum(x[i-1]*y[i] - x[i]*y[i-1] for i in range(length))
    center_x = sum((x[i-1] + x[i]) * (x[i-1]*y[i] - x[i]*y[i-1]) for i in range(length)) / (6*area)
    center_y = sum((y[i-1] + y[i]) * (x[i-1]*y[i] - x[i]*y[i-1]) for i in range(length)) / (6*area)
    return int(center_x), int(center_y)

def pred_plot(frame, original, output):
    orig = original.detach().numpy()
    output = output.detach().numpy()
    pred = np.where(output > 0.5, 1, 0).tolist()
    cell = 0
    normal = np.asarray([0] * NumClass + [1])
    for polygon in polygonList:
        points = [(x, y) for x, y in zip(polygon[0::2], polygon[1::2])]
        # each = pred[cell:cell + NumClass + 1]
        eachScore = output[cell:cell + NumClass + 1]
        each = orig[cell:cell + NumClass + 1]
        if not (each == normal).all():
            index = [i for i, x in enumerate(each) if x == 1]
            if len(index) == 0:
                index.append(eachScore.argmax())
                if eachScore.argmax() == NumClass:
                    continue
            center_x, center_y = polygon_center(points)
            poly_area = cv2.contourArea(np.array(points, np.int32))
            default_text_scale = 0.4  # 这是默认的字体大小，可以根据你的需要进行调整
            texts = []  # 用于存储所有的文本和对应的大小
            max_text_len = len(index)  # 计算最长的文本长度
            if max_text_len > 1:
                text_scale = default_text_scale * min(1, np.sqrt(poly_area) / max_text_len)
            else:
                text_scale = min(max(poly_area / 10000, 0.3), 0.6)
            for i in index:
                text_size, _ = cv2.getTextSize(class_names[i], cv2.FONT_HERSHEY_SIMPLEX, text_scale, 2)
                texts.append((class_names[i], text_size, text_scale))
            text_origin = [center_x, center_y - sum(text[1][1] for text in texts) // 2]
            line_spacing = 0.7  # 行间距，可以根据需要调整
            for text, text_size, text_scale in texts:
                text_origin[0] = center_x - text_size[0] // 2  # 每行的x坐标需要重新计算以保证居中
                text_origin[1] += int(text_size[1] * line_spacing) # y坐标加上当前行文本的高度的一部分
                cv2.putText(frame, text, tuple(text_origin), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (255, 255, 255), 1)
                text_origin[1] += int(text_size[1] * line_spacing)  # y坐标再加上当前行文本的高度的一部分，为下一行文本做准备
            cv2.polylines(frame, [np.array(points, np.int32)], isClosed=True, color=(0, 255, 255), thickness=3)
        cell += NumClass + 1
    # cv2.imshow("frame", frame)
    # cv2.waitKey(0)
    return frame


preprocess = transforms.Compose([transforms.ToTensor()])


def test():
    model.eval()
    with torch.no_grad():
        for batch_idx, (image, target, filename) in enumerate(testSet):
            resize_image = cv2.resize(image, (224, 224))
            input_tensor = preprocess(resize_image)
            input_batch = input_tensor.unsqueeze(0)
            output = model(input_batch)
            output = torch.sigmoid(output)
            frame = pred_plot(image, torch.Tensor.cpu(target), output[0])
            cv2.imwrite(os.path.join(path, filename + ".jpg"), frame)


current_path = os.getcwd()
path = os.path.join(current_path, save_name)
if not os.path.exists(path):
    os.makedirs(path)
test()


