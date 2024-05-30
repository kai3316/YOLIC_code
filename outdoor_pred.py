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
from torchvision import transforms
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

NumCell = 104  # number of cells
NumClass = 11  # number of classes except background class
model = mobilenet_v2()  # load the model
model.classifier[1] = nn.Linear(1280, NumCell * (NumClass + 1))
model.load_state_dict(torch.load("./weights/mobilenet_outdoor.pth.tar"))
save_name = 'Outdoor_mobilenet'  # name of the model
points_list = [(288, 166), (322, 166), (356, 166), (390, 166), (424, 166), (458, 166), (492, 166), (526, 166),
               (220, 200), (254, 200), (288, 200), (322, 200), (356, 200), (390, 200), (424, 200), (458, 200),
               (492, 200), (526, 200), (560, 200), (594, 200),
               (220, 234), (254, 234), (288, 234), (322, 234), (356, 234), (390, 234), (424, 234), (458, 234),
               (492, 234), (526, 234), (560, 234), (594, 234), (628, 234),
               (254, 268), (288, 268), (322, 268), (356, 268), (390, 268), (424, 268), (458, 268), (492, 268),
               (526, 268), (560, 268), (594, 268), (628, 268),
               (0, 268), (53, 268), (106, 268), (159, 268), (212, 268), (265, 268), (318, 268), (371, 268), (424, 268),
               (477, 268), (530, 268), (583, 268), (636, 268), (689, 268), (742, 268), (795, 268),
               (0, 321), (53, 321), (106, 321), (159, 321), (212, 321), (265, 321), (318, 321), (371, 321), (424, 321),
               (477, 321), (530, 321), (583, 321), (636, 321), (689, 321), (742, 321), (795, 321), (848, 321),
               (0, 374), (53, 374), (106, 374), (159, 374), (212, 374), (265, 374), (318, 374), (371, 374), (424, 374),
               (477, 374), (530, 374), (583, 374), (636, 374), (689, 374), (742, 374), (795, 374), (848, 374),
               (0, 427), (53, 427), (106, 427), (159, 427), (212, 427), (265, 427), (318, 427), (371, 427), (424, 427),
               (477, 427), (530, 427), (583, 427), (636, 427), (689, 427), (742, 427), (795, 427), (848, 427),
               (0, 480), (53, 480), (106, 480), (159, 480), (212, 480), (265, 480), (318, 480), (371, 480), (424, 480),
               (477, 480), (530, 480), (583, 480), (636, 480), (689, 480), (742, 480), (795, 480), (848, 480),
               (184, 0), (244, 0), (304, 0), (364, 0), (424, 0), (484, 0), (544, 0), (604, 0), (244, 60), (304, 60),
               (364, 60), (424, 60), (484, 60), (544, 60), (604, 60), (664, 60)]
cell_list = [[points_list[0], points_list[11]], [points_list[1], points_list[12]], [points_list[2], points_list[13]],
            [points_list[3], points_list[14]], [points_list[4], points_list[15]],
            [points_list[5], points_list[16]], [points_list[6], points_list[17]], [points_list[7], points_list[18]],
            [points_list[8], points_list[21]], [points_list[9], points_list[22]],
            [points_list[10], points_list[23]], [points_list[11], points_list[24]], [points_list[12], points_list[25]],
            [points_list[13], points_list[26]], [points_list[14], points_list[27]],
            [points_list[15], points_list[28]], [points_list[16], points_list[29]], [points_list[17], points_list[30]],
            [points_list[18], points_list[31]], [points_list[19], points_list[32]],
            [points_list[20], points_list[33]], [points_list[21], points_list[34]], [points_list[22], points_list[35]],
            [points_list[23], points_list[36]], [points_list[24], points_list[37]],
            [points_list[25], points_list[38]], [points_list[26], points_list[39]], [points_list[27], points_list[40]],
            [points_list[28], points_list[41]], [points_list[29], points_list[42]],
            [points_list[30], points_list[43]], [points_list[31], points_list[44]], [points_list[45], points_list[62]],
            [points_list[46], points_list[63]], [points_list[47], points_list[64]],
            [points_list[48], points_list[65]], [points_list[49], points_list[66]], [points_list[50], points_list[67]],
            [points_list[51], points_list[68]], [points_list[52], points_list[69]],
            [points_list[53], points_list[70]], [points_list[54], points_list[71]], [points_list[55], points_list[72]],
            [points_list[56], points_list[73]], [points_list[57], points_list[74]],
            [points_list[58], points_list[75]], [points_list[59], points_list[76]], [points_list[60], points_list[77]],
            [points_list[61], points_list[79]], [points_list[62], points_list[80]],
            [points_list[63], points_list[81]], [points_list[64], points_list[82]], [points_list[65], points_list[83]],
            [points_list[66], points_list[84]], [points_list[67], points_list[85]],
            [points_list[68], points_list[86]], [points_list[69], points_list[87]], [points_list[70], points_list[88]],
            [points_list[71], points_list[89]], [points_list[72], points_list[90]],
            [points_list[73], points_list[91]], [points_list[74], points_list[92]], [points_list[75], points_list[93]],
            [points_list[76], points_list[94]], [points_list[78], points_list[96]],
            [points_list[79], points_list[97]], [points_list[80], points_list[98]], [points_list[81], points_list[99]],
            [points_list[82], points_list[100]], [points_list[83], points_list[101]],
            [points_list[84], points_list[102]], [points_list[85], points_list[103]],
            [points_list[86], points_list[104]], [points_list[87], points_list[105]],
            [points_list[88], points_list[106]],
            [points_list[89], points_list[107]], [points_list[90], points_list[108]],
            [points_list[91], points_list[109]], [points_list[92], points_list[110]],
            [points_list[93], points_list[111]],
            [points_list[95], points_list[113]], [points_list[96], points_list[114]],
            [points_list[97], points_list[115]], [points_list[98], points_list[116]],
            [points_list[99], points_list[117]],
            [points_list[100], points_list[118]], [points_list[101], points_list[119]],
            [points_list[102], points_list[120]], [points_list[103], points_list[121]],
            [points_list[104], points_list[122]],
            [points_list[105], points_list[123]], [points_list[106], points_list[124]],
            [points_list[107], points_list[125]], [points_list[108], points_list[126]],
            [points_list[109], points_list[127]],
            [points_list[110], points_list[128]], [points_list[129], points_list[137]],
            [points_list[130], points_list[138]], [points_list[131], points_list[139]],
            [points_list[132], points_list[140]],
            [points_list[133], points_list[141]], [points_list[134], points_list[142]],
            [points_list[135], points_list[143]], [points_list[136], points_list[144]]]
class_names = ["Bump", "Column", "Dent", "Fence", "Creature", "Vehicle", "Wall", "Weed", "ZebraCrossing", "TrafficCone",
               "TrafficSign", "Background"]
color_box = [(10, 249, 72), (151, 157, 255), (134, 219, 61), (52, 147, 26), (29, 178, 255), (31, 112, 255), (49, 210, 207),
         (23, 204, 146), (56, 56, 255), (187, 212, 0), (168, 153, 44)]
def random_augmentation(image, label_list, seq_list):
    # flip image horizontally
    image = image.flip(1)
    n_groups = len(seq_list)
    n_labels = len(label_list)
    assert n_labels % n_groups == 0  # make sure it's evenly divisible

    group_size = n_labels // n_groups

    # divide the label_list into groups based on seq_list
    label_groups = []
    start_idx = 0
    for group_idx in seq_list:
        end_idx = start_idx + group_size
        label_groups.append(label_list[start_idx:end_idx])
        start_idx = end_idx

    # create a new label_list based on seq_list
    new_label_list = []
    for group_idx in seq_list:
        group = label_groups[group_idx]
        new_label_list.extend(group)

    return image, new_label_list
class MultiLabelRGBataSet(torch.utils.data.Dataset):
    def __init__(self, imgspath, imgslist, annotationpath, transforms=None, train=1):
        self.imgslist = imgslist
        self.imgspath = imgspath
        self.transform = transforms
        self.annotationpath = annotationpath
        self.train = train

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
label_dir = 'yoliclabel'
img_list = os.listdir(img_dir)
train_img, Val_Test = train_test_split(img_list, test_size=0.3, random_state=2)
val_img, test_img = train_test_split(Val_Test, test_size=0.6666, random_state=2)
test_dataset = MultiLabelRGBataSet(img_dir, test_img, label_dir, train=0)
def pred_plot(frame, original, output):
    orig = original.detach().numpy()
    output = output.detach().numpy()
    pred = np.where(output > 0.5, 1, 0).tolist()
    cell = 0
    normal = np.asarray([0] * NumClass + [1])
    for rect in cell_list:
        x1, y1 = rect[0]
        x2, y2 = rect[1]
        # cv2.rectangle(frame, tuple(rect[0]), tuple(rect[1]), color=(0, 0, 0), thickness=3)
        each = pred[cell:cell + NumClass + 1]
        eachScore = output[cell:cell + NumClass + 1]
        # each = orig[cell:cell + NumClass + 1]
        if not (each == normal).all():
            index = [i for i, x in enumerate(each) if x == 1]
            if len(index) == 0:
                index.append(eachScore.argmax())
                if eachScore.argmax() == NumClass:
                    continue
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2  # 计算矩形中心点
            poly_area = (x2 - x1) * (y2 - y1)  # 计算矩形面积
            default_text_scale = 0.4  # 这是默认的字体大小，可以根据你的需要进行调整
            texts = []  # 用于存储所有的文本和对应的大小
            max_text_len = len(index)  # 计算最长的文本长度
            if max_text_len > 1:
                text_scale = default_text_scale * min(1, np.sqrt(poly_area) / max_text_len)
            else:
                text_scale = min(max(poly_area / 10000, 0.3), 0.6)
            for i in index:
                if i == NumClass:
                    continue
                text_size, _ = cv2.getTextSize(class_names[i], cv2.FONT_HERSHEY_SIMPLEX, text_scale, 2)
                texts.append((class_names[i], text_size, text_scale))
            text_origin = [center_x, center_y - sum(text[1][1] for text in texts) // 2]
            line_spacing = 0.7  # 行间距，可以根据需要调整
            color = color_box[index[0]]
            for text, text_size, text_scale in texts:
                text_origin[0] = center_x - text_size[0] // 2  # 每行的x坐标需要重新计算以保证居中
                text_origin[1] += int(text_size[1] * line_spacing)  # y坐标加上当前行文本的高度的一部分
                # cv2.putText(frame, text, tuple(text_origin), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (255, 255, 255), 1)
                text_origin[1] += int(text_size[1] * line_spacing)  # y坐标再加上当前行文本的高度的一部分，为下一行文本做准备
            cv2.rectangle(frame, tuple(rect[0]), tuple(rect[1]), color=color, thickness=3)
        cell += NumClass + 1
    return frame


preprocess = transforms.Compose([transforms.ToTensor()])


def test():
    model.eval()
    with torch.no_grad():
        for batch_idx, (image, target, filename) in enumerate(test_dataset):
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

