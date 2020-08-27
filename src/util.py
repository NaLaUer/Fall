from tqdm import tqdm
import os
import cv2
import numpy as np

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # __file__获取执行文件相对路径，整行为取上一级的上一级目录
sys.path.append(BASE_DIR)


from lrcn_net import ConvLstm
from dataset import VideoDataset

def prepare_dataset(args):
    print('Training model on {} dataset...'.format(args.dataset))
    train_data_loader = DataLoader(VideoDataset(args=args, split='train'),
                                   batch_size=args.batch_size, shuffle=True,
                                   num_workers=args.num_workers, drop_last=True)
    val_data_loader = DataLoader(VideoDataset(args=args, split='val'),
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers, drop_last=True)
    test_data_loader = DataLoader(VideoDataset(args=args, split='test'),
                                  batch_size=args.batch_size,
                                  num_workers=args.num_workers, drop_last=True)

    train_val_loaders = {'train': train_data_loader, 'val': val_data_loader}
    train_val_sizes = {x: len(train_val_loaders[x].dataset) for x in ['train', 'val']}
    test_size = len(test_data_loader.dataset)
    return train_val_loaders, train_val_sizes, test_data_loader, test_size


def build (args):
    model = ConvLstm(latent_dim=args.latent_dim, hidden_size=args.hidden_size,
                     lstm_layers=args.lstm_layers, bidirectional=args.bidirectional,
                     n_class=args.num_classes, args=args)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=4e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    model.cuda()
    criterion.cuda()
    return model, criterion, scheduler, optimizer

def train_model(model, criterion, scheduler, optimizer, data_loader, data_size):
    running_loss = 0.0
    running_corrects = 0.0

    scheduler.step()
    model.train()

    for inputs, labels in tqdm(data_loader):
        inputs = inputs.cuda()
        labels = labels.cuda()

        optimizer.zero_grad()
        model.Lstm_model.resnet_hidden_state()
        outputs = model(inputs)

        probs = nn.Softmax(dim=1)(outputs)
        preds = torch.max(probs, 1)[1]

        loss = criterion(outputs, labels.long())
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / data_size
    epoch_acc = running_corrects.double() / data_size
    return epoch_loss, epoch_acc, model, optimizer

def test_model(model, criterion, scheduler, optimizer, data_loader, data_size):
    running_loss = 0.0
    running_corrects = 0.0
    model.eval()
    for inputs, labels in tqdm(data_loader):
        inputs = inputs.cuda()
        labels = labels.cuda()

        optimizer.zero_grad()
        model.Lstm_model.resnet_hidden_state()

        with torch.no_grad():
            outputs = model(inputs)

        probs = nn.Softmax(dim=1)(outputs)
        preds = torch.max(probs, 1)[1]

        loss = criterion(outputs, labels.long())

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / data_size
    epoch_acc = running_corrects.double() / data_size
    return epoch_loss, epoch_acc

def save_model (args, epoch, model, optimizer):
    # 保存模型
    save_dir_root = os.path.dirname(os.path.abspath(__file__))
    save_dir_root = os.path.dirname(save_dir_root)
    save_dir = os.path.join(save_dir_root, 'save')

    if epoch % args.snapshot == 0 and epoch != 0:
        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'opt_dict': optimizer.state_dict(),
        }, os.path.join(save_dir, args.dataset + '_epoch-' + str(epoch) + '.pth.tar'))
        print("Save model at {}\n".format(
            os.path.join(save_dir, args.dataset + '_epoch-' + str(epoch) + '.pth.tar')))


def CenterCrop(frame, size):
    h, w = np.shape(frame)[0:2]
    th, tw = size
    x1 = int(round((w - tw) / 2.))
    y1 = int(round((h - th) / 2.))

    frame = frame[y1:y1 + th, x1:x1 + tw, :]
    return np.array(frame).astype(np.uint8)


def center_crop(frame):
    return np.array(frame).astype(np.uint8)


def load_model (args, checkpoint):
    model = ConvLstm(latent_dim=args.latent_dim, hidden_size=args.hidden_size,
                     lstm_layers=args.lstm_layers, bidirectional=args.bidirectional,
                     n_class=args.num_classes,args=args)
    checkpoint = torch.load(
        checkpoint,
        map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    model.cuda()
    model.eval()

    return model

def video_reading (model, video, dir_path):
    cap = cv2.VideoCapture(video)
    retaining = True
    # 该参数是MPEG-4编码类型，后缀名为avi
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output = os.path.join(dir_path, 'output.avi')

    out = cv2.VideoWriter(output, fourcc, 20.0, (320, 240))
    clip = []
    while retaining:
        retaining, frame = cap.read()
        if not retaining and frame is None:
            continue

        tmp_ = center_crop(cv2.resize(frame, (224, 224)))
        clip.append(tmp_)
        if len(clip) == 1:
            inputs = np.array(clip).astype(np.float32)
            inputs = np.expand_dims(inputs, axis=0)
            inputs = np.transpose(inputs, (0, 1, 4, 2, 3))
            inputs = torch.from_numpy(inputs)
            inputs = inputs.cuda()
            with torch.no_grad():
                outputs = model.forward(inputs)
            probs = torch.nn.Softmax(dim=1)(outputs)
            label = torch.max(probs, 1)[1].detach().cpu().numpy()[0]

            if label == 1:
                label_string = 'normal'
            else:
                label_string = 'fall'

            cv2.putText(frame, "fall Prob: %.4f" % probs[0][0], (10, 230),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (138, 43, 226), 2)
            clip.pop(0)

        if retaining == True:
            out.write(frame)
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()