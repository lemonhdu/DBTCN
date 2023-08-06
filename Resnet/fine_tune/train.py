import os

import numpy as np
import torch

from params import *
from utils import init_seed, worker_init_fn

from dataset import ImageDataset

from torchvision_models import resnet50

from tqdm import tqdm

import torch.nn as nn


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--save',
                        action='store_true',
                        help='if set true, save the best model',
                        default=True)

    parser.add_argument('--split',
                        default=2)

    parser.add_argument('--lr',
                        type=float,
                        help='learning rate',
                        default=1e-4)

    parser.add_argument('--weight_decay',
                        type=float,
                        help='L2 weight decay',
                        default=1e-5)

    parser.add_argument('--seed',
                        type=int,
                        help='manual seed',
                        default=0)

    parser.add_argument('--num_workers',
                        type=int,
                        help='number of subprocesses for dataloader',
                        default=32)

    parser.add_argument('--gpu',
                        type=str,
                        help='id of gpu device(s) to be used',
                        default='0')

    parser.add_argument('--train_batch_size',
                        type=int,
                        help='batch size for training phase',
                        default=32)

    parser.add_argument('--test_batch_size',
                        type=int,
                        help='batch size for test phase',
                        default=32)

    parser.add_argument('--num_epochs',
                        type=int,
                        help='number of training epochs',
                        default=100)

    return parser

def get_models(args):

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    resnet_model = resnet50(num_classes=1000, pretrained='imagenet')

    if len(args.gpu.split(',')) > 1:
        resnet_model = nn.DataParallel(resnet_model)

    resnet_model = resnet_model.cuda()
    return resnet_model


def get_dataloaders(args):

    dataloaders = {}

    dataloaders['train'] = torch.utils.data.DataLoader(ImageDataset('train', args),
                                                       batch_size=args.train_batch_size,
                                                       num_workers=args.num_workers,
                                                       shuffle=True,
                                                       pin_memory=True,
                                                       worker_init_fn=worker_init_fn
                                                       )

    dataloaders['valid'] = torch.utils.data.DataLoader(ImageDataset('valid', args),
                                                       batch_size=args.train_batch_size,
                                                       num_workers=args.num_workers,
                                                       shuffle=True,
                                                       pin_memory=True,
                                                       worker_init_fn=worker_init_fn
                                                       )

    dataloaders['test'] = torch.utils.data.DataLoader(ImageDataset('test', args),
                                                      batch_size=args.test_batch_size,
                                                      num_workers=args.num_workers,
                                                      shuffle=False,
                                                      pin_memory=True,
                                                      worker_init_fn=worker_init_fn
                                                      )

    return dataloaders

def compute_loss_prob(pre_class, truth):

    criterion_ce = nn.CrossEntropyLoss(ignore_index=-100)

    loss = criterion_ce(pre_class, truth.long())

    return loss

def train_model(dataloaders, model, args):
    optimizer = torch.optim.Adam([*model.parameters()], lr=args.lr, weight_decay=args.weight_decay)
    best_cls_accu = 0.0
    save_train_log_path = r"../training_log2/training_gtea_split"+str(args.split)+".txt"
    # save_train_test_log_path = r"../training_log2/testing_gtea_split" + str(args.split) + ".txt"
    model_save_path = "../weights_gtea2/gtea_split"+str(args.split)+".pth"

    for epoch in range(args.num_epochs):
        for split in ['train', 'valid', 'test']:
            if split == 'train':
                model.train()
                torch.set_grad_enabled(True)
            else:
                model.eval()
                torch.set_grad_enabled(False)
            epoch_loss = 0.0
            correct = 0
            count = 0
            for data in tqdm(dataloaders[split]):
                image = data['image'].cuda()
                label = data['label'].cuda()

                # input_data = image.unsqueeze(0)
                input = torch.autograd.Variable(image)

                output = model(input)  # output:[1,19]
                # print(output, label)
                loss_prob = compute_loss_prob(output, label)
                epoch_loss += loss_prob

                # print(output)
                # max, argmax = output.data.squeeze().max(0)
                argmax = torch.argmax(output, dim=1)
                # print(argmax, label)
                # class_id = argmax.item()
                for i in range(len(argmax)):
                    if argmax[i] == label[i]:
                        correct += 1

                if split == 'train':
                    optimizer.zero_grad()
                    loss_prob.backward()
                    optimizer.step()

                count += 1
                correct_div = correct/args.train_batch_size
                if count % 500 == 0:
                    print(count, correct_div, correct_div/count)

            print("{:s}".format(split))
            print("epoch：{:03d}".format(epoch) + ", loss is {:.4f}".format(epoch_loss/len(dataloaders[split])))
            cls_accu = correct/len(dataloaders[split])/args.train_batch_size
            print("class accuracy is:{:.4f}".format(cls_accu))

            with open(save_train_log_path, "a") as f:
                f.write(split + "    " + str(epoch) + "    " + str(cls_accu) + "\n")
                f.close()

            if split == "valid" and cls_accu > best_cls_accu:
                best_cls_accu = cls_accu
                epoch_best = epoch
                print(
                    "find new classify model, epoch {:3d}".format(epoch_best) + ", best cls:{:.4f}".format(best_cls_accu))
                with open(save_train_log_path, "a") as f:
                    f.write("find new model"+"\n")
                    f.close()
                if args.save:
                    torch.save({'epoch': epoch_best,
                                'save_model': model.state_dict(),
                                'best_acc': best_cls_accu}, model_save_path)


if __name__ == '__main__':

    args = get_parser().parse_args()

    init_seed(args)

    trained_model = get_models(args)

    dataloaders = get_dataloaders(args)

    train_model(dataloaders, trained_model, args)


