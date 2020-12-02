import os
import time
import math
import argparse
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torchvision import datasets, transforms

from utils import train_ct, test, get_pred
from dataset import DATASET_CUSTOM
from networks.wideresnet import Wide_ResNet
from augmentation.autoaugment import CIFAR10Policy
from augmentation.cutout import Cutout

def log(path, str):
    print(str)
    with open(path, 'a') as file:
        file.write(str)
 
def main():
    # Settings
    parser = argparse.ArgumentParser(description='PyTorch CIFAR-100')
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--dp', type=float, default=0.2, help='dropout rate')
    parser.add_argument('--aug', type=str, default='strong', help='type of data augmentation {none, standard, strong}')
    parser.add_argument('--noise_pattern', type=str, default='uniform', help='Noise pattern')
    parser.add_argument('--noise_rate', type=float, default=0.2, help='Noise rate')
    parser.add_argument('--tau', type=float, default=0.2, help='maximum discard ratio of large-loss samples')
    parser.add_argument('--e_warm', type=int, default=0, help='warm-up epochs without discarding any samples')
    parser.add_argument('--val_size', type=int, default=5000, help='size of (noisy) validation set')
    parser.add_argument('--save_model', action='store_true', default=False, help='for Saving the current Model')
    parser.add_argument('--teacher_path', type=str, default=None, help='path of the teacher model')
    parser.add_argument('--gpu_id', type=int, default=0, help='index of gpu to use')
    parser.add_argument('--test_batch_size', type=int, default=200, help='input batch size for testing')
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
    args = parser.parse_args()

    if args.teacher_path is None:
        exp_name = 'ct_cifar100_{}{:.1f}_warm{}_dp{:.1f}_aug{}_seed{}'.format(args.noise_pattern, args.noise_rate, args.e_warm, args.dp, args.aug, args.seed)
    else:
        exp_name = 'ct_cifar100_{}{:.1f}_warm{}_dp{:.1f}_aug{}_student_seed{}'.format(args.noise_pattern, args.noise_rate, args.e_warm, args.dp, args.aug, args.seed)
    logpath = '{}.txt'.format(exp_name)
    log(logpath, 'Settings: {}\n'.format(args))

    torch.manual_seed(args.seed)
    device = torch.device('cuda:'+str(args.gpu_id) if torch.cuda.is_available() else 'cpu')

    # Datasets
    root = './data/CIFAR100'
    kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
    if args.aug=='standard':
        train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    elif args.aug=='strong':
        train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4, fill=128), transforms.RandomHorizontalFlip(),
                                              CIFAR10Policy(),
                                              transforms.ToTensor(),
                                              Cutout(n_holes=1, length=16), # (https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py)
                                              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    else:
        train_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
 
    dataset = datasets.CIFAR100(root, train=True, download=True)
    data, label = dataset.data, dataset.targets
    label_noisy= list(pd.read_csv(os.path.join('./data/CIFAR100/label_noisy', args.noise_pattern+str(args.noise_rate)+'.csv'))['label_noisy'].values.astype(int))
    train_dataset = DATASET_CUSTOM(root, data[:-args.val_size], label_noisy[:-args.val_size], transform=train_transform)
    val_dataset = DATASET_CUSTOM(root, data[-args.val_size:], label_noisy[-args.val_size:], transform=test_transform)
    test_dataset = datasets.CIFAR100(root, train=False, transform=test_transform)

    if args.teacher_path is not None:
        teacher_model = Wide_ResNet(args.dp, num_classes=100).to(device)
        teacher_model.load_state_dict(torch.load(args.teacher_path))
        distill_dataset = DATASET_CUSTOM(root, data[:-args.val_size], label_noisy[:-args.val_size], transform=test_transform)
        pred = get_pred(teacher_model, device, distill_dataset, args.test_batch_size)
        log(logpath, 'distilled noise rate: {:.2f}\n'.format(1-(np.array(label[:-args.val_size])==pred).sum()/len(pred)))
        train_dataset.targets = pred
        del teacher_model

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, **kwargs)


    # Building model
    def learning_rate(lr_init, epoch):
        optim_factor = 0
        if(epoch > 160):
            optim_factor = 3
        elif(epoch > 120):
            optim_factor = 2
        elif(epoch > 60):
            optim_factor = 1
        return lr_init*math.pow(0.2, optim_factor)

    def get_keep_ratio(e, tau=args.tau, e_warm=args.e_warm):
        return 1. - tau*min(max((e-e_warm)/10, 0), 1.)
    

    model1 = Wide_ResNet(args.dp, num_classes=100).to(device)
    model2 = Wide_ResNet(args.dp, num_classes=100).to(device)

    # Training
    val_best, epoch_best, test_at_best = 0, 0, 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        optimizer1 = optim.SGD(model1.parameters(), lr=learning_rate(args.lr, epoch), momentum=0.9, weight_decay=5e-4)
        optimizer2 = optim.SGD(model2.parameters(), lr=learning_rate(args.lr, epoch), momentum=0.9, weight_decay=5e-4)
        _, train_acc1, _, train_acc2 = train_ct(args, model1, model2, optimizer1, optimizer2, device, train_loader, get_keep_ratio(epoch))
        _, val_acc1 = test(args, model1, device, val_loader)
        _, val_acc2 = test(args, model2, device, val_loader)
        _, test_acc1 = test(args, model1, device, test_loader)
        _, test_acc2 = test(args, model2, device, test_loader)
        if max(val_acc1, val_acc2)>val_best:
            index = np.argmax([val_acc1, val_acc2])
            val_best, test_at_best, epoch_best = max(val_acc1, val_acc2), [test_acc1, test_acc2][index], epoch
            if args.save_model:
                torch.save([model1.state_dict(), model2.state_dict()][index], '{}_best.pth'.format(exp_name))

        log(logpath, 'Epoch: {}/{}, Time: {:.1f}s. '.format(epoch, args.epochs, time.time()-t0))
        log(logpath, 'Train1: {:.2f}%, Val1: {:.2f}%, Test1: {:.2f}%, Train2: {:.2f}%, Val2: {:.2f}%, Test2: {:.2f}%; Val_best: {:.2f}%, Test_at_best: {:.2f}%, Epoch_best: {}\n'.format(
            100*train_acc1, 100*val_acc1, 100*test_acc1, 100*train_acc2, 100*val_acc2, 100*test_acc2, 100*val_best, 100*test_at_best, epoch_best))

# wrong order 100*train_acc1, 100*train_acc2, 100*val_acc1, 100*test_acc1, 100*val_acc2, 100*test_acc2, 100*val_best, 100*test_at_best, epoch_best

    # Saving
    if args.save_model:
        torch.save([model1.state_dict(), model2.state_dict()][index], '{}_last.pth'.format(exp_name))

    
if __name__ == '__main__':
    main()
