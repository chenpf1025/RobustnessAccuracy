import os
import math
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision import transforms

from utils import train, test, get_pred
from dataset import Clothing1M
from networks.resnet import resnet50

def log(path, str):
    print(str)
    with open(path, 'a') as file:
        file.write(str)
   
def main():
    # Settings
    parser = argparse.ArgumentParser(description='PyTorch Clothing1M')
    parser.add_argument('--batch_size', type=int, default=256, help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-3, help='init learning rate')
    parser.add_argument('--save_model', action='store_true', default=False, help='For Saving the current Model')
    parser.add_argument('--use_noisy_val', action='store_true', default=False, help='Using the noisy validation setting. By default, using the benchmark setting.')
    parser.add_argument('--init_path', type=str, default=None, help='Path of a pretrained model)')
    parser.add_argument('--teacher_path', type=str, default=None, help='Path of the teacher model')
    parser.add_argument('--soft_targets', type=bool, default=True, help='Use soft targets')
    parser.add_argument('--n_gpu', type=int, default=2, help='number of gpu to use')
    parser.add_argument('--test_batch_size', type=int, default=256, help='input batch size for testing')
    parser.add_argument('--root', type=str, default='data/Clothing1M/', help='root of dataset')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    args = parser.parse_args()

    if args.teacher_path is None:
        exp_name = 'clothing1m_batch{}_seed{}'.format(args.batch_size, args.seed)
    else:
        teacher_name = args.teacher_path.replace('models/', '')
        teacher_name = teacher_name[:teacher_name.find('_')]
        if 'net1' in args.teacher_path:
            teacher_name = teacher_name+'net1'
        elif 'net2' in args.teacher_path:
            teacher_name = teacher_name+'net2'
        if args.soft_targets:
            exp_name = 'softstudent_of_{}_clothing1m_batch{}_seed{}'.format(teacher_name, args.batch_size, args.seed)
        else:
            exp_name = 'student_of_{}_clothing1m_batch{}_seed{}'.format(teacher_name, args.batch_size, args.seed)
        if args.init_path is None:
            args.init_path = args.teacher_path

    if args.use_noisy_val:
        exp_name = 'nv_'+exp_name
    logpath = '{}.txt'.format(exp_name)
    log(logpath, 'Settings: {}\n'.format(args))

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    # soft loss
    def soft_cross_entropy(output, target):
        output = F.log_softmax(output, dim=1)
        loss = -torch.mean(torch.sum(output*target, dim=1))
        return loss

    # Datasets
    root = args.root
    num_classes = 14
    kwargs = {'num_workers': 32, 'pin_memory': True} if torch.cuda.is_available() else {}
    train_transform = transforms.Compose([transforms.Resize((256)),
                                          transforms.RandomCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.6959, 0.6537, 0.6371),(0.3113, 0.3192, 0.3214)),
                                          ])
    test_transform = transforms.Compose([transforms.Resize((256)),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.6959, 0.6537, 0.6371),(0.3113, 0.3192, 0.3214)),
                                         ])
 
    train_dataset = Clothing1M(root, mode='train', transform=train_transform, use_noisy_val=args.use_noisy_val)
    val_dataset = Clothing1M(root, mode='val', transform=test_transform, use_noisy_val=args.use_noisy_val)
    test_dataset = Clothing1M(root, mode='test', transform=test_transform, use_noisy_val=args.use_noisy_val)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    if args.teacher_path is not None:
        teacher_model = resnet50(num_classes=num_classes).to(device)
        teacher_model = torch.nn.DataParallel(teacher_model, device_ids=list(range(args.n_gpu)))
        state_dict = torch.load(args.teacher_path)
        if not list(state_dict.keys())[0][:7]=='module.' :
            state_dict = dict(('module.'+key, value) for (key, value) in state_dict.items())
        teacher_model.load_state_dict(state_dict)
        distill_dataset = Clothing1M(root, mode='train', transform=test_transform, use_noisy_val=args.use_noisy_val)
        if args.soft_targets:
            pred = get_pred(teacher_model, device, distill_dataset, args.test_batch_size, num_workers=32, output_softmax=True)
            train_criterion = soft_cross_entropy
        else:
            pred = get_pred(teacher_model, device, distill_dataset, args.test_batch_size, num_workers=32)
            train_criterion = F.cross_entropy
        train_dataset.targets = pred
        log(logpath, 'Get label from teacher {}.\n'.format(args.teacher_path))
        del teacher_model
    else:
        train_criterion = F.cross_entropy


    # Building model
    def learning_rate(lr_init, epoch):
        optim_factor = 0
        if(epoch > 5):
            optim_factor = 1
        return lr_init*math.pow(0.1, optim_factor)


    model = resnet50(pretrained=True)
    model.fc = nn.Linear(2048, num_classes)
    model = torch.nn.DataParallel(model.to(device), device_ids=list(range(args.n_gpu)))
    if args.init_path is not None:
        state_dict = torch.load(args.init_path)
        if not list(state_dict.keys())[0][:7]=='module.' :
            state_dict = dict(('module.'+key, value) for (key, value) in state_dict.items())
        model.load_state_dict(state_dict)
        _, test_acc = test(args, model, device, test_loader, criterion=F.cross_entropy)
        log(logpath, 'Initialized testing accuracy: {:.2f}\n'.format(100*test_acc)) 
    cudnn.benchmark = True # Accelerate training by enabling the inbuilt cudnn auto-tuner to find the best algorithm to use for your hardware.
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3)

    # Training
    save_every_epoch = True
    if save_every_epoch:
        vals = []
        directory = 'models/'+exp_name
        if not os.path.exists(directory):
            os.makedirs(directory)

    val_best, epoch_best, test_at_best = 0, 0, 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        lr = learning_rate(args.lr, epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        _, train_acc = train(args, model, device, train_loader, optimizer, epoch, criterion=train_criterion)
        _, val_acc = test(args, model, device, val_loader, criterion=F.cross_entropy)
        _, test_acc = test(args, model, device, test_loader, criterion=F.cross_entropy)
        if val_acc>val_best:
            val_best, test_at_best, epoch_best = val_acc, test_acc, epoch
            if args.save_model:
                torch.save(model.state_dict(), '{}_best.pth'.format(exp_name))
        if save_every_epoch:
            vals.append(val_acc)
            torch.save(model.state_dict(), '{}/epoch{}.pth'.format(directory, epoch))

        log(logpath, 'Epoch: {}/{}, Time: {:.1f}s. '.format(epoch, args.epochs, time.time()-t0))
        log(logpath, 'Train: {:.2f}%, Val: {:.2f}%, Test: {:.2f}%; Val_best: {:.2f}%, Test_at_best: {:.2f}%, Epoch_best: {}\n'.format(
            100*train_acc, 100*val_acc, 100*test_acc, 100*val_best, 100*test_at_best, epoch_best))

    if save_every_epoch:
        np.save('{}/val.npy'.format(directory), vals)


if __name__ == '__main__':
    main()
