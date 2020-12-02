import os
import random
import numpy as np
import json
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class clothing_dataset(Dataset): 
    def __init__(self, root, transform, mode, num_samples=0, pred=[], probability=[], paths=[], num_class=14, train_targets=None, use_noisy_val=False): 
        
        self.mode = mode
        self.transform = transform 
        
        if not use_noisy_val: # benchmark setting
            trainfile = os.path.join(root, "annotations/noisy_train.txt")
            valfile = os.path.join(root, "annotations/clean_val.txt")
            testfile = os.path.join(root, "annotations/clean_test.txt")
        else: # using a nnoisy validation setting, saving clean labels for training
            trainfile = os.path.join(root, "noisy_val_annotations/nv_noisy_train.txt")
            valfile = os.path.join(root, "noisy_val_annotations/nv_noisy_val.txt")
            testfile = os.path.join(root, "noisy_val_annotations/nv_clean_test.txt")

        # if given training targets, e.g., distill targets from a teacher model

        # Training impaths and targets. The map is necessary since only images paths are given for the inilization of 'labeled subset'
        impaths, targets = [], []
        with open(trainfile,'r') as f:
            for line in f.readlines():
                row = line.split(" ")
                impaths.append(root + '/' + row[0])
                targets.append(int(row[1]))
        if train_targets is not None:
            targets = train_targets

        self.train_label_map = {impaths[i]:targets[i] for i in range(len(impaths))}

        # Randomly sampling class-balanced samples from the whole the training set (at each epoch)
        if mode == 'all':
            indexs = np.arange(len(impaths))                               
            random.shuffle(indexs)
            class_num = torch.zeros(num_class) # number of samples in each class
            self.impaths, self.targets = [], []
            for i in indexs:
                label = np.argmax(targets[i]) if isinstance(targets[i], (list, np.ndarray)) else targets[i]
                if class_num[label]<(num_samples/14) and len(self.impaths)<num_samples:
                    self.impaths.append(impaths[i])
                    self.targets.append(targets[i])
                    class_num[label]+=1

        # trusted labeled subset
        elif mode == 'labeled':   
            impaths = paths 
            pred_idx = pred.nonzero()[0]
            self.impaths = [impaths[i] for i in pred_idx]                
            self.probability = [probability[i] for i in pred_idx]
            self.targets = [self.train_label_map[img_path] for img_path in self.impaths]           
            print('{} data has a size of {}'.format(self.mode,len(self.impaths)))

        # unlabeled subset
        elif mode == 'unlabeled':  
            impaths = paths 
            pred_idx = (1-pred).nonzero()[0]  
            self.impaths = [impaths[i] for i in pred_idx]                
            self.probability = [probability[i] for i in pred_idx]            
            print('{} data has a size of {}'.format(self.mode,len(self.impaths)))

        # val/test
        elif mode in ['val', 'test']:
            self.impaths, self.targets = [], []
            flist = valfile if mode=='val' else testfile
            with open(flist, 'r') as f:
                for line in f.readlines():
                    row = line.split(" ")
                    self.impaths.append(root + '/' + row[0])
                    self.targets.append(int(row[1]))

                    
    def __getitem__(self, index):
        if self.mode=='all':
            img_path = self.impaths[index]   
            img = Image.open(img_path).convert('RGB')   
            img = self.transform(img)
            target = self.targets[index] 
            return img, target, img_path 

        elif self.mode=='labeled':
            img = Image.open(self.impaths[index]).convert('RGB')    
            img1 = self.transform(img) 
            img2 = self.transform(img)
            target = self.targets[index]
            prob = self.probability[index] 
            return img1, img2, target, prob  

        elif self.mode=='unlabeled':
            img = Image.open(self.impaths[index]).convert('RGB')    
            img1 = self.transform(img) 
            img2 = self.transform(img)
            return img1, img2  
   
        elif self.mode in ['val', 'test']:
            img = Image.open(self.impaths[index]).convert('RGB')
            img = self.transform(img)
            target = self.targets[index]           
            return img, target
     
    def __len__(self):
        return len(self.impaths)


        
class clothing_dataloader():  
    def __init__(self, root, batch_size, num_batches, num_workers, use_noisy_val=False):    
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_batches = num_batches
        self.root = root
        self.use_noisy_val = use_noisy_val
                   
        self.transform_train = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),                
                transforms.Normalize((0.6959, 0.6537, 0.6371),(0.3113, 0.3192, 0.3214)),
            ]) 
        self.transform_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.6959, 0.6537, 0.6371),(0.3113, 0.3192, 0.3214)),
            ])   


    def run(self,mode,pred=[],prob=[],paths=[], train_targets=None):  

        if mode=='warmup':
            warmup_dataset = clothing_dataset(self.root,transform=self.transform_train, mode='all',num_samples=self.num_batches*self.batch_size*2,train_targets=train_targets, use_noisy_val=self.use_noisy_val)
            warmup_loader = DataLoader(dataset=warmup_dataset, batch_size=self.batch_size*2, shuffle=True, num_workers=self.num_workers)  
            return warmup_loader

        elif mode=='train':
            labeled_dataset = clothing_dataset(self.root,transform=self.transform_train, mode='labeled',pred=pred, probability=prob,paths=paths,train_targets=train_targets, use_noisy_val=self.use_noisy_val)
            labeled_loader = DataLoader(dataset=labeled_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)           
            unlabeled_dataset = clothing_dataset(self.root,transform=self.transform_train, mode='unlabeled',pred=pred, probability=prob,paths=paths, use_noisy_val=self.use_noisy_val)
            unlabeled_loader = DataLoader(dataset=unlabeled_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)   
            return labeled_loader,unlabeled_loader

        elif mode=='eval_train':
            eval_dataset = clothing_dataset(self.root,transform=self.transform_test, mode='all', num_samples=self.num_batches*self.batch_size,train_targets=train_targets, use_noisy_val=self.use_noisy_val)
            eval_loader = DataLoader(dataset=eval_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)          
            return eval_loader

        elif mode in ['val', 'test']:
            dataset = clothing_dataset(self.root,transform=self.transform_test, mode=mode, use_noisy_val=self.use_noisy_val)
            loader = DataLoader(dataset=dataset, batch_size=128, shuffle=False, num_workers=self.num_workers)             
            return loader  