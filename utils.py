import os
import math
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torchvision import datasets
from sklearn.mixture import GaussianMixture



""" Training/testing """
# training
def train(args, model, device, loader, optimizer, epoch, criterion=F.nll_loss):
    model.train()
    train_loss = 0
    correct = 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += data.size(0)*loss.item()
        pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
        if len(target.size())==2: # soft target
           target = target.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    return train_loss/len(loader.dataset), correct/len(loader.dataset)


# testing
def test(args, model, device, loader, top5=False, criterion=F.nll_loss):
    model.eval()
    test_loss = 0
    correct = 0
    correct_k = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            if top5:
                _, pred = output.topk(5, 1, True, True)
                correct_k += pred.eq(target.view(-1,1)).sum().item()
    if top5:
        return test_loss/len(loader.dataset), correct_k/len(loader.dataset)
    else:
        return test_loss/len(loader.dataset), correct/len(loader.dataset)

# # naive training (CE) with soft labels
# def train_soft(args, model, device, loader, optimizer, epoch, need_log_softmax=True):
#     model.train()
#     train_loss = 0
#     correct = 0
#     for data, target_soft in loader:
#         target = torch.max(target_soft, dim=1)[1]
#         data, target_soft, target = data.to(device), target_soft.to(device), target.to(device)
#         output = model(data)
#         if need_log_softmax:
#             output = F.log_softmax(output, dim=1)
#         loss = -torch.mean(torch.sum(output*target_soft, dim=1))
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         train_loss += data.size(0)*loss.item()
#         pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
#         correct += pred.eq(target.view_as(pred)).sum().item()
#     return train_loss/len(loader.dataset), correct/len(loader.dataset)

# train co-teaching
def train_ct(args, model1, model2, optimizer1, optimizer2, device, loader, keep_ratio, criterion=F.nll_loss):
    model1.train(), model2.train()
    train_loss1, train_loss2 = 0, 0
    correct1, correct2 = 0, 0
    for data, target in loader:
        n_keep = round(keep_ratio*data.size(0))
        data, target = data.to(device), target.to(device)
        output1, output2 = model1(data), model2(data)
        loss1, loss2 = criterion(output1, target, reduction='none'), criterion(output2, target, reduction='none')

        # selecting #n_keep small loss instances
        _, index1 = torch.sort(loss1.detach())
        _, index2 = torch.sort(loss2.detach())
        index1, index2 = index1[:n_keep], index2[:n_keep]

        # taking a optimization step
        optimizer1.zero_grad()
        loss1[index2].mean().backward()
        optimizer1.step()
        optimizer2.zero_grad()
        loss2[index1].mean().backward()
        optimizer2.step()

        train_loss1, train_loss2 = train_loss1+loss1.sum().item(), train_loss2+loss2.sum().item()
        pred1, pred2 = output1.argmax(dim=1, keepdim=True), output2.argmax(dim=1, keepdim=True) # get the index of the max log-probability
        correct1, correct2 = correct1+pred1.eq(target.view_as(pred1)).sum().item(), correct2+pred2.eq(target.view_as(pred2)).sum().item()
    return train_loss1/len(loader.dataset), correct1/len(loader.dataset), train_loss2/len(loader.dataset), correct2/len(loader.dataset)

# Training
def train_dividemix(args, model1, model2, optimizer, epoch, labeled_trainloader, unlabeled_trainloader):
    model1.train()
    model2.eval() #fix one network and train the other
    
    labeled_loss = 0
    unlabeled_train_iter = iter(unlabeled_trainloader)    
    num_iter = (len(labeled_trainloader.dataset)//args.batch_size)+1
    for batch_idx, (inputs_x, inputs_x2, labels_x, w_x) in enumerate(labeled_trainloader):      
        try:
            inputs_u, inputs_u2 = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2 = unlabeled_train_iter.next()                 
        batch_size = inputs_x.size(0)
        
        # Transform label to one-hot
        if len(labels_x.size())==1: # hard label, i.e., an int
            labels_x = torch.zeros(batch_size, args.num_classes).scatter_(1, labels_x.view(-1,1), 1)
        else:
            labels_x = labels_x.type(torch.FloatTensor)
        w_x = w_x.view(-1,1).type(torch.FloatTensor) 

        inputs_x, inputs_x2, labels_x, w_x = inputs_x.cuda(), inputs_x2.cuda(), labels_x.cuda(), w_x.cuda()
        inputs_u, inputs_u2 = inputs_u.cuda(), inputs_u2.cuda()

        with torch.no_grad():
            # label co-guessing of unlabeled samples
            outputs_u11 = model1(inputs_u)
            outputs_u12 = model1(inputs_u2)
            outputs_u21 = model2(inputs_u)
            outputs_u22 = model2(inputs_u2)            
            
            pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) + torch.softmax(outputs_u21, dim=1) + torch.softmax(outputs_u22, dim=1)) / 4       
            ptu = pu**(1/args.T) # temparature sharpening
            
            targets_u = ptu / ptu.sum(dim=1, keepdim=True) # normalize
            targets_u = targets_u.detach()       
            
            # label refinement of labeled samples
            outputs_x = model1(inputs_x)
            outputs_x2 = model1(inputs_x2)            
            
            px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2
            px = w_x*labels_x + (1-w_x)*px              
            ptx = px**(1/args.T) # temparature sharpening 
                    
            targets_x = ptx / ptx.sum(dim=1, keepdim=True) # normalize           
            targets_x = targets_x.detach()       
        
        # mixmatch
        l = np.random.beta(args.alpha, args.alpha)        
        l = max(l, 1-l)        
        
        all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]
        
        mixed_input = l * input_a[:batch_size*2] + (1 - l) * input_b[:batch_size*2]        
        mixed_target = l * target_a[:batch_size*2] + (1 - l) * target_b[:batch_size*2]
                
        logits = model1(mixed_input)
        
        Lx = -torch.mean(torch.sum(F.log_softmax(logits, dim=1) * mixed_target, dim=1))
        
        # regularization
        prior = torch.ones(args.num_classes)/args.num_classes
        prior = prior.cuda()        
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior*torch.log(prior/pred_mean))
    
        loss = Lx + penalty
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        labeled_loss += Lx.item()*logits.size(0)

    return labeled_loss/len(labeled_trainloader) #Labeled loss: .4f

class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))
conf_penalty = NegEntropy()

def warmup_dividemix(model, optimizer, loader):
    model.train()
    train_loss = 0
    train_penalty = 0
    for batch_idx, (inputs, targets, path) in enumerate(loader):      
        inputs, targets = inputs.cuda(), targets.cuda() 
        optimizer.zero_grad()
        outputs = model(inputs)
        if len(targets.size())==1: # hard label, i.e., an int
            loss = F.cross_entropy(outputs, targets)
        else: # soft label
            loss = -torch.mean(torch.sum(F.log_softmax(outputs, dim=1)*targets, dim=1))
        
        penalty = conf_penalty(outputs)
        L = loss + penalty       
        L.backward()  
        optimizer.step()

        train_loss += loss.item()
        train_penalty += penalty.item()

    return train_loss/len(loader), train_penalty/len(loader)


def eval_train_dividemix(args, model, epoch, eval_loader):
    model.eval()
    num_samples = args.num_batches*args.batch_size
    losses = torch.zeros(num_samples)
    paths = []
    n=0
    with torch.no_grad():
        for batch_idx, (inputs, targets, path) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda() 
            outputs = model(inputs)
            if len(targets.size())==1: # hard label, i.e., an int
                loss = F.cross_entropy(outputs, targets, reduction='none')
            else: # soft label
                loss = -torch.sum(F.log_softmax(outputs, dim=1)*targets, dim=1)

            for b in range(inputs.size(0)):
                losses[n]=loss[b] 
                paths.append(path[b])
                n+=1
            
    losses = (losses-losses.min())/(losses.max()-losses.min())    
    losses = losses.reshape(-1,1)
    gmm = GaussianMixture(n_components=2,max_iter=10,reg_covar=5e-4,tol=1e-2)
    gmm.fit(losses)
    prob = gmm.predict_proba(losses) 
    prob = prob[:,gmm.means_.argmin()]

    return prob, paths  


def get_pred(model, device, dataset, batch_size, shuffle=False,  output_softmax=False, num_workers=4):
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    preds = []
    model.eval()
    with torch.no_grad():
            for data, _ in loader:
                data = data.to(device)
                output = model(data)
                if output_softmax:
                    pred = F.softmax(output, dim=1).cpu().numpy()
                else:
                    pred = output.argmax(dim=1).cpu().numpy()
                preds.append(pred)
    return np.concatenate(preds)
                


def flip_label(y, noise_pattern, noise_rate, one_hot=False, T=None):
    """
    Input y: clean label
    Output y_noisy: noisy label
    """
    y = np.argmax(y,axis=1) if one_hot else y
    n_class = max(y)+1
    y_noisy = y.copy()
    
    for i in range(len(y)):
        if T is not None:
            y_noisy[i] = np.random.choice(n_class, p=T[y[i]])
        elif noise_pattern=='uniform':
            p1 = noise_rate/(n_class-1)*np.ones(n_class)
            p1[y[i]] = 1-noise_rate
            y_noisy[i] = np.random.choice(n_class, p=p1)
        elif noise_pattern=='pair':
            y_noisy[i] = np.random.choice([y[i],(y[i]+1)%n_class], p=[1-noise_rate,noise_rate])
        else:
            print('Unsupported noise pattern')      
    
    y_noisy = np.eye(n_class)[y_noisy] if one_hot else y_noisy
    return y_noisy


def CIFAR10_noise_gen(noise_pattern, noise_rate, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    train_dataset = datasets.CIFAR10('./data/CIFAR10', train=True, download=True)
    label = np.array(train_dataset.targets)
    if noise_pattern=='asym':
        T = np.eye(10)
        # TRUCK -> AUTOMOBILE
        T[9,9]=1-noise_rate
        T[9,1]=noise_rate
        # BIRD -> AIRPLANE
        T[2,2]=1-noise_rate
        T[2,0]=noise_rate
        # DEER -> HORSE
        T[4,4]=1-noise_rate
        T[4,7]=noise_rate
        # CAT -> DOG
        T[3,3]=1-noise_rate
        T[3,5]=noise_rate
        # DOG -> CAT
        T[5,5]=1-noise_rate
        T[5,3]=noise_rate
        label_noisy = flip_label(label, noise_pattern, noise_rate, T=T)
    else:
        label_noisy = flip_label(label, noise_pattern, noise_rate)
    pd.DataFrame.from_dict({'label':label,'label_noisy':label_noisy}).to_csv(os.path.join(save_dir, noise_pattern+str(noise_rate)+'.csv'), index=False)
	
def CIFAR100_noise_gen(noise_pattern, noise_rate, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    train_dataset = datasets.CIFAR100('./data/CIFAR100', train=True, download=True)
    label = np.array(train_dataset.targets)
    label_noisy = flip_label(label, noise_pattern, noise_rate)
    pd.DataFrame.from_dict({'label':label,'label_noisy':label_noisy}).to_csv(os.path.join(save_dir, noise_pattern+str(noise_rate)+'.csv'), index=False)

# if __name__ == "__main__":
#     CIFAR10_noise_gen('asym', 0.4, save_dir='./data/CIFAR10/label_noisy')
    

