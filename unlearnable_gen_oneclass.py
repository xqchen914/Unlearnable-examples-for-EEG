import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from typing import Optional
from tqdm import tqdm
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader, dataset
from torch.nn import functional as F
from models import LoadModel, Classifier, Discriminator, CalculateOutSize
from models import *
from utils.pytorch_utils import init_weights, weight_for_balanced_classes
import logging
import attack_lib

def adjust_learning_rate(optimizer: nn.Module, epoch: int, args):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 50:
        lr = args.lr * 0.1
    if epoch >= 100:
        lr = args.lr * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def PGD(feature_ext: nn.Module, 
        discriminator: nn.Module, x: torch.Tensor, y: torch.Tensor, eps: float,
        alpha: float, steps: int, args):
    """ PGD attack """
    device = next(feature_ext.parameters()).device
    criterion_cal = nn.CrossEntropyLoss().to(device)
    criterion_prob = nn.MSELoss().to(device)

    data_loader = DataLoader(dataset=TensorDataset(x, y),
                             batch_size=128,
                             shuffle=False,
                             drop_last=False,num_workers=3)

    feature_ext.eval()
    discriminator.eval()
    for step, (batch_x, batch_y) in enumerate(data_loader):
        batch_x = batch_x.clone().detach().to(device)
        batch_y = batch_y.clone().detach().to(device)

        # craft adversarial examples
        batch_adv_x = batch_x.clone().detach() + torch.empty_like(
            batch_x).uniform_(-eps, eps)
        for _ in range(steps):
            batch_adv_x.requires_grad = True
            with torch.enable_grad():
                loss = criterion_cal(discriminator(feature_ext(batch_adv_x)),
                                      batch_y)
            grad = torch.autograd.grad(loss,
                                       batch_adv_x,
                                       retain_graph=False,
                                       create_graph=False)[0]

            batch_adv_x = batch_adv_x.detach() + alpha * grad.detach().sign()

            # projection
            delta = torch.clamp(batch_adv_x - batch_x, min=-eps, max=eps)

            batch_adv_x = (batch_x + delta).detach()

        if step == 0: adv_x = batch_adv_x
        else: adv_x = torch.cat([adv_x, batch_adv_x], dim=0)

    return adv_x.cpu()





def adv_linf(x, s_label,linf, unlearn_class, args):
    chans, samples = x.shape[2], x.shape[3]
    # 创建一个自定义函数来实现clip_by_value

    unlearn_class = unlearn_class
    
    x_ori = x.clone()
    s_label_ori = s_label.clone()


    ind = torch.BoolTensor([True if i ==unlearn_class else False for i in s_label ])
    x = x[ind]
    s_label = s_label[ind]

    s_label[s_label==unlearn_class] = 0

    data_loader = DataLoader(dataset=TensorDataset(x,  s_label),
                             batch_size=args.batch_size,
                             drop_last=False,num_workers=3)

    fs = []
    cs = []
    n_model = args.nmodel
    for i in range(n_model):
        fs.append(EEGNet(Chans=chans,
                        Samples=samples,
                        kernLenght=64,
                        F1=4,
                        D=2,
                        F2=8,
                        dropoutRate=0.25))


        fs[i].to(args.device)
        cs.append(Discriminator(
            input_dim=CalculateOutSize(fs[i], chans, samples),
            n_subjects=3).to(args.device))

        fs[i].apply(init_weights)
        cs[i].apply(init_weights)


        
        params = []
        for _, v in fs[i].named_parameters():
            params += [{'params': v, 'lr': args.lr}]
        for _, v in cs[i].named_parameters():
            params += [{'params': v, 'lr': args.lr}]
        optimizer = optim.Adam(params, weight_decay=5e-4, lr=1e-2)
        criterion = nn.CrossEntropyLoss().to(args.device)


        logging.info('train sub model')
        for epoch in range(args.epochs):
            adjust_learning_rate(optimizer=optimizer, epoch=epoch + 1, args=args)
            fs[i].train()
            cs[i].train()
            for step, (batch_x, batch_s) in enumerate(data_loader):
                batch_x, batch_s = batch_x.to(args.device),  batch_s.to(args.device)
                optimizer.zero_grad()
                feature = fs[i](batch_x)
                s_pred = cs[i](feature)


                loss = criterion(s_pred, batch_s)
                loss.backward()
                optimizer.step()

            if (epoch + 1) % 10 == 0:
                fs[i].eval()
                cs[i].eval()
                test_loss, test_acc = eval(fs[i], cs[i], criterion,
                                        x, s_label)

                logging.info(
                    'Epoch {}/{}:  subject loss: {:.4f} subject acc: {:.2f}'
                    .format(epoch + 1, args.epochs, 
                            test_loss, test_acc))

    logging.info('gen unlearnable examples')
    # gen unlearnable examples
    for i in range(n_model):
        fs[i].eval()
        cs[i].eval()


    perturbation = torch.zeros(
        size=[1, 1, chans, samples]).to(args.device)
    perturbation = Variable(perturbation, requires_grad=True)
    nn.init.normal_(perturbation, mean=0, std=1e-3)
    optimizer = optim.Adam([perturbation], lr=1e-1)


    for epoch in range(100):
        for step, (batch_x, batch_s) in enumerate(data_loader):
            batch_x,  batch_s = batch_x.to(args.device), batch_s.to(args.device)
            
            batch_x[batch_s==0] = batch_x[batch_s==0] + torch.tanh(perturbation) * linf

            # perb_batch_x = batch_x + torch.tanh(perturbation) * linf
            loss=0
            for i in range(n_model):
                feature = fs[i](batch_x)
                s_pred = cs[i](feature)

                loss -= criterion(s_pred, batch_s)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


        if (epoch + 1) % 10 == 0:
            # feature_ext.eval()
            # discriminator.eval()
            # x = x[s_label==0] + (torch.tanh(perturbation) * linf).cpu().detach()
            test_loss, test_acc = eval(
                fs[0], cs[0], criterion,
                x[s_label==0] + (torch.tanh(perturbation) * linf).cpu().detach(), s_label)

            
            logging.info(
                'Epoch {}/{}: subject loss: {:.4f} subject acc: {:.2f} | max perturbation:{:.2f}'
                .format(epoch + 1, args.epochs, 
                        test_loss, test_acc, torch.max(torch.abs((torch.tanh(perturbation) * linf).cpu())).item()))
    

    x_ori[s_label_ori==unlearn_class] = (x[s_label==0] + (torch.tanh(perturbation) * linf).cpu()).detach()

    return x_ori#.numpy()



def unlearnable_optim_linf(x, s_label,linf, unlearn_class, args):
    chans, samples = x.shape[2], x.shape[3]
    # 创建一个自定义函数来实现clip_by_value

    unlearn_class = unlearn_class

    x_ori = x.clone()
    s_label_ori = s_label.clone()

    # if args.model == 'EEGNet':
    feature_ext = EEGNet(Chans=chans,
                    Samples=samples,
                    kernLenght=64,
                    F1=4,
                    D=2,
                    F2=8,
                    dropoutRate=0.25)
    # elif args.model == 'DeepCNN':
    #     feature_ext = DeepConvNet(Chans=chans, Samples=samples, dropoutRate=0.5)
    # elif args.model == 'ShallowCNN':
    #     feature_ext = ShallowConvNet(Chans=chans, Samples=samples, dropoutRate=0.5)


    feature_ext.to(args.device)
    discriminator = Discriminator(
        input_dim=CalculateOutSize(feature_ext, chans, samples),
        n_subjects=3).to(args.device)

    feature_ext.apply(init_weights)
    discriminator.apply(init_weights)

    x = x[s_label==unlearn_class]
    s_label = s_label[s_label==unlearn_class] - s_label[s_label==unlearn_class]
    data_loader = DataLoader(dataset=TensorDataset(x,  s_label),
                             batch_size=args.batch_size,
                             drop_last=False,num_workers=3)

    criterion = nn.CrossEntropyLoss().to(args.device)

    logging.info('gen unlearnable examples')
    # gen unlearnable examples
    feature_ext.eval()
    discriminator.eval()

    perturbation = torch.zeros(
        size=[1, 1, chans, samples]).to(args.device)
    perturbation = Variable(perturbation, requires_grad=True)
    nn.init.normal_(perturbation, mean=0, std=1e-3)
    optimizer = optim.Adam([perturbation], lr=1e-1)

    from utils.data_transform import timeconshuffle
    tcf =  timeconshuffle(args.nchu)

    for epoch in range(100):
        for step, (batch_x, batch_s) in enumerate(data_loader):
            batch_x,  batch_s = batch_x.to(args.device), batch_s.to(args.device)
            

            # perb_batch_x = batch_x + torch.tanh(perturbation) * linf
            perb_batch_x = tcf(batch_x + torch.tanh(perturbation) * linf)
            feature = feature_ext(perb_batch_x)
            s_pred = discriminator(feature)

            loss = criterion(s_pred, batch_s)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


        if (epoch + 1) % 10 == 0:
            feature_ext.eval()
            discriminator.eval()
            test_loss, test_acc = eval(
                feature_ext, discriminator, criterion,
                (x + (torch.tanh(perturbation) * linf).cpu()).detach(), s_label)

            
            logging.info(
                'Epoch {}/{}: subject loss: {:.4f} subject acc: {:.2f} | max perturbation:{:.2f}'
                .format(epoch + 1, args.epochs, 
                        test_loss, test_acc, torch.max(torch.abs((torch.tanh(perturbation) * linf).cpu())).item()))
    
    x_ori[s_label_ori==unlearn_class] = (x + (torch.tanh(perturbation) * linf).cpu()).detach()

    return x_ori#.numpy()




def eval(model1: nn.Module, model2: nn.Module, criterion: nn.Module,
         x_test: torch.Tensor, y_test: torch.Tensor):
    device = next(model1.parameters()).device
    data_loader = DataLoader(dataset=TensorDataset(x_test, y_test),
                             batch_size=128,
                             shuffle=False,
                             drop_last=False,num_workers=3)

    loss, correct = 0., 0
    labels, preds = [], []
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            out = model2(model1(x))
            pred = nn.Softmax(dim=1)(out).cpu().argmax(dim=1)
            loss += criterion(out, y).item()
            correct += pred.eq(y.cpu().view_as(pred)).sum().item()
            labels.extend(y.cpu().tolist())
            preds.extend(pred.tolist())
    loss /= len(data_loader.dataset)
    acc = correct / len(data_loader.dataset)

    return loss, acc