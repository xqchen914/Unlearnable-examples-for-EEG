import os
import logging
import torch
import argparse
import datetime
import numpy as np
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix
import openpyxl
import pandas as pd
from pandas import Series, DataFrame
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from models import LoadModel, Classifier, CalculateOutSize
import attack_lib
from utils.data_loader import *
from utils.pytorch_utils import init_weights, print_args, seed, weight_for_balanced_classes, bca_score, CustomTensorDataset
import train
from unlearnable_gen_batclass import  unlearnable_optim_linf, unlearnable_optim_l2, unlearnable_optim_l2_linf, adv_linf, adv_l2, adv_l2_linf
from scipy.signal import square
import pandas as pd
import random
from utils import data_transform

def run(x: torch.Tensor, y: torch.Tensor, s_train: torch.Tensor,x_test: torch.Tensor,
        y_test: torch.Tensor,s_test: torch.Tensor, tra,unlearnable_class, args):
    #====================== initialize the model ======================
    args.num_classes = len(np.unique(y.numpy()))
    modelF = LoadModel(model_name=args.model,
                            Chans=x.shape[2],
                            Samples=x.shape[3])

    modelC = Classifier(input_dim=CalculateOutSize(modelF, x.shape[2],
                                                       x.shape[3]),
                            n_classes=args.num_classes)
    
    modelF.apply(init_weights).to(args.device)
    modelC.apply(init_weights).to(args.device)


    modelF_s = LoadModel(model_name=args.model,
                            Chans=x.shape[2],
                            Samples=x.shape[3])

    modelC_s = Classifier(input_dim=CalculateOutSize(modelF, x.shape[2],
                                                       x.shape[3]),
                            n_classes=len(np.unique(s_train.numpy())))
    
    modelF_s.apply(init_weights).to(args.device)
    modelC_s.apply(init_weights).to(args.device)


    # ======================trainable parameters======================
    params = []
    for _, v in modelF.named_parameters():
        params += [{'params': v, 'lr': args.lr}]
    for _, v in modelC.named_parameters():
        params += [{'params': v, 'lr': args.lr}]
    optimizer = optim.Adam(params, weight_decay=5e-4)


    params = []
    for _, v in modelF_s.named_parameters():
        params += [{'params': v, 'lr': args.lr}]
    for _, v in modelC_s.named_parameters():
        params += [{'params': v, 'lr': args.lr}]
    optimizer_s = optim.Adam(params, weight_decay=5e-4)

    criloss = nn.CrossEntropyLoss().to(args.device)
    usrloss = nn.CrossEntropyLoss().to(args.device)

    # ======================data loader======================
    sample_weights = weight_for_balanced_classes(y)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(
        sample_weights, len(sample_weights))
    
    transform = data_transform.timeconshuffle(num_chunks=50)
    
    # train_loader = DataLoader(dataset=CustomTensorDataset((x, y),
    #                                                           transform),
    #                           batch_size=args.batch_size,
    #                           sampler=sampler,
    #                           drop_last=False,num_workers=3)
    train_loader = DataLoader(dataset=TensorDataset(x, y),
                              batch_size=args.batch_size,
                              sampler=sampler,
                              drop_last=False,num_workers=3)
    
    test_loader = DataLoader(dataset=TensorDataset(x_test, y_test),
                             batch_size=args.batch_size,
                             shuffle=True,
                             drop_last=False,num_workers=3)
    

    sample_weights = weight_for_balanced_classes(s_train)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(
        sample_weights, len(sample_weights))
    
    # train_loader_s = DataLoader(dataset=CustomTensorDataset((x, s_train),
    #                                                           transform),
    #                           batch_size=args.batch_size,
    #                           sampler=sampler,
    #                           drop_last=False,num_workers=3)
    train_loader_s = DataLoader(dataset=TensorDataset(x, s_train),
                              batch_size=args.batch_size,
                              sampler=sampler,
                              drop_last=False,num_workers=3)
    
    test_loader_s = DataLoader(dataset=TensorDataset(x_test, s_test),
                             batch_size=args.batch_size,
                             shuffle=True,
                             drop_last=False,num_workers=3)

    # ======================train task======================
    logging.info(f'=======train classification task model=======')
    modelF, modelC = train(modelF,modelC,train_loader,test_loader,
                                        optimizer,criloss, tra,
                                        args)

    
    test_loss, test_acc, task_bca = eval(modelF,modelC, criloss,
         test_loader)
    logging.info('test_loss: {:.4f} test acc: {:.4f} test bca: {:.2f}'
                    .format(test_loss, test_acc, task_bca))

    # ======================train user======================
    logging.info(f'=======train user recognize model=======')
    modelF_s, modelC_s = train(modelF_s,modelC_s,train_loader_s,test_loader_s,
                                        optimizer_s,usrloss, tra,
                                        args, unlearnable_class)
    test_loss, test_acc, pid_bca, un_bca, clean_bca = eval(modelF_s,modelC_s, usrloss,
         test_loader_s, unlearnable_class)
    logging.info('test_loss: {:.4f} test acc: {:.4f} all test bca: {:.2f} un person bca: {:.2f} clean person bca: {:.2f}'
                    .format(test_loss, test_acc, pid_bca, un_bca, clean_bca))
    

    torch.save(modelF.state_dict(),
                    model_save_path + '/modelF.pt')
    torch.save(modelC.state_dict(),
                    model_save_path + '/modelC.pt')
    
    torch.save(modelF_s.state_dict(),
                    model_save_path + '/modelF_s.pt')
    torch.save(modelC_s.state_dict(),
                    model_save_path + '/modelC_s.pt')
    
    
    return task_bca, pid_bca, un_bca, clean_bca, modelF, modelC, modelF_s, modelC_s

def train(modelF, modelC, train_loader, test_loader, optimizer, criloss, tra, args, unlearnable_class=None):
    for epoch in range(args.epochs):
        # model training
        adjust_learning_rate(optimizer=optimizer, epoch=epoch + 1,args=args)
        modelF.train()
        modelC.train()
        for step, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(args.device), batch_y.to(
                args.device)
            
            if tra == 'ATchastd':
                    batch_x = attack_lib.PGD_batch_cha(nn.Sequential(modelF, modelC),
                                                batch_x,
                                                batch_y,
                                                eps=args.AT_eps,#e,
                                                alpha=args.AT_eps/5,#torch.round(e/5, decimals=4).to(self.device),
                                                steps=10,
                                                label_free=False)
                    modelF.train()
                    modelC.train()
            elif tra == 'ATchastd_fgsm':
                    batch_x = attack_lib.FGSM_batch_cha(nn.Sequential(modelF, modelC),
                                                batch_x,
                                                batch_y,
                                                eps=args.AT_eps,
                                                label_free=False)
                    modelF.train()
                    modelC.train()


            out = modelC(modelF(batch_x))
            loss = criloss(out, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            modelF.MaxNormConstraint()
            modelC.MaxNormConstraint()

        # if args.dataset != 'EPFL':

        #     if unlearnable_class is not None:
        #         train_loss, train_acc, train_bca, train_un_bca, train_clea_bca = eval(modelF, modelC, criloss, train_loader, unlearnable_class)
        #         test_loss, test_acc, test_bca, test_un_bca, test_clea_bca = eval(modelF, modelC, criloss, test_loader, unlearnable_class)

        #         logging.info(
        #             'Epoch {}/{}: train_loss: {:.4f} train acc: {:.4f} train bca: {:.2f} train un acc {:.2f} train clea acc {:.2f} | test_loss: {:.4f} test acc: {:.4f} test bca: {:.2f} test un acc {:.2f} test clean acc {:.2f}'
        #             .format(epoch + 1, args.epochs, train_loss, train_acc, train_bca, train_un_bca, train_clea_bca, test_loss,
        #                     test_acc, test_bca, test_un_bca, test_clea_bca))
        #     else:
        #         train_loss, train_acc, train_bca = eval(modelF, modelC, criloss, train_loader, unlearnable_class)
        #         test_loss, test_acc, test_bca = eval(modelF, modelC, criloss, test_loader, unlearnable_class)

        #         logging.info(
        #         'Epoch {}/{}: train_loss: {:.4f} train acc: {:.4f} train bca: {:.2f} | test_loss: {:.4f} test acc: {:.4f} test bca: {:.2f}'
        #         .format(epoch + 1, args.epochs, train_loss, train_acc, train_bca, test_loss,
        #                 test_acc, test_bca))
        # else:
        #     train_loss, train_acc, train_bca = eval(modelF, modelC, criloss, train_loader, unlearnable_class)
        #     logging.info(
        #         'Epoch {}/{}: train_loss: {:.4f} train acc: {:.4f} train bca: {:.2f}'
        #         .format(epoch + 1, args.epochs, train_loss, train_acc, train_bca))

            
    return modelF, modelC

def eval(model1: nn.Module, model2: nn.Module, criterion: nn.Module,
         data_loader: DataLoader, unlearnable_class=None):
    loss, correct = 0., 0
    labels, preds = [], []
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(args.device), y.to(args.device)
            out = model2(model1(x))
            pred = nn.Softmax(dim=1)(out).cpu().argmax(dim=1)
            loss += criterion(out, y).item()
            correct += pred.eq(y.cpu().view_as(pred)).sum().item()
            labels.extend(y.cpu().tolist())
            preds.extend(pred.tolist())
    loss /= len(data_loader.dataset)
    acc = correct / len(data_loader.dataset)
    bca = bca_score(labels, preds)


    if unlearnable_class is not None:

        bca_un = bca_score_partial(labels, preds, unlearnable_class.tolist())

        clean_class = np.unique(np.array(labels)).tolist()

        for i in unlearnable_class:
            clean_class.remove(i)

        bca_clean = bca_score_partial(labels, preds, clean_class)

        return loss, acc, bca, bca_un, bca_clean
    else:
        return loss, acc, bca

def bca_score_partial(y_true, y_pred, tclass):

    m = confusion_matrix(y_true, y_pred)
    acc_each_label = 0
    if len(tclass) != 0:
        for i in tclass:
            acc = m[i, i] / np.sum(m[i, :], keepdims=False).astype(np.float32)
            acc_each_label += acc
        return acc_each_label / len(tclass)
    else:
        return -1


def adjust_learning_rate(optimizer: nn.Module, epoch: int, args):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 50:
        lr = args.lr * 0.1
    if epoch >= 100:
        lr = args.lr * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model train')
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--model', type=str, default='ShallowCNN') #ShallowCNN DeepCNN EEGNet
    parser.add_argument('--dataset', type=str, default='physionet')# physionet MI2014001
    
    parser.add_argument('--premask', type=str, default='no')
    parser.add_argument('--postmask', nargs='+',default=['adv_linf'])# 0.1_['no', 'rand','square','rand_binary','rand_binary2','optim','optim_linf','optim_l2']
    parser.add_argument('--maskamp', type=float, default=0.3)#['no', 'rand', 'rand_binary2', 'optim_linf', 'adv_linf']
    parser.add_argument('--maskl2_ind', type=float, default=0.3)#['no', 'rand', 'rand_binary2']
    # ['optim_linf', 'adv_linf']
    # ['no', 'rand', 'rand_binary2']
    # ['optim_l2', 'adv_l2']

    # ['optim_linf', 'adv_linf']
    # ['no', 'rand', 'rand_binary2']
    # ['optim_l2', 'adv_l2']
    parser.add_argument('--nchu', type=int, default=5)
    parser.add_argument('--nmodel', type=int, default=3)

    parser.add_argument('--train',  nargs='+',default=['NT'])  # 0.1_['NT', 'ATchastd']
    parser.add_argument('--AT_eps', type=float,
                        default=0.1) 

    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--optim', type=str, default='Adam')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--commit', type=str, default='test')

    args = parser.parse_args()

    # 设置pandas选项以完整显示DataFrame内容
    pd.set_option('display.max_rows', None)  # 显示所有行
    pd.set_option('display.max_columns', None)  # 显示所有列
    pd.set_option('display.width', None)  # 自动调整列宽


    args.device = f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu'
    

    subject_num_dict = {'MI4C': 9, 'ERN': 16, 'EPFL': 8,'bench':35, 'tusz':154,'nicu':8, 'MI2014001':9,  'MI2014001-2':9,\
                        'MI2015001':12, 'MI2014004':9, 'MI2015004':9, 'weibo':10, 'weibocsp':10,\
                            'physionet':109, 'physionet2c':109, '2014009':10,'20140091':10}
    

    sf_dict = {'MI2014001':250, 'weibo':200 ,'ERN': 128,'bench':250,'nicu':256,  'tusz':250, 'physionet': 160,'EPFL':128, '2014009':256,'20140091':256}

    if args.dataset == 'tusz':
        args.batch_size = 256
        # args.epochs = 
    # ========================model path=======================
    model_path = f'/data1/cxq/model_id/{args.dataset}/bat_cla_{args.model}/'


    # ========================log name and excel name=======================
    log_path = f'/home/xqchen/attack_id_eegn/result/log'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_name = os.path.join(log_path,
                            f'bat_cla_{args.dataset}_{args.model}_{args.maskamp}-{args.maskl2_ind}-{args.postmask}_{args.AT_eps}-{args.train}_base.log')
    

    excel_path = f'/home/xqchen/attack_id_eegn/result/excel'
    if not os.path.exists(excel_path):
        os.makedirs(excel_path)
    excel_name = os.path.join(excel_path,f'bat_cla_{args.dataset}_{args.model}_{args.maskamp}-{args.maskl2_ind}-{args.postmask}_{args.AT_eps}-{args.train}_base.xlsx')
    
    # system time
    args.commit = datetime.datetime.now()
    log_name = log_name.replace('.log', f'_{args.commit}.log')
    excel_name = excel_name.replace('.xlsx', f'_{args.commit}.xlsx')

    recorder = np.zeros((round((args.repeat+2)*np.round(np.ceil(subject_num_dict[args.dataset]/10)).astype(np.int32)),round(4*len(args.train)*len(args.postmask))))
    recorder_pert = np.zeros((round((args.repeat+2)*np.round(np.ceil(subject_num_dict[args.dataset]/10)).astype(np.int32)),round(2*len(args.postmask))))
    
    # ========================logging========================
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # print logging
    print_log = logging.StreamHandler()
    logger.addHandler(print_log)
    # save logging
    save_log = logging.FileHandler(log_name, mode='w', encoding='utf8')
    logger.addHandler(save_log)

    logging.info(print_args(args) + '\n')

    # ========================model train========================
    for r in range(args.repeat):
        seed(r)
        for p,pm in enumerate(args.postmask):

            if args.dataset == 'MI2014001':
                x_train, y_train, s_train, x_test, y_test, s_test = MI2014001Load(premask=args.premask)
            elif args.dataset == 'physionet':
                x_train, y_train, s_train, x_test, y_test, s_test = physionetLoad(premask=args.premask)

            args.maskl2 = np.sqrt((x_train.shape[-1]*x_train.shape[-2])*(args.maskl2_ind * np.std(x_train))**2)

            logging.info(f'maskl2: {args.maskl2}')

            x_train = Variable(
                torch.from_numpy(x_train).type(torch.FloatTensor))
            y_train = Variable(
                torch.from_numpy(y_train).type(torch.LongTensor))
            s_train = Variable(
                torch.from_numpy(s_train).type(torch.LongTensor))
            x_test = Variable(torch.from_numpy(x_test).type(torch.FloatTensor))
            y_test = Variable(torch.from_numpy(y_test).type(torch.LongTensor))
            s_test = Variable(torch.from_numpy(s_test).type(torch.LongTensor))
            torch.cuda.empty_cache()

            for sub in range(np.round(np.ceil(subject_num_dict[args.dataset]/10)).astype(np.int32)):

                unlearn_class = np.unique(s_train)[:round((sub+1)*10)]

                logging.info(f'unlearn class: {unlearn_class}')

                if pm == 'rand':
                    #通过本函数可以返回一个或一组服从“0~1”均匀分布的随机样本值。随机样本取值范围是[0,1)，不包括1。 
                    template = np.random.rand(subject_num_dict[args.dataset],x_train.shape[2],x_train.shape[3]) * 2 - 1
                    l2 = 0
                    l_infs = []
                    for k in unlearn_class[round((sub)*10):round((sub+1)*10)]:
                        templ = template[k:k+1,:,:]
                        templ = torch.Tensor(templ) * torch.std(x_train[s_train==k]) * args.maskamp
                        
                        x_train[s_train==k] = x_train[s_train==k] + templ
                        
                        # templ = templ.numpy()
                        l2 += torch.sqrt(torch.sum(templ**2)).numpy()
                        l_infs.append(torch.max(torch.abs(templ)).numpy())
                    linf = np.mean(np.array(l_infs))
                    l2 = l2/1
                    logging.info('max perturbation: {:.4f}, l2: {:.4f}'.format(linf,l2))
                elif pm == 'sn':
                    cho = np.random.choice(range(1,1000),subject_num_dict[args.dataset],replace=False)
                    l2 = 0
                    l_infs = []
                    for k in unlearn_class[round((sub)*10):round((sub+1)*10)]:
                        temp = list( '{:0>10}'.format(str(bin(cho[k]))[2:]))
                        temp = [list(m*10) for m in temp]
                        temp = np.tile(np.array(temp).astype(int).reshape(-1),(x_train.shape[2],500))[:,:x_train.shape[3]] * 2 - 1
                        templ = np.tile(temp,(len(x_train[s_train==k]),1,1,1))

                        templ = torch.std(x_train[s_train==k]) * torch.Tensor(np.tile(np.random.uniform(0.5,1.5,x_train.shape[-2])[:,None],(x_train[s_train==k].shape[0],1,1,1))) * torch.Tensor(templ) * args.maskamp
                        l2 += torch.sqrt(torch.sum(templ**2,dim=(1,2,3))).mean().numpy()
                        l_infs.append(torch.max(torch.abs(templ)).numpy())
                        x_train[s_train==k] = x_train[s_train==k] + templ
                    linf = np.mean(np.array(l_infs))
                    l2 = l2/len(unlearn_class[round((sub)*10):round((sub+1)*10)])
                    logging.info('max perturbation: {:.4f}, l2: {:.4f}'.format(linf,l2))
                elif pm == 'optim_linf':
                    ind = torch.BoolTensor([True if i in torch.Tensor(unlearn_class[round((sub)*10):round((sub+1)*10)]) else False for i in s_train ])

                    templ = unlearnable_optim_linf(x_train, s_train, torch.std(x_train[ind]) * args.maskamp, unlearn_class[round((sub)*10):round((sub+1)*10)], args)

                    l2 = torch.mean(torch.sqrt(torch.sum((templ[ind]-x_train[ind]).reshape(len(x_train[ind]),-1)**2,axis=-1))).numpy()
                    linf = torch.max(torch.abs(templ[ind]-x_train[ind])).numpy()
                    
                    logging.info('max perturbation: {:.4f}, l2: {:.4f}'.format(linf,l2))
                    x_train = templ
                elif pm == 'adv_linf':
                    ind = torch.BoolTensor([True if i in torch.Tensor(unlearn_class[round((sub)*10):round((sub+1)*10)]) else False for i in s_train ])

                    templ = adv_linf(x_train, s_train, torch.std(x_train[ind]) * args.maskamp, unlearn_class[round((sub)*10):round((sub+1)*10)], args)
                    l2 = torch.mean(torch.sqrt(torch.sum((templ[ind]-x_train[ind]).reshape(len(x_train[ind]),-1)**2,axis=-1))).numpy()
                    linf = torch.max(torch.abs(templ[ind]-x_train[ind])).numpy()

                    logging.info('max perturbation: {:.4f}, l2: {:.4f}'.format(linf,l2))
                    x_train = templ
                if pm == 'no':
                    recorder_pert[r*np.round(np.ceil(subject_num_dict[args.dataset]/10)).astype(np.int32)+sub,2*p] = 0
                    recorder_pert[r*np.round(np.ceil(subject_num_dict[args.dataset]/10)).astype(np.int32)+sub,2*p+1] = 0
                else:
                    recorder_pert[r*np.round(np.ceil(subject_num_dict[args.dataset]/10)).astype(np.int32)+sub,2*p] = linf
                    recorder_pert[r*np.round(np.ceil(subject_num_dict[args.dataset]/10)).astype(np.int32)+sub,2*p+1] = l2


                logging.info(f'train: {x_train.shape},{x_train.mean()},{x_train.std()}, test: {x_test.shape},{x_test.mean()},{x_test.std()}')


                for t,tra in enumerate(args.train):
                    model_save_path = os.path.join(model_path, f'{r}/{args.maskamp}_{args.maskl2_ind}_{pm}/{args.AT_eps}_{tra}')
                    if not os.path.exists(model_save_path):
                        os.makedirs(model_save_path)
                    args.model_path = model_save_path

                    logging.info(f'********** repeat: {r}, postmask: {pm}, training: {tra} **********')

                    task_bca, pid_bca, un_bca, clean_bca, modelF,modelC,modelF_s, modelC_s = run(x_train, y_train, s_train,
                                                            x_test, y_test, s_test, tra, unlearn_class, args)
                    recorder[r*np.round(np.ceil(subject_num_dict[args.dataset]/10)).astype(np.int32)+sub, p*4*len(args.train)+t*4] = task_bca
                    recorder[r*np.round(np.ceil(subject_num_dict[args.dataset]/10)).astype(np.int32)+sub, p*4*len(args.train)+t*4+1] = pid_bca
                    recorder[r*np.round(np.ceil(subject_num_dict[args.dataset]/10)).astype(np.int32)+sub, p*4*len(args.train)+t*4+2] = un_bca
                    recorder[r*np.round(np.ceil(subject_num_dict[args.dataset]/10)).astype(np.int32)+sub, p*4*len(args.train)+t*4+3] = clean_bca

                    torch.cuda.empty_cache()
        

    recorder_df = DataFrame(recorder,
               index = pd.MultiIndex.from_product([[f'repeat_{g}' for g in range(args.repeat)]+['avg','std'],[f'add_person_{g}' for g in range(np.round(np.ceil(subject_num_dict[args.dataset]/10)).astype(np.int32))]]),
               columns = pd.MultiIndex.from_product([[g for g in args.postmask],
                                                   [g for g in args.train],['Task', 'all PID bca', 'un PID bca', 'clean PID bca']]))

    o = np.zeros_like(recorder_df.loc['avg'])
    for i in range(args.repeat):
        o += recorder_df.loc[f'repeat_{i}'].values
    o /= args.repeat
    recorder_df.loc['avg'] = o.copy()

    arrs = []
    for i in range(args.repeat):
        arrs.append(recorder_df.loc[f'repeat_{i}'].values[None,:,:])
    arrs = np.concatenate(arrs)
    recorder_df.loc['std'] = np.std(arrs,axis=0)

    recorder_pert_df = DataFrame(recorder_pert,
               index = pd.MultiIndex.from_product([[f'repeat_{g}' for g in range(args.repeat)]+['avg','std'],[f'add_person_{g}' for g in range(np.round(np.ceil(subject_num_dict[args.dataset]/10)).astype(np.int32))]]),
               columns = pd.MultiIndex.from_product([[g for g in args.postmask],['Linf', 'L2']]))

    o = np.zeros_like(recorder_pert_df.loc['avg'])
    for i in range(args.repeat):
        o += recorder_pert_df.loc[f'repeat_{i}'].values
    o /= args.repeat
    recorder_pert_df.loc['avg'] = o.copy()

    arrs = []
    for i in range(args.repeat):
        arrs.append(recorder_pert_df.loc[f'repeat_{i}'].values[None,:,:])
    arrs = np.concatenate(arrs)
    recorder_pert_df.loc['std'] = np.std(arrs,axis=0)

    logging.info(print_args(args) + '\n')
    logging.info('================================final result================================')
    logging.info(recorder_df)
    logging.info('  ')
    logging.info(recorder_pert_df)
    with pd.ExcelWriter(excel_name) as writer:
        recorder_df.to_excel(writer, sheet_name='bca')
        recorder_pert_df.to_excel(writer, sheet_name='distance')




