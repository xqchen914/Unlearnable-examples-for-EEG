import os
import logging
import torch
import argparse
import datetime
import numpy as np
import torch.nn as nn
import numpy as np
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
from unlearnable_gen import  unlearnable_optim_linf, adv_linf
import pandas as pd
import mne
from utils import data_transform

def surface_laplacian(eeg, args):
    if args.dataset == 'MI2014001':
        ch_path = f'/home/xqchen/attack_id_eegn/utils/MI4C.txt'
    elif args.dataset == 'physionet':
        ch_path = f'/home/xqchen/attack_id_eegn/utils/Phy.txt'
    elif args.dataset == 'bcimi':
        ch_path = f'/home/xqchen/attack_id_eegn/utils/bcimi.txt'
    elif args.dataset == 'ERN':
        ch_path = f'/home/xqchen/attack_id_eegn/utils/ERN.txt'
    elif args.dataset == 'EPFL':
        ch_path = f'/home/xqchen/attack_id_eegn/utils/P300.txt'
    elif args.dataset == 'p2014009':
        ch_path = f'/home/xqchen/attack_id_eegn/utils/p2014009.txt'
    elif args.dataset == 'MI2014004':
        ch_path = f'/home/xqchen/attack_id_eegn/utils/mi2014004.txt'

    ch_names = []
    with open(ch_path, 'r') as file:
        for line in file.readlines():
            line = line.replace('\n', '').split('\t')
            ch_names.append(line[-1])

    if args.dataset == 'MI2014001':
        info = mne.create_info(
            ch_names=ch_names,
            ch_types=['eeg'] * len(ch_names),
            sfreq=256
        )
    elif args.dataset == 'physionet':
        info = mne.create_info(
            ch_names=ch_names,
            ch_types=['eeg'] * len(ch_names),
            sfreq=160
        )
    elif args.dataset == 'bcimi':
        info = mne.create_info(
            ch_names=ch_names,
            ch_types=['eeg'] * len(ch_names),
            sfreq=128
        )
    elif args.dataset == 'ERN':
        info = mne.create_info(
            ch_names=ch_names,
            ch_types=['eeg'] * len(ch_names),
            sfreq=128
        )
    elif args.dataset == 'EPFL':
        info = mne.create_info(
            ch_names=ch_names,
            ch_types=['eeg'] * len(ch_names),
            sfreq=128
        )
    elif args.dataset == 'p2014009':
        info = mne.create_info(
            ch_names=ch_names,
            ch_types=['eeg'] * len(ch_names),
            sfreq=256
        )
    elif args.dataset == 'MI2014004':
        info = mne.create_info(
            ch_names=ch_names,
            ch_types=['eeg'] * len(ch_names),
            sfreq=250
        )
    info.set_montage('standard_1020')
    # if args.dataset == 'p2014009':
    #     info.set_montage('standard_1010')
    epochs = mne.EpochsArray(eeg.squeeze(), info)
    epochs_sl = mne.preprocessing.compute_current_source_density(epochs)
    return epochs_sl.get_data()




def average_referencing(eeg):
    eeg_ar = eeg - np.mean(eeg, axis=-2, keepdims=True)
    return eeg_ar




def run(x: torch.Tensor, y: torch.Tensor, s_train: torch.Tensor,x_test: torch.Tensor,
        y_test: torch.Tensor,s_test: torch.Tensor, tra, args):
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
    # transform = data_transform.Shifting(4, sf_dict[args.dataset])
    # transform = data_transform.timeshuffle()
    # transform = data_transform.timeconshuffle()
    if args.trans == 'shift':
        transform = data_transform.Shifting(4, sf_dict[args.dataset])
    elif args.trans == 'tcshuffle':
        transform = data_transform.timeconshuffle(num_chunks=args.num_chunks)
    elif args.trans == 'cshuffle':
        transform = data_transform.ChannelShuffle(shuffle_rate=args.shuffle_rate)
    else:
        transform = data_transform.no()


    sample_weights = weight_for_balanced_classes(y)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(
        sample_weights, len(sample_weights))
    
    train_loader = DataLoader(dataset=CustomTensorDataset((x, y),
                                                              transform),
                              batch_size=args.batch_size,
                              sampler=sampler,
                              drop_last=False,num_workers=3)
    # train_loader = DataLoader(dataset=TensorDataset(x, y),
    #                           batch_size=args.batch_size,
    #                           sampler=sampler,
    #                           drop_last=False,num_workers=3)
    
    
    test_loader = DataLoader(dataset=TensorDataset(x_test, y_test),
                             batch_size=args.batch_size,
                             shuffle=True,
                             drop_last=False,num_workers=3)
    

    sample_weights = weight_for_balanced_classes(s_train)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(
        sample_weights, len(sample_weights))
    
    train_loader_s = DataLoader(dataset=CustomTensorDataset((x, s_train),
                                                              transform),
                              batch_size=args.batch_size,
                              sampler=sampler,
                              drop_last=False,num_workers=3)
    # train_loader_s = DataLoader(dataset=TensorDataset(x, s_train),
    #                           batch_size=args.batch_size,
    #                           sampler=sampler,
    #                           drop_last=False,num_workers=3)
    
    test_loader_s = DataLoader(dataset=TensorDataset(x_test, s_test),
                             batch_size=args.batch_size,
                             shuffle=True,
                             drop_last=False,num_workers=3)

    # ======================train task======================
    # logging.info(f'=======train classification task model=======')
    # modelF, modelC = train(modelF,modelC,train_loader,test_loader,
    #                                     optimizer,criloss, tra,
    #                                     args)

    
    # test_loss, test_acc, task_bca = eval(modelF,modelC, criloss,
    #      test_loader)
    # logging.info('test_loss: {:.4f} test acc: {:.4f} test bca: {:.2f}'
    #                 .format(test_loss, test_acc, task_bca))
    task_bca = 0

    # ======================train user======================
    logging.info(f'=======train user recognize model=======')
    modelF_s, modelC_s = train(modelF_s,modelC_s,train_loader_s,test_loader_s,
                                        optimizer_s,usrloss, tra,
                                        args)
    test_loss, test_acc, pid_bca = eval(modelF_s,modelC_s, usrloss,
         test_loader_s)
    logging.info('test_loss: {:.4f} test acc: {:.4f} test bca: {:.2f}'
                    .format(test_loss, test_acc, pid_bca))
    

    # torch.save(modelF.state_dict(),
    #                 model_save_path + '/modelF.pt')
    # torch.save(modelC.state_dict(),
    #                 model_save_path + '/modelC.pt')
    
    # torch.save(modelF_s.state_dict(),
    #                 model_save_path + '/modelF_s.pt')
    # torch.save(modelC_s.state_dict(),
    #                 model_save_path + '/modelC_s.pt')
    
    
    return task_bca, pid_bca, modelF, modelC, modelF_s, modelC_s

def train(modelF, modelC, train_loader, test_loader, optimizer, criloss, tra, args):
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

        # if (epoch + 1) % 10 == 0 or (epoch + 1) == args.epochs:
        #     if args.dataset != 'EPFL':
        #         train_loss, train_acc, train_bca = eval(modelF, modelC, criloss, train_loader)
        #         test_loss, test_acc, test_bca = eval(modelF, modelC, criloss, test_loader)

        #         logging.info(
        #             'Epoch {}/{}: train_loss: {:.4f} train acc: {:.4f} train bca: {:.2f} | test_loss: {:.4f} test acc: {:.4f} test bca: {:.2f}'
        #             .format(epoch + 1, args.epochs, train_loss, train_acc, train_bca, test_loss,
        #                     test_acc, test_bca))
        #     else:
        #         train_loss, train_acc, train_bca = eval(modelF, modelC, criloss, train_loader)
        #         logging.info(
        #             'Epoch {}/{}: train_loss: {:.4f} train acc: {:.4f} train bca: {:.2f}'
        #             .format(epoch + 1, args.epochs, train_loss, train_acc, train_bca))

            
    return modelF, modelC

def eval(model1: nn.Module, model2: nn.Module, criterion: nn.Module,
         data_loader: DataLoader):
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

    return loss, acc, bca



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
    parser.add_argument('--model', type=str, default='EEGNet') #ShallowCNN DeepCNN EEGNet
    parser.add_argument('--dataset', type=str, default='bcimi')# physionet MI2014001
    
    parser.add_argument('--premask', type=str, default='no')#['no','rand','sn']  'optim_linf','adv_linf','optim_l2','adv_l2' 

    parser.add_argument('--postmask', nargs='+',default=['adv_linf'])# 0.1_['no', 'rand','square','rand_binary','rand_binary2','optim','optim_linf','optim_l2']
    parser.add_argument('--maskamp', type=float, default=0.3)#['no', 'rand', 'sn', 'optim_linf', 'adv_linf']
    parser.add_argument('--maskl2_ind', type=float, default=0.3)#['no', 'rand', 'sn']
    
    parser.add_argument('--trans', type=str, default='sl')#['sl','ar','shift','tcshuffle','cshuffle'] ['optim_linf','adv_linf','optim_l2','adv_l2']
    parser.add_argument('--num_chunks', type=int, default=50)#['sl','ar','shift','tcshuffle','cshuffle'] ['optim_linf','adv_linf','optim_l2','adv_l2']
    parser.add_argument('--shuffle_rate', type=float, default=100)#['sl','ar','shift','tcshuffle','cshuffle'] ['optim_linf','adv_linf','optim_l2','adv_l2']
    
    parser.add_argument('--nchu', type=int, default=5)
    parser.add_argument('--nmodel', type=int, default=1)

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

    pd.set_option('display.max_rows', None)  
    pd.set_option('display.max_columns', None)  
    pd.set_option('display.width', None)  


    args.device = f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu'
    

    subject_num_dict = {'MI4C': 9, 'ERN': 16, 'EPFL': 8,'bench':35, 'tusz':192,'nicu':8, 'MI2014001':9,  'MI2014001-2':9,\
                        'MI2015001':12, 'MI2014004':9, 'MI2015004':9, 'weibo':10, 'weibocsp':10,\
                            'physionet':109, 'physionet2c':109, 'p2014009':10,'20140091':10,'bcimi':60}
    

    sf_dict = {'MI2014001':250,'MI2014004':250,'MI2015001':512,\
        'weibo':200 ,'ERN': 128,'bench':250,'nicu':256,'p2014009':256,  \
            'tusz':250, 'physionet': 160,'EPFL':128, '2014009':256,'20140091':256,\
                'bcimi':128}

    if args.dataset == 'tusz':
        args.batch_size = 256
        # args.epochs = 
    if args.postmask == ['adv_linf']:
        args.nmodel = 3
    if args.postmask == ['optim_linf']:
        args.nmodel = 1
    # ========================model path=======================
    model_path = f'/data1/cxq/model_id/{args.dataset}/trans_{args.model}/'

    # ========================log name and excel name=======================
    log_path = f'/home/xqchen/attack_id_eegn/result/log'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_name = os.path.join(log_path,
                            f'trans_{args.dataset}_{args.model}_{args.maskamp}-{args.maskl2_ind}-{args.postmask}_{args.num_chunks}-{args.shuffle_rate}-{args.trans}.log')
    

    excel_path = f'/home/xqchen/attack_id_eegn/result/excel'
    if not os.path.exists(excel_path):
        os.makedirs(excel_path)
    excel_name = os.path.join(excel_path,f'trans_{args.dataset}_{args.model}_{args.maskamp}-{args.maskl2_ind}-{args.postmask}_{args.num_chunks}-{args.shuffle_rate}-{args.trans}.xlsx')
    
    # system time
    args.commit = datetime.datetime.now()
    log_name = log_name.replace('.log', f'_{args.commit}.log')
    excel_name = excel_name.replace('.xlsx', f'_{args.commit}.xlsx')

    recorder = np.zeros((args.repeat+2,round(2*len(args.postmask))))
    recorder_pert = np.zeros((args.repeat+2,round(2*len(args.postmask))))
    
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
            elif args.dataset == 'MI2014004':
                x_train, y_train, s_train, x_test, y_test, s_test = MI2014004Load(premask=args.premask)
            elif args.dataset == 'ERN':
                x_train, y_train, s_train, x_test, y_test, s_test = ERNLoad(premask=args.premask)
            elif args.dataset == 'p2014009':
                x_train, y_train, s_train, x_test, y_test, s_test = p3002014009Load(premask=args.premask)
            elif args.dataset == 'bcimi':
                x_train, y_train, s_train, x_test, y_test, s_test = bcimiLoad(premask=args.premask)

            args.maskl2 = np.sqrt((x_train.shape[-1]*x_train.shape[-2])*(args.maskl2_ind * np.std(x_train))**2)

            logging.info(f'maskl2: {args.maskl2}')

            if pm == 'rand':
                #通过本函数可以返回一个或一组服从“0~1”均匀分布的随机样本值。随机样本取值范围是[0,1)，不包括1。 
                template = np.random.rand(subject_num_dict[args.dataset],x_train.shape[2],x_train.shape[3]) * 2 - 1
                l2 = 0
                l_infs = []
                for k in range(len(np.unique(s_train))):
                    templ = template[k:k+1,:,:]
                    templ = templ * np.std(x_train[s_train==k]) * args.maskamp

                    l2 += np.sqrt(np.sum(templ**2))
                    l_infs.append(np.max(np.abs(templ)))
                    x_train[s_train==k] = x_train[s_train==k] + templ
                linf = np.mean(np.array(l_infs))
                l2 = l2/subject_num_dict[args.dataset]
                logging.info('max perturbation: {:.4f}, l2: {:.4f}'.format(linf,l2))
            elif pm == 'sn':
                cho = np.random.choice(range(1,1000),subject_num_dict[args.dataset],replace=False)
                l2 = 0
                l_infs = []
                for k in range(len(np.unique(s_train))):
                    temp = list( '{:0>10}'.format(str(bin(cho[k]))[2:]))
                    temp = [list(m*10) for m in temp]
                    temp = np.tile(np.array(temp).astype(int).reshape(-1),(x_train.shape[2],500))[:,:x_train.shape[3]] * 2 - 1
                    templ = np.tile(temp,(len(x_train[s_train==k]),1,1,1))
                    templ = np.std(x_train[s_train==k]) * np.tile(np.random.uniform(0.5,1.5,x_train.shape[-2])[:,None],(x_train[s_train==k].shape[0],1,1,1)) * templ * args.maskamp
                    l2 += np.sqrt(np.sum(templ**2,axis=(1,2,3))).mean()
                    l_infs.append(np.max(np.abs(templ)))
                    x_train[s_train==k] = x_train[s_train==k] + templ
                linf = np.mean(np.array(l_infs))
                l2 = l2/subject_num_dict[args.dataset]
                logging.info('max perturbation: {:.4f}, l2: {:.4f}'.format(linf,l2))

            elif pm == 'optim_linf':
                templ = unlearnable_optim_linf(x_train, s_train, np.std(x_train) * args.maskamp, args)
                l2 = np.mean(np.sqrt(np.sum((templ-x_train).reshape(len(x_train),-1)**2,axis=-1)))
                l_infs = []
                for k in range(len(np.unique(s_train))):
                    l_infs.append(np.max(np.abs(templ[s_train==k]-x_train[s_train==k])))
                linf = np.mean(np.array(l_infs))
                logging.info('max perturbation: {:.4f}, l2: {:.4f}'.format(linf,l2))
                x_train = templ
            elif pm == 'adv_linf':
                templ = adv_linf(x_train, s_train, np.std(x_train) * args.maskamp, args)
                l2 = np.mean(np.sqrt(np.sum((templ-x_train).reshape(len(x_train),-1)**2,axis=-1)))
                l_infs = []
                for k in range(len(np.unique(s_train))):
                    l_infs.append(np.max(np.abs(templ[s_train==k]-x_train[s_train==k])))
                linf = np.mean(np.array(l_infs))
                logging.info('max perturbation: {:.4f}, l2: {:.4f}'.format(linf,l2))
                x_train = templ


            if pm == 'no':
                recorder_pert[r,2*p] = 0
                recorder_pert[r,2*p+1] = 0
            else:
                recorder_pert[r,2*p] = linf
                recorder_pert[r,2*p+1] = l2


            if args.trans == 'sl':
                x_train = surface_laplacian(np.squeeze(x_train),args)[:,None,:,:]
            # elif args.trans == 'ar':
            #     x_train = average_referencing(x_train)
            # elif args.trans == 'ea':
            #     ca = centroid_align(center_type='euclid', cov_type='lwf')
            #     for k in range(len(np.unique(s_train))):
            #         x_train[s_train==k] = ca.fit_transform(np.squeeze(x_train[s_train==k]))[:,None,:,:]
            # elif args.trans == 'fil':
            #     from scipy import signal
            #     if args.postmask == 'optim_linf' or args.postmask == 'optim_l2' or args.postmask == 'sn':
            #         b, a = signal.butter(6, 2*20/sf_dict[args.dataset], 'highpass')   #配置滤波器 8 表示滤波器的阶数
            #         x_train = signal.filtfilt(b, a, x_train,axis=-1)
            #     elif args.postmask == 'adv_linf' or args.postmask == 'adv_l2':
            #         b, a = signal.butter(6,  [ 2*20/sf_dict[args.dataset] , 2*40/sf_dict[args.dataset] ], 'bandstop')   #配置滤波器 8 表示滤波器的阶数
            #         x_train = signal.filtfilt(b, a, x_train,axis=-1)

            

            logging.info(f'train: {x_train.shape},{x_train.mean()},{x_train.std()}, test: {x_test.shape},{x_test.mean()},{x_test.std()}')
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

            # for t,tra in enumerate(args.trans):
            model_save_path = os.path.join(model_path, f'{r}/{args.maskamp}_{args.maskl2_ind}_{pm}_{args.num_chunks}-{args.shuffle_rate}-{args.trans}/')
            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)
            args.model_path = model_save_path

            logging.info(f'********** repeat: {r}, postmask: {pm}, trans: {args.trans}, training: {args.train} **********')

            task_bca, pid_bca, modelF,modelC,modelF_s, modelC_s = run(x_train, y_train, s_train,
                                                    x_test, y_test, s_test, args.train, args)
            recorder[r,p*2] = task_bca
            recorder[r,p*2+1] = pid_bca

            torch.cuda.empty_cache()
    
    recorder[-2,:] = np.mean(recorder[:-2,:],axis=0)
    recorder[-1,:] = np.std(recorder[:-2,:],axis=0)

    recorder_pert[-2,:] = np.mean(recorder_pert[:-2,:],axis=0)
    recorder_pert[-1,:] = np.std(recorder_pert[:-2,:],axis=0)

    recorder_df = DataFrame(recorder,
               index = [f'repeat_{g}' for g in range(args.repeat)]+['mean', 'std'],
               columns = pd.MultiIndex.from_product([[g for g in args.postmask],
                                                   [args.trans],['Task', 'PID']]))

    recorder_pert_df = DataFrame(recorder_pert,
               index = [f'repeat_{g}' for g in range(args.repeat)]+['mean', 'std'],
               columns = pd.MultiIndex.from_product([[g for g in args.postmask],['Linf', 'L2']]))

    logging.info(print_args(args) + '\n')
    logging.info('================================final result================================')
    logging.info(recorder_df)
    logging.info('  ')
    logging.info(recorder_pert_df)
    with pd.ExcelWriter(excel_name) as writer:
        recorder_df.to_excel(writer, sheet_name='bca')
        recorder_pert_df.to_excel(writer, sheet_name='distance')




