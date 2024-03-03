import os
import logging
import torch
import argparse
import datetime
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from pyriemann.tangentspace import TangentSpace
from pyriemann.estimation import XdawnCovariances
from pyriemann.channelselection import ElectrodeSelection
import numpy as np
import torch.nn as nn
import xgboost as xgb
import numpy as np
import openpyxl
import pandas as pd
from sklearn.preprocessing import StandardScaler
import mne
from pandas import Series, DataFrame
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from models import LoadModel, Classifier, CalculateOutSize
import attack_lib
import scipy.io as io
from utils.data_loader import *
from utils.pytorch_utils import init_weights, print_args, seed, weight_for_balanced_classes, bca_score
from unlearnable_gen import  unlearnable_optim_linf, adv_linf
import pandas as pd
import pywt
from scipy.fftpack import *
from statsmodels.tsa.arima.model import ARIMA
from pyentrp import entropy as entrp

from statsmodels.regression.linear_model import burg
def AR_burg(data:np.ndarray|list,order:int=5,is_win:bool=False,windowlen:int=200,step:int=100):
    T, C, S = data.shape # n_trials, n_channels, n_samples
    if not is_win:
        AR_fea = np.empty((T,C,order))
        for i in range(T):
            for j in range(C):
                AR_fea[i,j], _ = burg(data[i,j],order=order)
    elif is_win:
        win_num = (S - windowlen) // step
        AR_fea = np.empty((T,C,order*win_num))
        for i in range(T):
            for j in range(C):
                for k in range(win_num):
                    tmp = data[i,j,k*step:k*step+windowlen]
                    AR_fea[i,j,k*order:(k+1)*order], _ =  burg(tmp,order=order)
    else:
        raise NotImplementedError
    return AR_fea



        
def STFT(data, stft_para):
    # initialize the parameters
    # tips:u could input the paras or just use the global variable
    # STFTN = stft_para['stftn']
    STFTN = data.shape[-1]
    fStart = stft_para['fStart']
    fStop = stft_para['fStop']
    fs = stft_para['fs']
    # EyeTime = stft_para['EyeTime']

    TrialLength = data.shape[-1]#int(fs * EyeTime)   
    Hwindow = np.hanning(TrialLength)

    fStartNum = np.array([int(f / fs * STFTN) for f in fStart])
    fStopNum = np.array([int(f / fs * STFTN) for f in fStop])

    trial_num, n, m = data.shape
    l = m // TrialLength
    k = len(fStart)   
    psd = np.zeros((trial_num, n, l, k))
    de = np.zeros((trial_num, n, l, k))

    for idx in range(trial_num):
        for i in range(l):
            dataNow = data[idx, :, TrialLength * i:TrialLength * (i + 1)] # (n,TrialLength)
            for j in range(n):
                temp = dataNow[j, :]
                Hdata = temp

    
                FFTdata = fft(Hdata)
                freqs = fftfreq(TrialLength) * STFTN
                magFFTdata = np.abs(FFTdata[0:int(STFTN / 2)]) 

                for p in range(k):
                    E = 0
                    for p0 in range(fStartNum[p],fStopNum[p]):
                        E += magFFTdata[p0] ** 2  # for every freqence point
                    E = E / (fStopNum[p] - fStartNum[p] + 1)
                    psd[idx, j, i, p] = E
                    de[idx, j, i, p] = np.log2(100 * E + 1e-6)

    return psd#, de

def WaveletPacket(signal):
    trials, channels, _ = signal.shape
    bands = {'delta': [1, 3], 
            'theta': [4, 7],
            'alpha': [8, 13],
            'beta': [14, 30],
            'gamma': [31, 50]}
    levels = [2, 3, 4, 5, 6]  
    features = np.empty((trials,channels,3*len(bands)))
    for m in range(trials):
        for n in range(channels):
            tepdata = np.squeeze(signal[m,n,:])
            coeffs = pywt.wavedec(tepdata, wavelet="db4", level=max(levels), axis=-1)
            for band, level in zip(bands.keys(), levels):
                coeff = coeffs[level] 
                freq_min, freq_max = bands[band]
                mean, std = np.mean(coeff), np.std(coeff)
                entropy = np.array([-np.abs(x)**2 * np.log(np.abs(x)**2) for x in coeff]).sum()
                features[m,n, (level-2)*3:(level-1)*3] = [mean,std,entropy]
    return features

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model train')
    parser.add_argument('--gpu_id', type=str, default='7')
    parser.add_argument('--model', type=str, default='EEGNet') #ShallowCNN DeepCNN EEGNet
    parser.add_argument('--dataset', type=str, default='MI2014001')# physionet MI2014001 MI2014004 bcimi
    
    parser.add_argument('--premask', type=str, default='no')#'optim_l2','adv_linf'  'optim_linf','adv_linf'
    parser.add_argument('--postmask', nargs='+',default=['no'])# 0.1_['no', 'rand','square','rand_binary','rand_binary2','optim','optim_linf','optim_l2']
    parser.add_argument('--maskamp', type=float, default=1)#['no', 'rand', 'sn', 'optim_linf', 'adv_linf']
    parser.add_argument('--maskl2_ind', type=float, default=0.3)#['no', 'rand', 'rand_binary2']

    
    parser.add_argument('--alpha', type=float, default=0.01)

    parser.add_argument('--nchu', type=int, default=5)
    parser.add_argument('--nmodel', type=int, default=1)
    parser.add_argument('--method', type=str, default='ar')# wave stft


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
    model_path = f'/data1/cxq/model_id/{args.dataset}/{args.model}/'


    # ========================log name and excel name=======================
    log_path = f'/home/xqchen/attack_id_eegn/result/log'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_name = os.path.join(log_path,
                            f'uid_{args.dataset}_{args.method}_{args.maskamp}-{args.maskl2_ind}-{args.postmask}.log')
    

    excel_path = f'/home/xqchen/attack_id_eegn/result/excel'
    if not os.path.exists(excel_path):
        os.makedirs(excel_path)
    excel_name = os.path.join(excel_path,f'uid_{args.dataset}_{args.method}_{args.maskamp}-{args.maskl2_ind}-{args.postmask}.xlsx')
    
    # system time
    args.commit = datetime.datetime.now()
    log_name = log_name.replace('.log', f'_{args.commit}.log')
    excel_name = excel_name.replace('.xlsx', f'_{args.commit}.xlsx')

    recorder = np.zeros((args.repeat+2,round(2*len(args.train)*len(args.postmask))))
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
            elif args.dataset == 'ERN':
                x_train, y_train, s_train, x_test, y_test, s_test = ERNLoad(premask=args.premask)
            elif args.dataset == 'MI2014004':
                x_train, y_train, s_train, x_test, y_test, s_test = MI2014004Load(premask=args.premask)
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
                    # linf = np.max(np.abs(templ-x_train))
                linf = np.mean(np.array(l_infs))
                logging.info('max perturbation: {:.4f}, l2: {:.4f}'.format(linf,l2))
                x_train = templ

            if pm == 'no':
                recorder_pert[r,2*p] = 0
                recorder_pert[r,2*p+1] = 0
            else:
                recorder_pert[r,2*p] = linf
                recorder_pert[r,2*p+1] = l2

            x_train = np.squeeze(x_train)
            x_test = np.squeeze(x_test)

            if args.method == 'wave':
                x_train_fea = WaveletPacket(x_train)
                x_test_fea = WaveletPacket(x_test)
                model = xgb.XGBClassifier()
                model.fit(x_train_fea.reshape(len(x_train_fea),-1), s_train)
                y_pred = model.predict(x_test_fea.reshape(len(x_test_fea),-1))
                bca = bca_score(s_test.astype('int32'), y_pred)
            elif args.method == 'stft':
                x_train_fea = STFT(x_train,{"fStart": [1,4,8,14,31],"fStop": [3,7,13,30,50],"fs": sf_dict[args.dataset]})
                x_test_fea = STFT(x_test,{"fStart": [1,4,8,14,31],"fStop": [3,7,13,30,50],"fs": sf_dict[args.dataset]})
                model = xgb.XGBClassifier()
                model.fit(x_train_fea.reshape(len(x_train_fea),-1), s_train)
                y_pred = model.predict(x_test_fea.reshape(len(x_test_fea),-1))
                bca = bca_score(s_test.astype('int32'), y_pred)
            elif args.method == 'ar':
                x_train_fea = AR_burg(x_train)
                x_test_fea = AR_burg(x_test)
                model = xgb.XGBClassifier()
                model.fit(x_train_fea.reshape(len(x_train_fea),-1), s_train)
                y_pred = model.predict(x_test_fea.reshape(len(x_test_fea),-1))
                bca = bca_score(s_test.astype('int32'), y_pred)




            #     # # ===================

            logging.info(f'********** repeat: {r}, postmask: {pm},  pid bca: {bca} **********')

            recorder[r,p*2*len(args.train)] = bca

    
    recorder[-2,:] = np.mean(recorder[:-2,:],axis=0)
    recorder[-1,:] = np.std(recorder[:-2,:],axis=0)

    recorder_pert[-2,:] = np.mean(recorder_pert[:-2,:],axis=0)
    recorder_pert[-1,:] = np.std(recorder_pert[:-2,:],axis=0)

    recorder_df = DataFrame(recorder,
               index = [f'repeat_{g}' for g in range(args.repeat)]+['mean', 'std'],
               columns = pd.MultiIndex.from_product([[g for g in args.postmask],
                                                   [g for g in args.train],['Task', 'PID']]))

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




