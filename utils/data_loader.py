from re import T
from typing import Optional
from scipy import signal
from scipy.signal import resample
import scipy.io as scio
import numpy as np
from utils.data_align import centroid_align
import os

def split(x, y):
    idx = np.arange(len(x))
    train_size = 240

    return x[idx[:train_size]], y[idx[:train_size]], x[idx[train_size:]], y[
        idx[train_size:]]



def MI2014001Load(premask: Optional[str] = 'no'):
    if  premask == 'rand':
        data_path = '/data1/cxq/data/markprocessedMI2014001_4s_sea/'
    elif premask == 'no':
        data_path = '/data1/cxq/data/processedMI2014001_4s_sea/'
    
    x_train, y_train, x_test, y_test = [], [], [], []
    s_label_train = []
    s_label_test = []

    for i in range(9):

        data = scio.loadmat(data_path + f's{i}E.mat')
        x1, y1 = data['x'], data['y']
        y1 = np.squeeze(y1.flatten())


        data = scio.loadmat(data_path + f's{i}T.mat')
        x2, y2 = data['x'], data['y']
        y2 = np.squeeze(y2.flatten())

        x_train.append(x1)
        y_train.append(y1)

        x_test.append(x2)
        y_test.append(y2)

        s_label_train.append(np.array([i]*len(x1)))
        s_label_test.append(np.array([i]*len(x2)))


    x_train = np.concatenate(x_train)
    y_train = np.hstack(y_train)


    x_test = np.concatenate(x_test)
    y_test = np.hstack(y_test)

    s_label_train = np.hstack(s_label_train)
    s_label_test = np.hstack(s_label_test)
    
    print(x_train.shape)
    print(y_train.shape)

    return x_train, y_train.squeeze(), s_label_train.squeeze(), x_test, y_test.squeeze(), s_label_test.squeeze()


def bcimiLoad(premask: Optional[str] = 'no'):
    if  premask == 'rand':
        data_path = '/data1/cxq/data/markprocessedMI2014001_4s_sea/'
    elif premask == 'no':
        data_path = '/data1/cxq/data/bci_mi_processed/'
    
    x_train, y_train, x_test, y_test = [], [], [], []
    s_label_train = []
    s_label_test = []

    for i in range(60):

        data = scio.loadmat(data_path + f's{i}_e.mat')
        x1, y1 = data['x'], data['y']
        y1 = np.squeeze(y1.flatten())


        data = scio.loadmat(data_path + f's{i}_t.mat')
        x2, y2 = data['x'], data['y']
        y2 = np.squeeze(y2.flatten())

        x_train.append(x1)
        y_train.append(y1)

        x_test.append(x2)
        y_test.append(y2)

        s_label_train.append(np.array([i]*len(x1)))
        s_label_test.append(np.array([i]*len(x2)))


    x_train = np.concatenate(x_train)
    y_train = np.hstack(y_train)


    x_test = np.concatenate(x_test)
    y_test = np.hstack(y_test)

    s_label_train = np.hstack(s_label_train)
    s_label_test = np.hstack(s_label_test)
    
    print(x_train.shape)
    print(y_train.shape)
    x_train *= 1e5
    x_test *= 1e5

    return x_train, y_train.squeeze(), s_label_train.squeeze(), x_test, y_test.squeeze(), s_label_test.squeeze()


def MI2014004Load(premask: Optional[str] = 'no'):
    if  premask == 'rand':
        data_path = '/data1/cxq/data/randprocessedMI2014004sea/'
    elif premask == 'no':
        data_path = '/data1/cxq/data/processedMI2014004/'
        data_path = '/data1/cxq/data/processedMI2014004sea/'
    
    x_train, y_train, x_test, y_test = [], [], [], []
    s_label_train = []
    s_label_test = []

    for i in range(9):

        data = scio.loadmat(data_path + f's{i}e.mat')
        x1, y1 = data['x'], data['y']
        y1 = np.squeeze(y1.flatten())


        data = scio.loadmat(data_path + f's{i}t.mat')
        x2, y2 = data['x'], data['y']
        y2 = np.squeeze(y2.flatten())

        x_train.append(x1)
        y_train.append(y1)

        x_test.append(x2)
        y_test.append(y2)

        s_label_train.append(np.array([i]*len(x1)))
        s_label_test.append(np.array([i]*len(x2)))


    x_train = np.concatenate(x_train)
    y_train = np.hstack(y_train)


    x_test = np.concatenate(x_test)
    y_test = np.hstack(y_test)

    s_label_train = np.hstack(s_label_train)
    s_label_test = np.hstack(s_label_test)
    
    print(x_train.shape)
    print(y_train.shape)

    return x_train, y_train.squeeze(), s_label_train.squeeze(), x_test, y_test.squeeze(), s_label_test.squeeze()


def weiboLoad(id: int, setup: Optional[str] = 'within',ea: Optional[str] = 'no',p=2):
    if ea == 'sub':
        data_path = '/data1/cxq/data/processed-weibo-2014subea/'
    elif ea == 'no':
        data_path = '/data1/cxq/data/processed-weibo-2014/'
    elif ea == 'sess':
        data_path = '/data1/cxq/data/processed-weibo-2014sea/'
    x_train, y_train, x_test, y_test = [], [], [], []
    if setup == 'within':
            data = scio.loadmat(data_path + f's{id}.mat')
            x, y = data['x'], data['y']
            y = np.squeeze(np.array(y).flatten())
            # y -= 1

            x_train, x_test = x[:round(60*p)], x[round(60*p):]
            y_train, y_test = y[:round(60*p)], y[round(60*p):]
    elif setup == 'cross':
        for i in range(10):

            data = scio.loadmat(data_path + f's{i}.mat')
            x, y = data['x'], data['y']
            y = np.squeeze(np.array(y).flatten())
            # y -= 1
            
            if i == id:
                x_test, y_test = x, y
            else:
                x_train.append(x)
                y_train.append(y)

        x_train = np.concatenate(x_train)
        y_train = np.hstack(y_train)
        
        print(x_train.shape)
        print(y_train.shape)
    else:
        raise Exception('No such Experiment setup.')

    return x_train, y_train.squeeze(), x_test, y_test.squeeze()



def epflLoad(premask: Optional[str] = 'no'):
    if  premask == 'no':
        data_path = '/data1/cxq/data/processedepfliirsea/'

    x_train, y_train, x_test, y_test = [], [], [], []
    s_label_train = []
    s_label_test = []

    for i in range(8):

        data = scio.loadmat(data_path + f's{i}_0.mat')
        x1, y1 = data['x'], data['y']
        y1 = np.squeeze(y1.flatten())

        x_train.append(x1)
        y_train.append(y1)

        data = scio.loadmat(data_path + f's{i}_1.mat')
        x2, y2 = data['x'], data['y']
        y2 = np.squeeze(y2.flatten())

        data = scio.loadmat(data_path + f's{i}_2.mat')
        x3, y3 = data['x'], data['y']
        y3 = np.squeeze(np.array(y3).flatten())

        data = scio.loadmat(data_path + f's{i}_3.mat')
        x4, y4 = data['x'], data['y']
        y4 = np.squeeze(np.array(y4).flatten())
    
        x_test.append(x2)
        y_test.append(y2)

        x_test.append(x3)
        y_test.append(y3)

        x_test.append(x4)
        y_test.append(y4)

        s_label_train.append(np.array([i]*len(x1)))
        s_label_test.append(np.array([i]*(len(x2)+len(x3)+len(x4))))

    x_train = np.concatenate(x_train)
    y_train = np.hstack(y_train)

    x_test = np.concatenate(x_test)
    y_test = np.hstack(y_test)

    s_label_train = np.hstack(s_label_train)
    s_label_test = np.hstack(s_label_test)
    
    print(x_train.shape)
    print(y_train.shape)

    return x_train, y_train.squeeze(), s_label_train.squeeze(), x_test, y_test.squeeze(), s_label_test.squeeze()


def p3002014009Load(premask: Optional[str] = 'no'):
    if  premask == 'no':
        data_path = '/data1/cxq/data/processed2014009/'
    elif premask == 'sess':
        data_path = '/data1/cxq/data/processed2014009sea/'

    x_train, y_train, x_test, y_test = [], [], [], []
    s_label_train = []
    s_label_test = []

    for i in range(10):

            data = scio.loadmat(data_path + f's{i}_0.mat')
            x1, y1 = data['x'], data['y']
            y1 = np.squeeze(y1.flatten())

            x_train.append(x1)
            y_train.append(y1)

            data = scio.loadmat(data_path + f's{i}_1.mat')
            x2, y2 = data['x'], data['y']
            y2 = np.squeeze(y2.flatten())

            data = scio.loadmat(data_path + f's{i}_2.mat')
            x3, y3 = data['x'], data['y']
            y3 = np.squeeze(np.array(y3).flatten())

        
            x_test.append(x2)
            y_test.append(y2)

            x_test.append(x3)
            y_test.append(y3)


            s_label_train.append(np.array([i]*len(x1)))
            s_label_test.append(np.array([i]*(len(x2)+len(x3))))

    x_train = np.concatenate(x_train)
    y_train = np.hstack(y_train)

    x_test = np.concatenate(x_test)
    y_test = np.hstack(y_test)

    s_label_train = np.hstack(s_label_train)
    s_label_test = np.hstack(s_label_test)
    
    print(x_train.shape)
    print(y_train.shape)

    return x_train, y_train.squeeze(), s_label_train.squeeze(), x_test, y_test.squeeze(), s_label_test.squeeze()



def physionetLoad(premask: Optional[str] = 'no'):

    data_path = '/data1/cxq/data/processedphysionet/'
    x_train, y_train, x_test, y_test = [], [], [], []
    s_label_train = []
    s_label_test = []

    for i in range(109):

        data = scio.loadmat(data_path + f's{i}r1.mat')
        x1, y1 = data['x'], data['y']
        y1 = np.squeeze(np.array(y1).flatten())

        x_train.append(x1)
        y_train.append(y1)

        data = scio.loadmat(data_path + f's{i}r2.mat')
        x2, y2 = data['x'], data['y']
        y2 = np.squeeze(np.array(y2).flatten())

        data = scio.loadmat(data_path + f's{i}r3.mat')
        x3, y3 = data['x'], data['y']
        y3 = np.squeeze(np.array(y3).flatten())

        x_test.append(x2)
        y_test.append(y2)

        x_test.append(x3)
        y_test.append(y3)

        s_label_train.append(np.array([i]*len(x1)))
        s_label_test.append(np.array([i]*(len(x2)+len(x3))))

    x_train = np.concatenate(x_train)
    y_train = np.hstack(y_train)


    x_test = np.concatenate(x_test)
    y_test = np.hstack(y_test)

    s_label_train = np.hstack(s_label_train)
    s_label_test = np.hstack(s_label_test)
    
    print(x_train.shape)
    print(y_train.shape)

    return x_train, y_train.squeeze(), s_label_train.squeeze(), x_test, y_test.squeeze(), s_label_test.squeeze()



def ERNLoad(premask: Optional[str] = 'no'):

    data_path = '/data1/cxq/data/processedERN/'
        
    x_train, y_train, x_test, y_test = [], [], [], []
    s_label_train = []
    s_label_test = []


    for i in range(16):
        data = scio.loadmat(data_path + f's{i}.mat')

        x, y = data['x'], data['y']
        y = np.squeeze(y.flatten())

        x_train.append(x[:60])
        y_train.append(y[:60])

        x_test.append(x[60:])
        y_test.append(y[60:])    

        s_label_train.append(np.array([i]*60))
        s_label_test.append(np.array([i]*(len(y)-60)))


    x_train = np.concatenate(x_train)
    y_train = np.hstack(y_train)


    x_test = np.concatenate(x_test)
    y_test = np.hstack(y_test)

    s_label_train = np.hstack(s_label_train)
    s_label_test = np.hstack(s_label_test)
    
    print(x_train.shape)
    print(y_train.shape)

    return x_train, y_train.squeeze(), s_label_train.squeeze(), x_test, y_test.squeeze(), s_label_test.squeeze()


def benchLoad(premask: Optional[str] = 'no'):

    data_path = '/data1/cxq/data/benchmark/processed/'
        
    x_train, y_train, x_test, y_test = [], [], [], []
    s_label_train = []
    s_label_test = []


    for i in range(16):
        data = scio.loadmat(data_path + f'S{i}.mat')

        x, y = data['x'], data['y']
        y = np.squeeze(y.flatten())

        x_train.append(np.concatenate(x[:3]))
        y_train.append(y[:round(3*40)])

        x_test.append(np.concatenate(x[3:]))
        y_test.append(y[round(3*40):])    

        s_label_train.append(np.array([i]*120))
        s_label_test.append(np.array([i]*(len(y)-120)))


    x_train = np.concatenate(x_train)
    y_train = np.hstack(y_train)


    x_test = np.concatenate(x_test)
    y_test = np.hstack(y_test)

    s_label_train = np.hstack(s_label_train)
    s_label_test = np.hstack(s_label_test)
    
    print(x_train.shape)
    print(y_train.shape)

    return x_train[:,None,:,:], y_train.squeeze(), s_label_train.squeeze(), x_test[:,None,:,:], y_test.squeeze(), s_label_test.squeeze()


# def tuszLoad(premask: Optional[str] = 'no'):

#     data_path = '/data1/cxq/data/TUSZ/processed_mine/'
        
#     x_train, y_train, x_test, y_test = [], [], [], []
#     s_label_train = []
#     s_label_test = []


#     for i in range(154):
#         data = scio.loadmat(data_path + f'S{i}.mat')

#         x0, y0 = data['x0'], data['y0'].reshape(-1)
#         x1, y1 = data['x1'], data['y1'].reshape(-1)

#         x_train.append(x0)
#         y_train.append(y0)

#         x_test.append(x1)
#         y_test.append(y1)    

#         s_label_train.append(np.array([i]*len(x0)))
#         s_label_test.append(np.array([i]*len(x1)))


#     x_train = np.concatenate(x_train)
#     y_train = np.hstack(y_train)


#     x_test = np.concatenate(x_test)
#     y_test = np.hstack(y_test)

#     s_label_train = np.hstack(s_label_train)
#     s_label_test = np.hstack(s_label_test)
    
#     print(x_train.shape)
#     print(y_train.shape)

#     return x_train[:,None,:,:], y_train.squeeze(), s_label_train.squeeze(), x_test[:,None,:,:], y_test.squeeze(), s_label_test.squeeze()
def tuszLoad(premask: Optional[str] = 'no'):

    data_path = '/data1/cxq/data/TUSZ/processed_mine/'
    data_path = '/data1/cxq/data/TUSZ/processed_4s/'
        
    x_train, y_train, x_test, y_test = [], [], [], []
    s_label_train = []
    s_label_test = []

    files = os.listdir('/data1/cxq/data/TUSZ/processed_mine')
    files = os.listdir('/data1/cxq/data/TUSZ/processed_4s')

    uid = 0 
    for i in range(300):
        if f'S{i}_0.mat' in files and f'S{i}_1.mat' in files:
            data1 = scio.loadmat('/data1/cxq/data/TUSZ/processed_4s/' + f'S{i}_0.mat')
            data2 = scio.loadmat('/data1/cxq/data/TUSZ/processed_4s/' + f'S{i}_1.mat')

            if len(data1['x']) >=200 and len(data2['x']) >=200:

                x0, y0 = data1['x'], data1['y'].reshape(-1)
                x1, y1 = data2['x'], data2['y'].reshape(-1)
            # if len(data1['x']) >=200 and len(data2['x']) >=100:

            #     x0, y0 = data1['x'][:100], data1['y'].reshape(-1)[:100]
            #     x1, y1 = data1['x'][100:200], data1['y'].reshape(-1)[100:200]

                x_train.append(x0)
                y_train.append(y0)

                x_test.append(x1)
                y_test.append(y1)    

                s_label_train.append(np.array([uid]*len(x0)))
                s_label_test.append(np.array([uid]*len(x1)))
                uid += 1


    x_train = np.concatenate(x_train)
    y_train = np.hstack(y_train)


    x_test = np.concatenate(x_test)
    y_test = np.hstack(y_test)

    s_label_train = np.hstack(s_label_train)
    s_label_test = np.hstack(s_label_test)
    
    print(x_train.shape)
    print(y_train.shape)

    return x_train[:,None,:,:], y_train.squeeze(), s_label_train.squeeze(), x_test[:,None,:,:], y_test.squeeze(), s_label_test.squeeze()



def nicuLoad(premask: Optional[str] = 'no'):

    data_path = '/data1/cxq/data/processedNICU/'
        
    x_train, y_train, x_test, y_test = [], [], [], []
    s_label_train = []
    s_label_test = []
    percent = 0.5


    for i in range(8):
        data = scio.loadmat(data_path + f's{i}.mat')

        x0 = data['cla0']
        x1 = data['cla1']

        tx_test0 = x0[round(percent*len(x0)):]
        tx_test1 = x1[round(percent*len(x1)):]
        ty_test0 = np.zeros(len(tx_test0))
        ty_test1 = np.ones(len(tx_test1))

        tx_train0 = x0[:round(percent*len(x0))]
        tx_train1 = x1[:round(percent*len(x1))]
        ty_train0 = np.zeros(len(tx_train0))
        ty_train1 = np.ones(len(tx_train1))

        tx_test = np.concatenate((tx_test0, tx_test1))
        ty_test = np.concatenate((ty_test0, ty_test1))

        tx_train = np.concatenate((tx_train0, tx_train1))
        ty_train = np.concatenate((ty_train0, ty_train1))


        x_train.append(tx_train)
        y_train.append(ty_train)

        x_test.append(tx_test)
        y_test.append(ty_test)    

        s_label_train.append(np.array([i]*len(tx_train)))
        s_label_test.append(np.array([i]*len(tx_test)))


    x_train = np.concatenate(x_train)
    y_train = np.hstack(y_train)


    x_test = np.concatenate(x_test)
    y_test = np.hstack(y_test)

    s_label_train = np.hstack(s_label_train)
    s_label_test = np.hstack(s_label_test)
    
    print(x_train.shape)
    print(y_train.shape)

    return x_train, y_train.squeeze(), s_label_train.squeeze(), x_test, y_test.squeeze(), s_label_test.squeeze()



def MI2015001Load(premask: Optional[str] = 'no'):
    if  premask == 'no':
        data_path = '/data1/cxq/data/processedMI2015001/'
       # x_train, y_train, x_test, y_test = [], [], [], []
    
    x_train, y_train, x_test, y_test = [], [], [], []
    s_label_train = []
    s_label_test = []

    for i in range(12):

        data = scio.loadmat(data_path + f's{i}a.mat')
        x1, y1 = data['x'], data['y']
        y1 = np.squeeze(y1.flatten())

        # x1 = x1[:,None,:,:]

        data = scio.loadmat(data_path + f's{i}b.mat')
        x2, y2 = data['x'], data['y']
        y2 = np.squeeze(y2.flatten())

        if i in [7,8,9,10]:
            data = scio.loadmat(data_path + f's{i}c.mat')
            x3, y3 = data['x'], data['y']
            y3 = np.squeeze(y3.flatten())

            x1 = np.concatenate((x1,x2))
            y1 = np.concatenate((y1,y2))

            x2 = x3
            y2 = y3

        x_train.append(x1)
        y_train.append(y1)   
        x_test.append(x2)
        y_test.append(y2)   
        s_label_train.append(np.array([i]*len(x1)))
        s_label_test.append(np.array([i]*(len(x2))))


    x_train = np.concatenate(x_train)
    y_train = np.hstack(y_train)

    x_test = np.concatenate(x_test)
    y_test = np.hstack(y_test)

    s_label_train = np.hstack(s_label_train)
    s_label_test = np.hstack(s_label_test)
    
    print(x_train.shape)
    print(y_train.shape)

    return x_train, y_train.squeeze(), s_label_train.squeeze(), x_test, y_test.squeeze(), s_label_test.squeeze()
