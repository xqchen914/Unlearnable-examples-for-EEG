import torch
import torch.nn as nn
from typing import Optional


def CalculateOutSize(model, Chans, Samples):
    '''
    Calculate the output based on input size.
    model is from nn.Module and inputSize is a array.
    '''
    device = next(model.parameters()).device
    x = torch.rand(1, 1, Chans, Samples).to(device)
    out = model(x)
    return out.shape[-1]


def LoadModel(model_name, Chans, Samples):
    if model_name == 'EEGNet':
        model = EEGNet(Chans=Chans,
                       Samples=Samples,
                       kernLenght=64,
                       F1=4,
                       D=2,
                       F2=8,
                       dropoutRate=0.25)
    elif model_name == 'DeepCNN':
        model = DeepConvNet(Chans=Chans, Samples=Samples, dropoutRate=0.5)
    elif model_name == 'ShallowCNN':
        model = ShallowConvNet(Chans=Chans, Samples=Samples, dropoutRate=0.5)
    else:
        raise 'No such model'
    return model




class EEGNet(nn.Module):
    """
    :param
    """
    def __init__(self,
                 Chans: int,
                 Samples: int,
                 kernLenght: int,
                 F1: int,
                 D: int,
                 F2: int,
                 dropoutRate: Optional[float] = 0.5,
                 SAP_frac: Optional[float] = None):
        super(EEGNet, self).__init__()

        self.Chans = Chans
        self.Samples = Samples
        self.kernLenght = kernLenght
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.dropoutRate = dropoutRate
        self.SAP_frac = SAP_frac

        self.block1 = nn.Sequential(
            nn.ZeroPad2d((self.kernLenght // 2 - 1,
                          self.kernLenght - self.kernLenght // 2, 0,
                          0)),  # left, right, up, bottom
            nn.Conv2d(in_channels=1,
                      out_channels=self.F1,
                      kernel_size=(1, self.kernLenght),
                      stride=1,
                      bias=False),
            nn.BatchNorm2d(num_features=self.F1),
            # DepthwiseConv2d
            nn.Conv2d(in_channels=self.F1,
                      out_channels=self.F1 * self.D,
                      kernel_size=(self.Chans, 1),
                      groups=self.F1,
                      bias=False),
            nn.BatchNorm2d(num_features=self.F1 * self.D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(p=self.dropoutRate))

        self.block2 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            # SeparableConv2d
            nn.Conv2d(in_channels=self.F1 * self.D,
                      out_channels=self.F1 * self.D,
                      kernel_size=(1, 16),
                      stride=1,
                      groups=self.F1 * self.D,
                      bias=False),
            nn.BatchNorm2d(num_features=self.F1 * self.D),
            nn.Conv2d(in_channels=self.F1 * self.D,
                      out_channels=self.F2,
                      kernel_size=(1, 1),
                      stride=1,
                      bias=False),
            nn.BatchNorm2d(num_features=self.F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(self.dropoutRate))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.block1(x)
        output = self.block2(output)
        output = output.reshape(output.size(0), -1)

        return output

    def MaxNormConstraint(self):
        for n, p in self.block1.named_parameters():
            if n == '3.weight':
                p.data = torch.renorm(p.data, p=2, dim=0, maxnorm=1.0)

    def pred_ent(self, x):
        logits = self(x)
        lsm = nn.LogSoftmax(dim=-1)
        log_probs = lsm(logits)
        probs = torch.exp(log_probs)
        p_log_p = log_probs * probs
        predictive_entropy = -p_log_p.sum(axis=1)
        return predictive_entropy



        
class DeepConvNet(nn.Module):
    def __init__(self,
                 Chans: int,
                 Samples: int,
                 dropoutRate: Optional[float] = 0.5,
                 d1: Optional[int] = 25,
                 d2: Optional[int] = 50,
                 d3: Optional[int] = 100):
        super(DeepConvNet, self).__init__()

        self.Chans = Chans
        self.Samples = Samples
        self.dropoutRate = dropoutRate

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=d1, kernel_size=(1, 5)),
            nn.Conv2d(in_channels=d1, out_channels=d1, kernel_size=(Chans, 1)),
            nn.BatchNorm2d(num_features=d1), nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(self.dropoutRate))

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=d1, out_channels=d2, kernel_size=(1, 5)),
            nn.BatchNorm2d(num_features=d2), nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(self.dropoutRate))

        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=d2, out_channels=d3, kernel_size=(1, 5)),
            nn.BatchNorm2d(num_features=d3), nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(self.dropoutRate))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.block1(x)
        output = self.block2(output)
        output = self.block3(output)
        output = output.reshape(output.size(0), -1)
        return output

    def MaxNormConstraint(self):
        for block in [self.block1, self.block2, self.block3]:
            for n, p in block.named_parameters():
                if hasattr(n, 'weight') and (
                        not n.__class__.__name__.startswith('BatchNorm')):
                    p.data = torch.renorm(p.data, p=2, dim=0, maxnorm=2.0)


class Activation(nn.Module):
    def __init__(self, type):
        super(Activation, self).__init__()
        self.type = type

    def forward(self, input):
        if self.type == 'square':
            output = input * input
        elif self.type == 'log':
            output = torch.log(torch.clamp(input, min=1e-6))
        else:
            raise Exception('Invalid type !')

        return output


class ShallowConvNet(nn.Module):
    def __init__(
        self,
        Chans: int,
        Samples: int,
        dropoutRate: Optional[float] = 0.5, midDim: Optional[int] = 40,
    ):
        super(ShallowConvNet, self).__init__()

        self.Chans = Chans
        self.Samples = Samples
        self.dropoutRate = dropoutRate

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=midDim, kernel_size=(1, 13)),
            nn.Conv2d(in_channels=midDim,
                      out_channels=midDim,
                      kernel_size=(self.Chans, 1)),
            nn.BatchNorm2d(num_features=midDim), 
            nn.ELU(), #Activation('square'),
            nn.AvgPool2d(kernel_size=(1, 35), stride=(1, 7)),
            nn.ELU(), # Activation('log'), 
            nn.Dropout(self.dropoutRate))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.block1(x)
        output = output.reshape(output.size(0), -1)
        return output

    def MaxNormConstraint(self):
        for n, p in self.block1.named_parameters():
            if hasattr(n, 'weight') and (
                    not n.__class__.__name__.startswith('BatchNorm')):
                p.data = torch.renorm(p.data, p=2, dim=0, maxnorm=2.0)


class Classifier(nn.Module):
    def __init__(self, input_dim, n_classes):
        super(Classifier, self).__init__()

        self.input_dim = input_dim
        self.n_classes = n_classes

        self.block = nn.Sequential(
            nn.Linear(in_features=self.input_dim,
                      out_features=self.n_classes,
                      bias=True))

    def forward(self, feature):
        output = self.block(feature)

        return output

    def MaxNormConstraint(self):
        for n, p in self.block.named_parameters():
            if n == '0.weight':
                p.data = torch.renorm(p.data, p=2, dim=0, maxnorm=0.25)



    
class DomainDiscriminator(nn.Module):
    """
    Domain discriminator module - 2 layers MLP

    Parameters:
        - input_dim (int): dim of input features
        - hidden_dim (int): dim of hidden features
    """
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int):
        super(DomainDiscriminator, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)
    

class Discriminator(nn.Module):
    def __init__(self, input_dim, n_subjects):
        super(Discriminator, self).__init__()

        self.input_dim = input_dim
        self.n_subjects = n_subjects

        self.block = nn.Sequential(
            nn.Linear(in_features=self.input_dim, out_features=50, bias=True),
            nn.Linear(in_features=50, out_features=self.n_subjects, bias=True))

    def forward(self, feature):
        output = self.block(feature)

        return output



# class CNN_LSTM(nn.Module):
#     def __init__(self,channels,hidden_size,input_size,n_classes,num_layers,spatial_num=8,drop_out=0.25):
#         super(CNN_LSTM, self).__init__()

#         self.channels = channels
#         self.n_classes = n_classes
#         self.hidden_size = hidden_size
#         self.input_size = input_size 
#         self.drop_out = drop_out  
#         self.spatial_num = spatial_num
#         self.num_layers = num_layers

#         self.block1 = nn.Sequential(
#             nn.Conv2d(1,self.spatial_num,(self.channels,1),bias=False),
#             nn.BatchNorm2d(self.spatial_num),
#             nn.ELU(),
#             nn.AvgPool2d((1, 2)),
#             nn.Dropout(self.drop_out))
#         self.block2 = nn.Sequential(
#             nn.Conv2d(self.spatial_num,2*self.spatial_num,(1,1),bias=False),
#             nn.BatchNorm2d(2*self.spatial_num),
#             nn.ELU(),
#             nn.AvgPool2d((1, 2)),
#             nn.Dropout(self.drop_out))
#         self.block3 = nn.Sequential(
#             nn.Conv2d(2*self.spatial_num,4*self.spatial_num,(1,1),bias=False),
#             nn.BatchNorm2d(4*self.spatial_num),
#             nn.ELU(),
#             nn.AvgPool2d((1, 2)),
#             nn.Dropout(self.drop_out))
        
#         self.lstm = nn.LSTM(self.input_size//2, self.hidden_size, self.num_layers, batch_first=True)
#         self.clf = nn.Sequential(nn.Linear(in_features=self.hidden_size*self.spatial_num//2, out_features=self.hidden_size),
#                                  nn.Linear(in_features=self.hidden_size, out_features=self.n_classes))
        
    
#     def forward(self,x:torch.Tensor)-> torch.Tensor:
        
#         # X (batch_size,1,channels,time_points) = (B,1,C,T)
#         x = self.block1(x) # (B,spatial_num,1,T//2)
#         x = self.block2(x) # (B,2*spatial_num,1,T//4)
#         x = self.block3(x) # (B,4*spatial_num,1,T//8)  eg:(32,128,1,125)
#         x = x.reshape(x.shape[0],-1,self.input_size//2) # (32,16,1000) # (B,spatial_num//2,T) 
#         x, _ = self.lstm(x) # (B,spatial_num,hidden_size) eg:(32,16,192)
#         x = x.reshape(x.shape[0],-1)
#         x = self.clf(x) # (B,n_classes)

#         return x


class LSTM_CNN(nn.Module):
    def __init__(self, channels, input_size, hidden_size, num_layers, 
                 time_kernel_size=16, spatial_num=8, drop_out=0.5):
        super(LSTM_CNN, self).__init__()

        drop_out = 0.5
        self.channels = channels
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        
        # self.block_1 = nn.Sequential(
        #     nn.ZeroPad2d((time_kernel_size // 2 - 1, time_kernel_size // 2, 0, 0)),
        #     nn.Conv2d(1, 8, (1, time_kernel_size), bias=False),
        #     nn.BatchNorm2d(8),
        #     nn.ELU(),
        #     nn.AvgPool2d((1, 4))
        # )
        
        self.block_2 = nn.Sequential(
            nn.Conv2d(1, spatial_num, (channels, 1), bias=False),
            nn.BatchNorm2d(spatial_num),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(drop_out)
        )
        
        
    def forward(self, x):
        # x: (N, C, T)  N: batch_size; C: channels; T: times
        x = x.reshape(-1, self.channels, self.input_size)
        N, C, T = x.shape
        x = x.reshape(N * C, T // self.input_size, self.input_size)
        lstm_out, _ = self.lstm(x, None)
        
        # x: (N, 1, C, H)  H: hidden_size
        x = lstm_out[:, -1, :].reshape(N, 1, C, self.hidden_size)
        
        # x = self.block_1(x)
        x = self.block_2(x)
        
        x = x.view(x.size(0), -1)

        return x
    
    
class CNN_LSTM(nn.Module):
    def __init__(self,channels,time_points,hidden_size=128,num_layers=1,spatial_num=32,drop_out=0.25):
        super(CNN_LSTM, self).__init__()

        self.channels = channels
        self.time_points = time_points
        # self.n_classes = n_classes
        self.hidden_size = hidden_size
        self.drop_out = drop_out  
        self.spatial_num = spatial_num
        self.num_layers = num_layers

        self.block1 = nn.Sequential(
            nn.Conv2d(1,self.spatial_num,(self.channels,1),bias=False),
            nn.BatchNorm2d(self.spatial_num),
            nn.ELU(),
            nn.AvgPool2d((1, 2)),
            nn.Dropout(self.drop_out))
        self.block2 = nn.Sequential(
            nn.Conv2d(self.spatial_num,2*self.spatial_num,(1,1),bias=False),
            nn.BatchNorm2d(2*self.spatial_num),
            nn.ELU(),
            nn.AvgPool2d((1, 2)),
            nn.Dropout(self.drop_out))
        self.block3 = nn.Sequential(
            nn.Conv2d(2*self.spatial_num,4*self.spatial_num,(1,1),bias=False),
            nn.BatchNorm2d(4*self.spatial_num),
            nn.ELU(),
            nn.AvgPool2d((1, 2)),
            nn.Dropout(self.drop_out))
        self.convblock = nn.Sequential(self.block1,self.block2,self.block3)#torch.Size([1, 128, 1, 125])
        self.convT = CalculateOutSize(self.convblock,self.channels,self.time_points)
        
        self.lstm = nn.LSTM(8*self.convT, self.hidden_size, self.num_layers, batch_first=True)
        # self.clf = nn.Sequential(nn.Linear(in_features=self.hidden_size*self.spatial_num//2, out_features=self.hidden_size),
        #                          nn.Linear(in_features=self.hidden_size, out_features=self.n_classes))
        
    
    def forward(self,x:torch.Tensor)-> torch.Tensor:
        # X (batch_size,1,channels,time_points) = (B,1,C,T)
        x = self.block1(x) # (B,spatial_num,1,T//2)
        x = self.block2(x) # (B,2*spatial_num,1,T//4)
        x = self.block3(x) # (B,4*spatial_num,1,T//8)  eg:(32,128,1,125)
        x = x.reshape(x.shape[0],-1,8*self.convT)
        x = self.lstm(x)
        # x = self.clf(x) # (B,n_classes)

        return x