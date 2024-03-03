import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
from typing import Optional
from torch.optim import optimizer
from collections import OrderedDict

from scipy.signal import hilbert
def diff_in_weights(model: nn.Module, proxy: nn.Module):
    diff_dict = OrderedDict()
    model_state_dict = model.state_dict()
    proxy_state_dict = proxy.state_dict()
    for (old_k, old_w), (new_k, new_w) in zip(model_state_dict.items(),
                                              proxy_state_dict.items()):
        if len(old_w.size()) <= 1:
            continue
        if 'weight' in old_k:
            diff_w = new_w - old_w
            diff_dict[old_k] = old_w.norm() / (diff_w.norm() + 1e-20) * diff_w
    return diff_dict


def add_into_weights(model: nn.Module,
                     diff: OrderedDict,
                     coeff: Optional[float] = 1.0):
    names_in_diff = diff.keys()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in names_in_diff:
                param.add_(coeff * diff[name])


def tradesLoss(model: nn.Module,
               x: torch.Tensor,
               y: torch.Tensor,
               optimizer: optimizer,
               epoch: int,
               eps: Optional[float] = 0.05,
               alpha: Optional[float] = 0.005,
               steps: Optional[int] = 20,
               distance: Optional[str] = 'l_inf',
               beta: Optional[float] = 1.0,
               awp: Optional[bool] = False,
               gamma: Optional[float] = 0.005):
    """ trades loss """
    device = next(model.parameters()).device
    # KL loss
    criterion_kl = nn.KLDivLoss(reduction='sum').to(device)
    criterion_cal = nn.CrossEntropyLoss().to(device)
    batch_size = len(x)

    # generate adversarial example, PGD for l_inf, optimize for l_2
    model.eval()
    adv_x = x.clone().detach() + torch.empty_like(x).uniform_(-eps, eps)
    if distance == 'l_inf':
        for _ in range(steps):
            adv_x.requires_grad = True
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(adv_x), dim=1),
                                       F.softmax(model(x), dim=1))
            grad = torch.autograd.grad(loss_kl,
                                       adv_x,
                                       retain_graph=False,
                                       create_graph=False)[0]
            adv_x = adv_x.detach() + alpha * grad.detach().sign()
            # projection
            delta = torch.clamp(adv_x - x, min=-eps, max=eps)
            adv_x = (x + delta).detach()
    elif distance == 'l_2':
        delta = 0.001 * torch.randn(x.shape).detach().to(device)
        delta = Variable(delta.data, requires_grad=True)

        # setup optimizer
        optimizer_delta = optim.SGD([delta], lr=eps / steps * 2)
        for _ in range(steps):
            adv_x = x + delta
            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * criterion_kl(F.log_softmax(model(adv_x), dim=1),
                                           F.softmax(model(x), dim=1))
            loss.backward()
            # renorm gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.rand_like(
                    delta.grad[grad_norms == 0])
            optimizer_delta.step()
            # projection
            delta.data.renorm_(p=2, dim=0, maxnorm=eps)
        adv_x = (x + delta).detach()
    else:
        raise f'No {distance} distance'

    # cal awp
    if awp and epoch >= 10:
        proxy = copy.deepcopy(model)
        proxy_optimizer = optim.SGD(proxy.parameters(), lr=0.001)
        proxy.train()

        l_cal = criterion_cal(proxy(x), y)
        l_rob = (1.0 / batch_size) * criterion_kl(
            F.log_softmax(proxy(adv_x), dim=1), F.softmax(proxy(x), dim=1))
        l = -1.0 * (l_cal + beta * l_rob)
        proxy_optimizer.zero_grad()
        l.backward()
        proxy_optimizer.step()
        # the adversary weight perturb
        diff = diff_in_weights(model, proxy)
        # add the weight perturb
        add_into_weights(model, diff, coeff=1.0 * gamma)

    # cal loss
    model.train()
    adv_logits = model(adv_x)
    logits = model(x)

    loss_cal = criterion_cal(logits, y)
    loss_rob = (1.0 / batch_size) * criterion_kl(
        F.log_softmax(adv_logits, dim=1), F.softmax(logits, dim=1))
    loss = loss_cal + beta * loss_rob

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if awp and epoch >= 10:
        add_into_weights(model, diff, coeff=-1.0 * gamma)

    return loss.item()


class LabelSmoothLoss(nn.Module):
    def __init__(self, n_class, alpha) -> None:
        super(LabelSmoothLoss, self).__init__()
        self.n_class = n_class
        self.lb_pos = 1.0 - alpha
        self.lb_neg = alpha / (n_class - 1)

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            label_onehot = torch.empty_like(pred).fill_(self.lb_neg).scatter_(
                dim=1, index=target.data.unsqueeze(1), value=self.lb_pos)
        return torch.mean(-torch.sum(label_onehot * pred, dim=-1))


class Guassian(object):
    def __init__(self, am) -> None:
        super().__init__()
        self.am = am

    def __call__(self, input):
        input = input + self.am * torch.randn(input.size())
        return input





class ChannelWiseGuassian(object):
    def __init__(self, max_am) -> None:
        super().__init__()
        self.max_am = max_am

    def __call__(self, input):
        '''
        input.shape
        torch.Size([1, 22, 1000])   
        '''
        input_am = torch.std(input, dim=-1)[:,:,None]
        shape = input.shape
        am = self.max_am * input_am #[0,1)均匀分布
        input += am * torch.randn(size=shape)

        return input

class Flip(object):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, input):
        '''
        input.shape
        torch.Size([1, 22, 1000])   
        '''
        if  torch.randn(1) >0:
            input = input.max() - input

        return input
    
class Scale(object):
    def __init__(self) -> None:
        super().__init__()
        self.scale = 0.05

    def __call__(self, input):
        '''
        input.shape
        torch.Size([1, 22, 1000])   
        '''
        c = np.random.choice([1,2,3])
        if  c == 2:
            input = input * (1+self.scale)
        elif c == 3:
            input = input * (1-self.scale)

        return input
    
class FreqShift(object):
    def __init__(self, sf) -> None:
        super().__init__()
        self.shiftphase = 0.2
        self.sf = sf

    def __call__(self, input):
        '''
        input.shape
        torch.Size([1, 22, 1000])   
        '''
        c = np.random.choice([1,2,3])
        if  c == 2:
            input = self.shift(input,self.shiftphase,1/self.sf,input.shape[1])
        elif c==2:
            input = self.shift(input,-self.shiftphase,1/self.sf,input.shape[1])

        return input
    
    def shift(self, x, f_shift, dt=1 / 250, num_channels=22):
        shifted_sig = torch.zeros_like(x)
        len_x = x.shape[-1]

        padding_len = 2 ** self.nextpow2(len_x)
        padding = torch.zeros(1, num_channels, padding_len - len_x,dtype=torch.float32)
        with_padding = torch.cat((x, padding),dim=-1)

        hilb_T = torch.tensor(hilbert(with_padding.numpy(), axis=-1), dtype=torch.complex64)
        t = torch.arange(0, padding_len, dtype=torch.float32)
        shift_func = torch.exp(2j * np.pi * f_shift * dt * t)

        for i in range(num_channels):
            shifted_sig[0,i, :] = (hilb_T[0,i, :] * shift_func)[:len_x].real

        return shifted_sig
    
    def nextpow2(self,x):
        return int(np.ceil(np.log2(np.abs(x))))



class ChannelAMChange(object):
    def __init__(self, max_ratio=1.5):
        super().__init__()
        self.max_ratio = max_ratio

    def __call__(self, input):
        shape = input.shape
        input = self.max_ratio * torch.rand(size=(1, shape[1], 1)) * input
        return input


class no(object):
    # def __init__(self, max_ratio=1.5):
    #     super().__init__()
    #     self.max_ratio = max_ratio

    def __call__(self, input):
        # shape = input.shape
        # input = self.max_ratio * torch.rand(size=(1, shape[1], 1)) * input
        return input

class Sampling(object):
    def __init__(self, scale_factor) -> None:
        super().__init__()
        self.scale_factor = scale_factor

    def __call__(self, input):
        shape = list(input.shape)
        sf = torch.rand(1) * self.scale_factor + 0.5
        input = F.interpolate(input,
                              size=[int(shape[-1] * sf)],
                              mode='nearest')
        input = F.interpolate(input, size=shape[-1], mode='nearest')
        return input


class Shifting(object):
    def __init__(self, shift_scale, sample_freq):
        self.shift_scale = shift_scale
        self.sample_freq = sample_freq

    def __call__(self, input):
        shift = np.random.randint(int(self.shift_scale * self.sample_freq))
        direction = 1 if np.random.random(1) >= 0.5 else -1
        shift = direction * shift

        input = torch.roll(input, shift)

        return input
    
class timeshuffle(object):
    # def __init__(self):

    def __call__(self, input):
        # shift = np.random.randint(int(self.shift_scale * 128))
        # direction = 1 if np.random.random(1) >= 0.5 else -1
        # shift = direction * shift

        # input = torch.roll(input, shift)
        input = input[:,:,torch.randperm(input.shape[-1])]

        return input
    
class timeconshuffle(object):
    def __init__(self, num_chunks):
        self.num_chunks = num_chunks

    def __call__(self, input):

        num_chunks = self.num_chunks
        chunks = torch.chunk(input, num_chunks, dim=-1)
        random_order = torch.randperm(len(chunks))
        shuffled_tensor = torch.cat([chunks[i] for i in random_order], dim=-1)


        return shuffled_tensor


class ChannelShuffle(object):
    def __init__(self, shuffle_rate):
        self.shuffle_rate = shuffle_rate

    def __call__(self, input):
        num = int(self.shuffle_rate * input.shape[-2])
        idx = np.random.permutation(np.arange(input.shape[-2]))
        idx = idx[:num]
        shuffle_idx = np.random.permutation(idx)
        input[:, idx, :] = input[:, shuffle_idx, :]

        return input


class RandomTransform(object):
    def __init__(self, gs_am, scale_factor, shift_scale, shuffle_rate,
                 max_ratio, n_transform) -> None:
        super().__init__()
        self.gs = ChannelWiseGuassian(gs_am)
        self.sampling = Sampling(scale_factor)
        self.shifting = Shifting(shift_scale)
        self.amchange = ChannelAMChange(max_ratio)
        self.transforms = [
            self.gs, self.sampling, self.shifting, self.shuffling
        ]
        self.n_transform = n_transform

    def __call__(self, input):
        transform_idx = np.arange(len(self.transforms))
        transform_idx = np.random.permutation(transform_idx)
        for idx in transform_idx[:self.n_transform]:
            input = self.transforms[idx](input)

        return input


def get_transform(transform_name='random',
                  gs_am=0.5,
                  scale_factor=0.5,
                  shift_scale=0.5,
                  shuffle_rate=0.2,
                  max_ratio=1.5,
                  sf=250,
                  n_transform=2):
    transform = None
    if transform_name == 'guassian':
        transform = ChannelWiseGuassian(max_am=gs_am)
    elif transform_name == 'flip':
        transform = Flip()
    elif transform_name == 'scale':
        transform = Scale()
    elif transform_name == 'freqshift':
        transform = FreqShift(sf)
    elif transform_name == 'random':
        transform = RandomTransform(gs_am=gs_am,
                                    scale_factor=scale_factor,
                                    shift_scale=shift_scale,
                                    shuffle_rate=shuffle_rate,
                                    max_ratio=max_ratio,
                                    n_transform=n_transform)
    else:
        raise 'No such transform'

    return transform
