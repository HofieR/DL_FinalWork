import math
import numpy as np
import pandas as pd
import megengine as mge
import megengine.module as M
import megengine.functional as F
import megengine.data.transform as T
import pandas as pd
import copy
from tqdm import tqdm
from typing import Tuple
from megengine import tensor
from sklearn.model_selection import train_test_split
from megengine.data import DataLoader, RandomSampler
from megengine.data.dataset import Dataset
from megengine.tensor import Parameter
from megengine.optimizer import Adam
from megengine.autodiff import GradManager
from megengine.autodiff import Function


def calcScaleZeroPoint(min_val, max_val, num_bits=8):
    qmin = 0.
    qmax = 2. ** num_bits - 1.
    scale = float((max_val - min_val) / (qmax - qmin))

    zero_point = qmax - max_val / scale

    if zero_point < qmin:
        zero_point = qmin
    elif zero_point > qmax:
        zero_point = qmax

    zero_point = int(zero_point)

    return scale, zero_point


def quantize_tensor(x, scale, zero_point, num_bits=8, signed=False):
    if signed:
        qmin = - 2. ** (num_bits - 1)
        qmax = 2. ** (num_bits - 1) - 1
    else:
        qmin = 0.
        qmax = 2. ** num_bits - 1.

    q_x = zero_point + x / scale
    q_x.clamp_(qmin, qmax).round_()

    return q_x


def dequantize_tensor(q_x, scale, zero_point):
    return scale * (q_x - zero_point)


class QParam:

    def __init__(self, num_bits=8):
        self.num_bits = num_bits
        self.scale = None
        self.zero_point = None
        self.min = None
        self.max = None

    def update(self, tensor):
        if self.max is None or self.max < tensor.max():
            self.max = tensor.max()
        self.max = 0 if self.max < 0 else self.max

        if self.min is None or self.min > tensor.min():
            self.min = tensor.min()
        self.min = 0 if self.min > 0 else self.min

        self.scale, self.zero_point = calcScaleZeroPoint(self.min, self.max, self.num_bits)

    def quantize_tensor(self, tensor):
        return quantize_tensor(tensor, self.scale, self.zero_point, num_bits=self.num_bits)

    def dequantize_tensor(self, q_x):
        return dequantize_tensor(q_x, self.scale, self.zero_point)

    def __str__(self):
        info = 'scale: %.10f ' % self.scale
        info += 'zp: %d ' % self.zero_point
        info += 'min: %.6f ' % self.min
        info += 'max: %.6f' % self.max
        return info


class QModule(M.Module):

    def __init__(self, qi=True, qo=True, num_bits=8):
        super(QModule, self).__init__()
        if qi:
            self.qi = QParam(num_bits=num_bits)
        if qo:
            self.qo = QParam(num_bits=num_bits)

    def freeze(self):
        pass

    def quantize_inference(self, x):
        raise NotImplementedError('quantize_inference should be implemented.')


class QLinear(QModule):

    def __init__(self, fc_module, qi=True, qo=True, num_bits=8):
        super(QLinear, self).__init__(qi=qi, qo=qo, num_bits=num_bits)
        self.num_bits = num_bits
        self.fc_module = fc_module
        self.qw = QParam(num_bits=num_bits)

    def freeze(self, qi=None, qo=None):

        if hasattr(self, 'qi') and qi is not None:
            raise ValueError('qi has been provided in init function.')
        if not hasattr(self, 'qi') and qi is None:
            raise ValueError('qi is not existed, should be provided.')

        if hasattr(self, 'qo') and qo is not None:
            raise ValueError('qo has been provided in init function.')
        if not hasattr(self, 'qo') and qo is None:
            raise ValueError('qo is not existed, should be provided.')

        if qi is not None:
            self.qi = qi
        if qo is not None:
            self.qo = qo
        self.M = self.qw.scale * self.qi.scale / self.qo.scale

        self.fc_module.weight.data = self.qw.quantize_tensor(self.fc_module.weight.data)
        self.fc_module.weight.data = self.fc_module.weight.data - self.qw.zero_point
        self.fc_module.bias.data = quantize_tensor(self.fc_module.bias.data, scale=self.qi.scale * self.qw.scale,
                                                   zero_point=0, num_bits=32, signed=True)

    def forward(self, x):
        if hasattr(self, 'qi'):
            self.qi.update(x)

        self.qw.update(self.fc_module.weight.data)

        if hasattr(self, 'qo'):
            self.qo.update(x)

        return x

    def quantize_inference(self, x):
        x = x - self.qi.zero_point
        x = self.fc_module(x)
        x = self.M * x
        x.round_()
        x = x + self.qo.zero_point
        x.clamp_(0., 2. ** self.num_bits - 1.).round_()
        return x


class QConv2d(QModule):

    def __init__(self, conv_module, qi=True, qo=True, num_bits=8):
        super(QConv2d, self).__init__(qi=qi, qo=qo, num_bits=num_bits)
        self.num_bits = num_bits
        self.conv_module = conv_module
        self.qw = QParam(num_bits=num_bits)

    def freeze(self, qi=None, qo=None):

        if hasattr(self, 'qi') and qi is not None:
            raise ValueError('qi has been provided in init function.')
        if not hasattr(self, 'qi') and qi is None:
            raise ValueError('qi is not existed, should be provided.')

        if hasattr(self, 'qo') and qo is not None:
            raise ValueError('qo has been provided in init function.')
        if not hasattr(self, 'qo') and qo is None:
            raise ValueError('qo is not existed, should be provided.')

        if qi is not None:
            self.qi = qi
        if qo is not None:
            self.qo = qo
        self.M = self.qw.scale * self.qi.scale / self.qo.scale

        self.conv_module.weight.data = self.qw.quantize_tensor(self.conv_module.weight.data)
        self.conv_module.weight.data = self.conv_module.weight.data - self.qw.zero_point

        self.conv_module.bias.data = quantize_tensor(self.conv_module.bias.data, scale=self.qi.scale * self.qw.scale,
                                                     zero_point=0, num_bits=32, signed=True)

    def forward(self, x):
        if hasattr(self, 'qi'):
            self.qi.update(x)

        self.qw.update(self.conv_module.weight.data)

        if hasattr(self, 'qo'):
            self.qo.update(x)

        return x

    def quantize_inference(self, x):
        x = x - self.qi.zero_point
        x = self.conv_module(x)
        x = self.M * x
        x.round_()
        x = x + self.qo.zero_point
        x.clamp_(0., 2. ** self.num_bits - 1.).round_()
        return x


class QReLU(QModule):

    def __init__(self, qi=False, num_bits=None):
        super(QReLU, self).__init__(qi=qi, num_bits=num_bits)

    def freeze(self, qi=None):

        if hasattr(self, 'qi') and qi is not None:
            raise ValueError('qi has been provided in init function.')
        if not hasattr(self, 'qi') and qi is None:
            raise ValueError('qi is not existed, should be provided.')

        if qi is not None:
            self.qi = qi

    def forward(self, x):
        if hasattr(self, 'qi'):
            self.qi.update(x)

        x = F.relu(x)

        return x

    def quantize_inference(self, x):
        x = x.clone()
        x[x < self.qi.zero_point] = self.qi.zero_point
        return x


class QConvBNReLU(QModule):

    def __init__(self, conv_module, bn_module, qi=True, qo=True, num_bits=8):
        super(QConvBNReLU, self).__init__(qi=qi, qo=qo, num_bits=num_bits)
        self.num_bits = num_bits
        self.conv_module = conv_module
        self.bn_module = bn_module
        self.qw = QParam(num_bits=num_bits)
        self.qb = QParam(num_bits=32)

    def fold_bn(self, mean, std):
        if self.bn_module.affine:
            gamma_ = self.bn_module.weight / std
            weight = self.conv_module.weight * gamma_.reshape(self.conv_module.out_channels, 1, 1, 1)
            if self.conv_module.bias is not None:
                bias = gamma_ * self.conv_module.bias - gamma_ * mean + self.bn_module.bias
            else:
                bias = self.bn_module.bias - gamma_ * mean
        else:
            gamma_ = 1 / std
            weight = self.conv_module.weight * gamma_
            if self.conv_module.bias is not None:
                bias = gamma_ * self.conv_module.bias - gamma_ * mean
            else:
                bias = -gamma_ * mean

        return weight, bias

    def forward(self, x):

        if hasattr(self, 'qi'):
            self.qi.update(x)

        if self.training:
            y = F.conv2d(x, self.conv_module.weight, self.conv_module.bias,
                         stride=self.conv_module.stride,
                         padding=self.conv_module.padding,
                         dilation=self.conv_module.dilation,
                         groups=self.conv_module.groups)
            y = y.permute(1, 0, 2, 3)  # NCHW -> CNHW
            y = y.contiguous().reshape(self.conv_module.out_channels, -1)  # CNHW -> C,NHW
            # mean = y.mean(1)
            # var = y.var(1)
            mean = y.mean(1).detach()
            var = y.var(1).detach()
            self.bn_module.running_mean = \
                self.bn_module.momentum * self.bn_module.running_mean + \
                (1 - self.bn_module.momentum) * mean
            self.bn_module.running_var = \
                self.bn_module.momentum * self.bn_module.running_var + \
                (1 - self.bn_module.momentum) * var
        else:
            mean = tensor(self.bn_module.running_mean)
            var = tensor(self.bn_module.running_var)

        std = F.sqrt(var + self.bn_module.eps)

        weight, bias = self.fold_bn(mean, std)

        self.qw.update(weight)

        x = F.relu(x)

        if hasattr(self, 'qo'):
            self.qo.update(x)

        return x

    def freeze(self, qi=None, qo=None):
        if hasattr(self, 'qi') and qi is not None:
            raise ValueError('qi has been provided in init function.')
        if not hasattr(self, 'qi') and qi is None:
            raise ValueError('qi is not existed, should be provided.')

        if hasattr(self, 'qo') and qo is not None:
            raise ValueError('qo has been provided in init function.')
        if not hasattr(self, 'qo') and qo is None:
            raise ValueError('qo is not existed, should be provided.')

        if qi is not None:
            self.qi = qi
        if qo is not None:
            self.qo = qo
        self.M = self.qw.scale * self.qi.scale / self.qo.scale

        weight, bias = self.fold_bn(self.bn_module.running_mean, self.bn_module.running_var)
        self.conv_module.weight.data = self.qw.quantize_tensor(weight.data)
        self.conv_module.weight.data = self.conv_module.weight.data - self.qw.zero_point

        self.conv_module.bias.data = quantize_tensor(bias, scale=self.qi.scale * self.qw.scale,
                                                     zero_point=0, num_bits=32, signed=True)

    def quantize_inference(self, x):
        x = x - self.qi.zero_point
        x = self.conv_module(x)
        x = self.M * x
        x.round_()
        x = x + self.qo.zero_point
        x.clamp_(0., 2. ** self.num_bits - 1.).round_()
        return x


class Net(M.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv0 = M.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.relu0 = M.ReLU()
        self.bn0 = M.BatchNorm2d(64)
        self.conv1 = M.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu1 = M.ReLU()
        self.bn1 = M.BatchNorm2d(64)
        self.conv2 = M.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = M.ReLU()
        self.bn2 = M.BatchNorm2d(64)
        self.conv3 = M.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu3 = M.ReLU()
        self.bn3 = M.BatchNorm2d(64)
        self.conv4 = M.Conv2d(64, 128, kernel_size=1, stride=1, padding=0)
        self.relu4 = M.ReLU()
        self.fc = M.Linear(10368, 81 * 9)

    def forward(self, x):
        x = self.conv0(x)
        x = self.relu0(x)
        x = self.bn0(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.bn3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = F.flatten(x, 1)
        x = self.fc(x)
        x = F.reshape(x, (-1, 81, 9))
        return x

    def quantize(self, num_bits=8):
        self.qconv0 = QConvBNReLU(self.conv0, self.bn0, qi=True, qo=True, num_bits=num_bits)
        self.qconv1 = QConvBNReLU(self.conv1, self.bn1, qi=False, qo=True, num_bits=num_bits)
        self.qconv2 = QConvBNReLU(self.conv2, self.bn2, qi=False, qo=True, num_bits=num_bits)
        self.qconv3 = QConvBNReLU(self.conv3, self.bn3, qi=False, qo=True, num_bits=num_bits)
        self.qconv4 = QConv2d(self.conv4, qi=False, qo=False, num_bits=num_bits)
        self.qrelu0 = QReLU(qi=False, num_bits=num_bits)
        self.qfc = QLinear(self.fc, qi=False, qo=True, num_bits=num_bits)

    def quantize_forward(self, x):
        x = self.qconv0(x)
        x = self.qconv1(x)
        x = self.qconv2(x)
        x = self.qconv3(x)
        x = self.qconv4(x)
        x = self.qrelu0(x)
        x = self.qfc(x)
        return x

    def freeze(self):
        self.qconv0.freeze()
        self.qconv1.freeze(qi=self.qconv0.qo)
        self.qconv2.freeze(qi=self.qconv1.qo)
        self.qconv3.freeze(qi=self.qconv2.qo)
        self.qconv4.freeze(qi=self.qconv3.qo)
        self.qrelu0.freeze(qi=self.qconv4.qo)
        #
        self.qfc.freeze(qi=self.qconv4.qo)

    def quantize_inference(self, x):
        qx = self.qconv0.qi.quantize_tensor(x)
        qx = self.qconv0.quantize_inference(qx)
        qx = self.qconv1.quantize_inference(qx)
        qx = self.qconv2.quantize_inference(qx)
        qx = self.qconv3.quantize_inference(qx)
        qx = self.qconv4.quantize_inference(qx)
        qx = self.qrelu0.quantize_inference(qx)
        # print(qx.shape())
        # qx = qx.view(-1, 5*5*40)

        qx = self.qfc.quantize_inference(qx)

        out = self.qfc.qo.dequantize_tensor(qx)
        return out


model = Net()


def get_data(file):
    data = pd.read_csv(file)
    feat_raw = data['quizzes']
    label_raw = data['solutions']
    feat = []
    label = []

    for i in feat_raw:
        x = np.array([int(j) for j in i]).reshape((1, 9, 9))
        feat.append(x)
    feat = np.array(feat)
    feat = feat / 9
    feat -= .5
    for i in label_raw:
        x = np.array([int(j) for j in i]).reshape((1, 81)) - 1
        label.append(x)
    label = np.array(label)

    del (feat_raw)
    del (label_raw)

    x_train, x_test, y_train, y_test = train_test_split(feat, label, test_size=0.2, random_state=42)

    return x_train, x_test, y_train, y_test


x_train, x_test, y_train, y_test = get_data('./dataset/dataset-2115/sudoku.csv')


class TrainDataset(Dataset):
    def __init__(self, x_train, y_train):
        super().__init__()
        self.data_x = x_train
        self.data_y = y_train

    def __getitem__(self, index: int) -> Tuple:
        return self.data_x[index], self.data_y[index]

    def __len__(self) -> int:
        return len(self.data_x)


train_dataset = TrainDataset(x_train, y_train)
train_sampler = RandomSampler(dataset=train_dataset, batch_size=32)
train_dataloader = DataLoader(dataset=train_dataset, sampler=train_sampler)

opt = Adam(model.parameters(), lr=1e-3)


def direct_quantize(model, test_loader):
    for i, (data, target) in enumerate(test_loader, 1):
        output = model.forward(data)
        if i % 500 == 0:
            break
    print('direct quantization finish')


epochs = 0
for epoch in range(epochs):
    iter = 0
    with tqdm(total=train_dataloader.__len__() / 100, desc="epoch {}:training process".format(epoch)) as tq:
        for i, (feat, label) in enumerate(train_dataloader):

            gm = GradManager().attach(model.parameters())
            with gm:
                logits = model(tensor(feat))
                label = label.reshape(32, 81)
                loss = F.loss.cross_entropy(logits, label, axis=2)

                iter += 1
                if iter % 100 == 0:
                    tq.set_postfix({"loss": "{0:1.5f}".format(loss.numpy().item()), })
                    tq.update(1)

                gm.backward(loss)
            opt.step()
            opt.clear_grad()

mge.save(model, "1.mge")
print("saved")
num_bits = 8
model.quantize(num_bits=num_bits)
model.eval()
print('Quantization bit: %d' % num_bits)

def valid_sudo(sudo):
    all = []
    for i in range(27):
        one = []
        all.append(one)
    for i in range(9):
        for j in range(9):
            num = sudo[i, j]
            if num not in all[i]:
                all[i].append(num)
            else:
                return 1
            if num not in all[j + 9]:
                all[j + 9].append(num)
            else:
                return 1
            if num not in all[3 * (i // 3) + j // 3 + 18]:
                all[3 * (i // 3) + j // 3 + 18].append(num)
            else:
                return 1


def inference_sudoku(sample):
    feat = copy.copy(sample)

    while (1):
        out = F.softmax(model(tensor(feat.reshape((1, 1, 9, 9)))), axis=2)
        out = out.reshape(81, 9)

        pred = F.argmax(out, axis=1).reshape((9, 9)) + 1
        prob = np.around(F.max(out, axis=1).reshape((9, 9)), 2)
        feat = denorm(feat).reshape((9, 9))
        mask = (feat == 0)
        if mask.sum() == 0:
            break

        prob_new = prob * mask
        ind = F.argmax(prob_new)
        x, y = (ind // 9), (ind % 9)

        val = pred[x][y].numpy()
        feat[x][y] = int(val)
        feat = norm(feat)

    return feat

model = Net()
model = mge.load("1.mge")
model.eval()

eval_dataset = TrainDataset(x_test, y_test)
eval_sampler = RandomSampler(dataset = eval_dataset, batch_size=32)
eval_dataloader = DataLoader(dataset = eval_dataset, sampler=eval_sampler)

average_loss = 0
iter = 0
sum = 0
with tqdm(total=eval_dataloader.__len__() / 100, desc="evaluating process") as tq:
    for i, (feat, label) in enumerate(eval_dataloader):
            logits = model(tensor(feat))
            label = label.reshape(32, 81)
            legal_loss = 0
            for i in range(32):
                game = feat[i].reshape((9,9))
                legal_loss += valid_sudo(game)/32
            print(legal_loss)
            loss = F.loss.cross_entropy(logits, label, axis=2)
            sum += legal_loss
            iter += 1
            if iter % 100 == 0:
                tq.set_postfix({"loss": "{0:1.5f}".format(loss.numpy().item()),})
                tq.update(1)
            average_loss += loss.numpy().item()

print(average_loss / eval_dataloader.__len__())
print(sum / eval_dataloader.__len__())

model = Net()
model = mge.load("1.mge")
model.eval()


def norm(x):
    return (x / 9) - .5


def denorm(x):
    return (x + .5) * 9


def solve_sudoku(game):
    game = game.replace('\n', '')
    game = game.replace(' ', '')
    game = np.array([int(j) for j in game]).reshape((9, 9, 1))
    game = norm(game)
    game = inference_sudoku(game)
    return game


game = '''
          0 8 0 0 3 2 0 0 1
          7 0 3 0 8 0 0 0 2
          5 0 0 0 0 7 0 3 0
          0 5 0 0 0 1 9 7 0
          6 0 0 7 0 9 0 0 8
          0 4 7 2 0 0 0 5 0
          0 2 0 6 0 0 0 0 9
          8 0 0 0 9 0 3 0 5
          3 0 0 8 2 0 0 1 0
      '''

game = solve_sudoku(game)

print('solved puzzle:\n')
print(game.astype("uint8"))
