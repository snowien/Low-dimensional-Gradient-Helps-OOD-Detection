import torch
import random
import time
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm

import seaborn as sns
import torch
import numpy as np
import matplotlib.pyplot as plt  
import pandas as pd

######################################## log #######################################################

class LoadingBar:
    def __init__(self, length: int = 40):
        self.length = length
        self.symbols = ['┈', '░', '▒', '▓']

    def __call__(self, progress: float) -> str:
        p = int(progress * self.length*4 + 0.5)
        d, r = p // 4, p % 4
        return '┠┈' + d * '█' + ((self.symbols[r]) + max(0, self.length-1-d) * '┈' if p < self.length*4 else '') + "┈┨"


class Log:
    def __init__(self, log_each: int, initial_epoch=-1):
        self.loading_bar = LoadingBar(length=27)
        self.best_accuracy = 0.0
        self.log_each = log_each
        self.epoch = initial_epoch
        self.is_train = False

    def train(self, len_dataset: int) -> None:
        self.epoch += 1
        if self.epoch == 0:
            self._print_header()
        else:
            self.flush()

        self.is_train = True
        self.last_steps_state = {"loss": 0.0, "accuracy": 0.0, "steps": 0}
        self._reset(len_dataset)

    def eval(self, len_dataset: int) -> None:
        self.flush()
        self.is_train = False
        self._reset(len_dataset)

    def __call__(self, model, loss, accuracy, learning_rate: float = None) -> None:
        if self.is_train:
            self._train_step(model, loss, accuracy, learning_rate)
        else:
            self._eval_step(loss, accuracy)

    def flush(self) -> None:
        if self.is_train:
            loss = self.epoch_state["loss"] / self.epoch_state["steps"]
            accuracy = self.epoch_state["accuracy"] / self.epoch_state["steps"]

            print(
                f"\r┃{self.epoch:12d}  ┃{loss:12.4f}  │{100*accuracy:10.2f} %  ┃{self.learning_rate:12.3e}  │{self._time():>12}  ┃",
                end="",
                flush=True,
            )

        else:
            loss = self.epoch_state["loss"] / self.epoch_state["steps"]
            accuracy = self.epoch_state["accuracy"] / self.epoch_state["steps"]

            print(f"{loss:12.4f}  │{100*accuracy:10.2f} %  ┃", flush=True)

            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy

    def _train_step(self, model, loss, accuracy, learning_rate: float) -> None:
        self.learning_rate = learning_rate
        self.last_steps_state["loss"] += loss.sum().item()
        self.last_steps_state["accuracy"] += accuracy.sum().item()
        self.last_steps_state["steps"] += loss.size(0)
        self.epoch_state["loss"] += loss.sum().item()
        self.epoch_state["accuracy"] += accuracy.sum().item()
        self.epoch_state["steps"] += loss.size(0)
        self.step += 1

        if self.step % self.log_each == self.log_each - 1:
            loss = self.last_steps_state["loss"] / self.last_steps_state["steps"]
            accuracy = self.last_steps_state["accuracy"] / self.last_steps_state["steps"]

            self.last_steps_state = {"loss": 0.0, "accuracy": 0.0, "steps": 0}
            progress = self.step / self.len_dataset

            print(
                f"\r┃{self.epoch:12d}  ┃{loss:12.4f}  │{100*accuracy:10.2f} %  ┃{learning_rate:12.3e}  │{self._time():>12}  {self.loading_bar(progress)}",
                end="",
                flush=True,
            )

    def _eval_step(self, loss, accuracy) -> None:
        self.epoch_state["loss"] += loss.sum().item()
        self.epoch_state["accuracy"] += accuracy.sum().item()
        self.epoch_state["steps"] += loss.size(0)

    def _reset(self, len_dataset: int) -> None:
        self.start_time = time.time()
        self.step = 0
        self.len_dataset = len_dataset
        self.epoch_state = {"loss": 0.0, "accuracy": 0.0, "steps": 0}

    def _time(self) -> str:
        time_seconds = int(time.time() - self.start_time)
        return f"{time_seconds // 60:02d}:{time_seconds % 60:02d} min"

    def _print_header(self) -> None:
        print(f"┏━━━━━━━━━━━━━━┳━━━━━━━╸T╺╸R╺╸A╺╸I╺╸N╺━━━━━━━┳━━━━━━━╸S╺╸T╺╸A╺╸T╺╸S╺━━━━━━━┳━━━━━━━╸V╺╸A╺╸L╺╸I╺╸D╺━━━━━━━┓")
        print(f"┃              ┃              ╷              ┃              ╷              ┃              ╷              ┃")
        print(f"┃       epoch  ┃        loss  │    accuracy  ┃        l.r.  │     elapsed  ┃        loss  │    accuracy  ┃")
        print(f"┠──────────────╂──────────────┼──────────────╂──────────────┼──────────────╂──────────────┼──────────────┨")

######################################## initialize #######################################################
def initialize(args, seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


class StepLR:
    def __init__(self, optimizer, learning_rate: float, total_epochs: int):
        self.optimizer = optimizer
        self.total_epochs = total_epochs
        self.base = learning_rate

    def __call__(self, epoch):
        if epoch < self.total_epochs * 3/10:
            lr = self.base
        elif epoch < self.total_epochs * 6/10:
            lr = self.base * 0.2
        elif epoch < self.total_epochs * 8/10:
            lr = self.base * 0.2 ** 2
        else:
            lr = self.base * 0.2 ** 3

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]


def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)

def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def smooth_crossentropy(pred, gold, smoothing=0.1):
    n_class = pred.size(1)

    one_hot = torch.full_like(pred, fill_value=smoothing / (n_class - 1))
    one_hot.scatter_(dim=1, index=gold.unsqueeze(1), value=1.0 - smoothing)
    log_prob = F.log_softmax(pred, dim=1)

    loss = F.kl_div(input=log_prob, target=one_hot, reduction='none')
    loss = loss.sum(-1)
   
    return loss


############################################################################################################

class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()  # create instance for 'from utils.plots import colors'

### get forward feature by Hook
# -------------------- 第一步：定义接收feature的函数 ---------------------- #
# 这里定义了一个类，类有一个接收feature的函数hook_fun。定义类是为了方便提取多个中间层。
class HookTool: 
    def __init__(self):
        self.fea = None 

    def hook_fun(self, module, fea_in, fea_out):
        self.fea = fea_out

# ---------- 第二步：注册hook，告诉模型我将在哪些层提取feature,比如提取'fc'后的feature，即output -------- #
def get_feas_by_hook(model, extract_module=['fc']):
    fea_hooks = []
    for n, m in model.named_modules():
        # print('name:', n)
        # # if isinstance(m, extract_module):
        # print(extract_module)
        # if n == 'avg_pool':
        #     print('True')
        if n in extract_module:
            cur_hook = HookTool()
            m.register_forward_hook(cur_hook.hook_fun)
            fea_hooks.append(cur_hook)
            
    return fea_hooks


## feature distribution visualization per channel
# https://blog.csdn.net/hustlei/article/details/123091236
# https://www.jianshu.com/p/eac2c6fa5039
def feature_distribution_per_channel(feature, feature_adv, save_path='fig/feature_density/'):
    os.makedirs(save_path, exist_ok=True)
    print('feature.shape:', feature.shape)
    # preprocess feature, (b,c,h,w)->(b,c,h*w)->(c,b*h*w)
    # feature = feature[:,0:16,:,:]
    # print('fea1:', feature.shape)
    b,c,h,w = feature.shape[0], feature.shape[1], feature.shape[2], feature.shape[3]
    feature = feature.reshape((b,c,h*w))
    fea = []
    for i in range(c):
        fea.append(feature[:,i,:].reshape(-1).unsqueeze(0))
    prepeared_fea = torch.cat(fea,0)
    # print('fea2:', prepeared_fea.shape)

    # preprocess feature, (b,c,h,w)->(b,c,h*w)->(c,b*h*w)
    # feature_adv = feature_adv[:,0:16,:,:]
    # print('fea1_adv:', feature_adv.shape)
    b,c,h,w = feature_adv.shape[0], feature_adv.shape[1], feature_adv.shape[2], feature_adv.shape[3]
    feature_adv = feature_adv.reshape((b,c,h*w))
    fea_adv = []
    for i in range(c):
        fea_adv.append(feature_adv[:,i,:].reshape(-1).unsqueeze(0))
    prepeared_fea_adv = torch.cat(fea_adv,0)
    # print('fea2_adv:', prepeared_fea_adv.shape)

    # construct DataFrame for plotting
    # prepeared_fea = prepeared_fea.detach().cpu().numpy()
    # f_list = [prepeared_fea[i][j] for i in range(c) for j in range(b*h*w)]
    # c_list = [i for i in range(c) for j in range(b*h*w)]
    # print(len(c_list), len(f_list))
    # dict = {'channel':c_list, 'feature':f_list}
    # data = pd.DataFrame(dict)
    # print(data)

    prepeared_fea = prepeared_fea.detach().cpu().numpy()
    prepeared_fea_adv = prepeared_fea_adv.detach().cpu().numpy()

    for i in range(c):
        f_list_clean = [prepeared_fea[i][j] for j in range(b*h*w)]
        bw_method = 1.06 * np.std(prepeared_fea[i]) * math.pow(b*h*w, -1/5)

        f_list_adv = [prepeared_fea_adv[i][j] for j in range(b*h*w)]
        f_list = f_list_clean + f_list_adv

        c_list_clean = ['clean' for j in range(b*h*w)]
        c_list_adv = ['adv' for j in range(b*h*w)]
        c_list = c_list_clean + c_list_adv

        # print(len(c_list), len(f_list))
        dict = {'type':c_list, 'feature':f_list}
        data = pd.DataFrame(dict)

        # plot
        plt.style.use('seaborn')
        try:
            plt.subplot(4, 4, i%16 + 1)
            if i%16==15 or i==c-1:
                legend = True
            else:
                legend = False
            ax = sns.kdeplot(data=data, x="feature", hue="type", multiple="stack", bw_method=bw_method, bw_adjust=1, legend=legend)
            x_min = float(min(prepeared_fea[i].min(), prepeared_fea_adv[i].min()))
            x_max = float(max(prepeared_fea[i].max(), prepeared_fea_adv[i].max()))
            ax.set_xlim(x_min, x_max)
            if i%16!=15:
                ax.set_xticks([]) 
            # ax.set_xlabel('channel_'+str(i))
            # ax.set_ylabel('')
            # ax.set_ylim(0,20)
                
            if i%16==15 or i==c-1:  
                plt.savefig(save_path + 'channel_' + str(int(i-15)) + '_' + str(int(i)) + '.png', dpi=600)
                plt.clf()
        except:
            print('channel-', i , ', plot failed!')
        
    # print(fsdfds)

    return







