########## 第一阶段：计算降维矩阵V：平均梯度算法
########## The first stage: Calculate the dimensionality reduction matrix V: average gradient algorithm
import sys
import argparse
import os
import math
import torch
import torch.nn as nn
import numpy as np
import torch.multiprocessing as mp
import torch.distributed as dist
from github_utils import initialize
from github_model import resnet18
from github_dataloader import IN_DATA
from datetime import timedelta
from functorch import make_functional_with_buffers, vmap, grad, jvp, vjp
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(global_rank, world_size, port='12345'):
    # 配置Master Node的信息
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port

    # 初始化Process Group
    # 关于init_method, 参数详见https://pytorch.org/docs/stable/distributed.html#initialization
    dist.init_process_group("nccl", init_method='env://', rank=global_rank, world_size=world_size, timeout=timedelta(seconds=5))

def cleanup():
    dist.destroy_process_group()

class Logger(object):
    def __init__(self, logFile="Default.log"):
        self.terminal = sys.stdout
        self.log = open(logFile, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adaptive", default=False, type=bool, help="True if you want to use the Adaptive SAM.")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size used in the training and validation loop.")
    parser.add_argument("--depth", default=16, type=int, help="Number of layers.")
    parser.add_argument("--dropout", default=0.0, type=float, help="Dropout rate.")
    parser.add_argument("--epochs", default=200, type=int, help="Total number of epochs.")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0.0 for no label smoothing.")
    parser.add_argument("--learning_rate", default=0.1, type=float, help="Base learning rate at the start of the training.")
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
    parser.add_argument("--threads", default=0, type=int, help="Number of CPU threads for dataloaders.")
    #parser.add_argument("--rho", default=0.05, type=int, help="Rho parameter for SAM.")
    parser.add_argument("--rho", default=0.05, type=float, help="Rho parameter for SAM.")
    parser.add_argument("--weight_decay", default=0.0005, type=float, help="L2 weight decay.")
    parser.add_argument("--width_factor", default=8, type=int, help="How many times wider compared to normal ResNet.")
    parser.add_argument("--data", default='cifar10', type=str, help="cifar10 or cifar100.")
    parser.add_argument("--model", default='resnet18', type=str, help="resnet18 or wideresnet or VGG16BN")
    parser.add_argument("--save_name", default='1', type=str, help="index of model")
    parser.add_argument("--kernel", default='NTK', type=str, help="NTK or NFK")
    parser.add_argument('--k', default=128, type=int, help='dimension reduce')
    parser.add_argument('--start_k', default=0, type=int, help='start dimension reduce')
    parser.add_argument('--interval_k', default=5, type=int, help='interval dimension reduce')
    parser.add_argument('--sample_num', default=1000, type=int, help='sample number of fisher kernel')
    parser.add_argument('--print_freq', default=100, type=int, help='print frequency')
    ################ 多卡
    parser.add_argument('--seed', default=42, type=int, help='randomize seed')
    parser.add_argument('--nproc_per_node', type=int, default=2, help='number of process in each node, equal to number of gpus')
    parser.add_argument('--nnode', type=int, default=1, help='computer number')
    parser.add_argument('--node_rank', type=int, default=0, help='computer rank')
    parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')
    parser.add_argument("--num_gpus", default=1, type=int, help="Number of gpus.")
    parser.add_argument("--port", default='12345', type=str, help="master port")
    args = parser.parse_args()
    return args

def test(data_loader, model, device):
    num = 0
    num_all = 0
    model.eval()
    for batch in data_loader:
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)
      
        predictions = model(inputs)

        correct = (torch.argmax(predictions, 1) == targets)
        num = num + (torch.nonzero(correct==True)).shape[0]
        num_all = num_all + targets.shape[0]
        print(num_all)
    acc = num/num_all
    print('test_acc:', acc)
    return acc

class Energy_Model(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        y = torch.log(torch.sum(torch.exp(self.model(x)),dim=1))
        # print('y:', y.shape)
        return y
    
def reduce_value(value, op=dist.ReduceOp.SUM):
    world_size = dist.get_world_size()
    if world_size < 2:  # single GPU
        return value
 
    with torch.no_grad():
        dist.all_reduce(value, op)
        return value
    

def get_model_grad_vec_torch_2(model):
    vec = []
    for n,p in model.named_parameters():
        vec.append(p.grad.data.detach().reshape(-1)) 
    return torch.cat(vec, 0)


def pre_compute(model, dataloader, save_dir, device, rank):
    sum_ = 0
    nexample = 0
    for batch in dataloader:
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)

        nexample += inputs.shape[0]
        model.zero_grad()
        
        y = model(inputs)
        loss = torch.mean(y)
        loss.backward()
        a = get_model_grad_vec_torch_2(model)
        sum_ = sum_ + a*inputs.shape[0]
    
    nexample = torch.Tensor([nexample]).to(device)
    torch.distributed.barrier()
    sum_ = reduce_value(sum_)
    nexample = reduce_value(nexample)
    z_theta = sum_/nexample
    print('nexample:', nexample)
    if rank == 0:
        np.save(save_dir + 'z_theta.npy', z_theta.cpu().detach().numpy())

    L = 0
    for batch in dataloader:
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)
        for i in range(inputs.shape[0]):
            model.zero_grad()
            
            y_i = model(inputs[i].unsqueeze(0))
            y_i.backward()
            z_i = get_model_grad_vec_torch_2(model)
            u_i =  z_i - z_theta
            ### diagonal approximation
            L_i = torch.square(u_i)
            L = L + L_i
    torch.distributed.barrier()
    L = reduce_value(L)
    L = L / nexample
    # L^(-1/2)
    I_2 = 1/ torch.sqrt(L)
    #### count inf_value in I_2
    count = (I_2==np.inf).sum()
    ratio = 100*(count/len(I_2))
    print('inf-value, count:', count, 'ratio:', float(ratio), '%')
    #### deal with inf_value in I_2
    I_2_ = torch.where(torch.isinf(I_2), torch.full_like(I_2, 1), I_2)
    I_2 = torch.where(I_2_==1, torch.full_like(I_2_, float(I_2_.max())), I_2_)

    return z_theta, I_2

######## calculate average gradient per class(label)
def cal_grad_per_class(model, dataloader, label, kernel, z_theta, I_2, save_dir, device):
    mean_v = 0 
    sum_v = 0
    num = 0
    print('################ label=', label, ' ################')
    
    for batch in dataloader:
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        indices = torch.nonzero(targets==label)
        previous_num = num
        if indices.shape[0]!=0:
            inputs = inputs[indices[:,0]]
            targets = targets[indices[:,0]]
            # print(targets[0])
            model.zero_grad()
            y = model(inputs)
            loss = torch.sum(y)
            loss.backward()
            sum_v = sum_v + get_model_grad_vec_torch_2(model)
            num = num + inputs.shape[0]
            print('num:', num)
    
    num = torch.Tensor([num]).to(device)
    torch.distributed.barrier()
    sum_v = reduce_value(sum_v)
    num = reduce_value(num)

    mean_v = sum_v/num  
    print('mean_v:', mean_v.shape)
    if kernel == 'NFK':
        mean_v =  mean_v - z_theta
        mean_v = I_2 * mean_v 
    mean_v = mean_v.detach().cpu()
    np.save(save_dir + str(label) +'.npy', mean_v.numpy())
    # np.save(save_dir +'cifar10/avg_grad_per_class/'+ str(label) +'.npy', mean_v.numpy())
    return mean_v

######## calculate G(x) for batch x
def cal_fisher_vector(model, x, kernel='NTK', z_theta=None, I_2=None):
    bz = x.shape[0]
    fmodel, params, buffers = make_functional_with_buffers(model)
    def compute_loss_stateless_model(params, buffers, sample):
        batch = sample.unsqueeze(0)
        predictions = fmodel(params, buffers, batch).squeeze()
        return predictions
    ft_compute_grad = grad(compute_loss_stateless_model)
    ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0))
    ft_per_sample_grads = ft_compute_sample_grad(params, buffers, x)
    ft_per_sample_grads_ = []
    for i in range(len(ft_per_sample_grads)):
        ft_per_sample_grads_.append(ft_per_sample_grads[i].view(bz, -1))
    ft_per_sample_grads_ = torch.cat(ft_per_sample_grads_, dim=1)

    if kernel == 'NFK':
        ft_per_sample_grads_ =  ft_per_sample_grads_ - z_theta
        def dot_per_element(I_2, z_i):
            out = I_2 * z_i
            return out
        dot = vmap(dot_per_element, in_dims=(None, 0))
        ft_per_sample_grads_ = dot(I_2, ft_per_sample_grads_)

    return ft_per_sample_grads_

######## 
def try_2(model, train_dataset, test_dataloader, kernel, z_theta, I_2, num_classes, save_dir, save_name):
    knn_feature = []
    for i in range(999, num_classes):
    # for i in range(43):
        reconstruct_error = []
        if not os.path.exists(save_dir + 'imagenet/avg_grad_per_class/' + str(i) +'.npy'):
            avg_grad = cal_grad_per_class(model, train_dataset, i, kernel, z_theta, I_2, save_dir)
        else:
            avg_grad = torch.from_numpy(np.load(save_dir + 'imagenet/avg_grad_per_class/' +str(i) +'.npy'))
        num = 0
        for batch in test_dataloader:
            inputs, targets = batch
            inputs = inputs.cuda()
            targets = targets.cuda()
            grad_feature = cal_fisher_vector(model, inputs, kernel=kernel, z_theta=z_theta, I_2=I_2).detach().cpu()
            feature = grad_feature @ avg_grad.unsqueeze(1)
            reconstruct_error.append(feature)
            num = num + feature.shape[0]
            if num > 50000:
                break
            print(num)
        reconstruct_error = torch.cat(reconstruct_error, dim=0)
        np.save(save_dir + 'imagenet/' + save_name + str(i) + '.npy', reconstruct_error.numpy())
        knn_feature.append(reconstruct_error)
    knn_feature = torch.cat(knn_feature, dim=1)
    return knn_feature


def main(local_rank, args):
    # 计算global_rank和world_size
    global_rank = local_rank + args.node_rank * args.nproc_per_node
    world_size = args.nnode * args.nproc_per_node
    print('global_rank:', global_rank, 'world_size:', world_size)
    setup(global_rank=global_rank, world_size=world_size, port=args.port)
    # 设置seed
    # torch.manual_seed(args.seed)
    initialize(args, seed=args.seed)
    # 输出记录log
    file_name=os.path.basename(__file__).split(".")[0]
    save_dir = file_name + '/' + args.model + '_' + args.data + '_Average_Gradient_extraction/' + args.kernel + '_sample_num=' + str(args.sample_num) + '_K=' + str(args.k) + '/'
    os.makedirs(save_dir, exist_ok=True)
    print('save_dir: ', save_dir)
    sys.stdout = Logger(save_dir + 'output.log')

    # 设置数据集
    if args.data == 'mnist':
        dataset = MNIST(args.batch_size, args.threads, args.num_gpus)
    else:
        dataset = IN_DATA(args.data, args.batch_size, args.threads, args.nproc_per_node)

    # 创建模型
    if args.model == 'vgg16bn':
        model = VGG16BN(10 if args.data == 'cifar10' else 100)
    elif args.model == 'vgg19bn':
        model = VGG19BN(10 if args.data == 'cifar10' else 100)
    elif args.model == 'resnet18':
        model = resnet18(10 if args.data == 'cifar10' else 100)
    elif args.model == 'wideresnet':
        model = WideResNet(args.depth, args.width_factor, args.dropout, in_channels=3, labels=10 if args.data == 'cifar10' else 100)
    elif args.model == 'lenet':
        model = LeNet()
    elif args.model == 'resnet50':
        import torchvision
        from torchvision import models
        model = models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    
    ##### load model
    if args.data == 'cifar10':
        checkpoint = torch.load('2023_03_06_compare_sgd/resnet18_cifar10_best_rho=0.05_labsmooth=0.1/seed_42.pt')
        print('epoch:', checkpoint['epoch'])
        model.load_state_dict(checkpoint['model'], strict=False)
        model.set_nor(False)
    elif args.data == 'mnist':
        checkpoint = torch.load('2023_03_06_compare_sgd/lenet_mnist_best_rho=0.05_labsmooth=0.0/seed_42.pt')
        print('epoch:', checkpoint['epoch'])
        model.load_state_dict(checkpoint['model'], strict=False)
    elif args.data == 'cifar100':
        checkpoint = torch.load('2023_03_06_compare_sgd/resnet18_cifar100_best_rho=0.05.pt')
        print('epoch:', checkpoint['epoch'])
        model.load_state_dict(checkpoint['model'], strict=False)
        model.set_nor(False)


    # 移动模型到local_rank对应的GPU上
    model = model.to(local_rank)
    # model = DDP(model.cuda(), device_ids=[local_rank], output_device=local_rank)
    # 测试模型准确率   
    device = torch.device('cuda:'+ str(local_rank))
    with torch.no_grad():
        test(dataset.test, model, device)
    #######
    model = Energy_Model(model)
    model.eval()
    
    ######################## obtain z_theta, I_2, p
    sample_num = args.sample_num
    kernel_dataset, _ = torch.utils.data.random_split(dataset.train_set, [sample_num, len(dataset.train_set)-sample_num], generator=torch.Generator().manual_seed(0))
    sample_num_per_gpu = math.floor(sample_num/world_size)
    if local_rank != world_size-1:
        start_sample_num = local_rank*sample_num_per_gpu
        sub_sample_num = sample_num_per_gpu
        end_sample_num = start_sample_num + sub_sample_num
    else:
        start_sample_num = local_rank*sample_num_per_gpu
        sub_sample_num = sample_num-(world_size-1)*sample_num_per_gpu
        end_sample_num = start_sample_num + sub_sample_num
    indices = [i for i in range(start_sample_num, end_sample_num)]
    sub_kernel_dataset = torch.utils.data.Subset(kernel_dataset, indices)
    print('dataset_len:', len(sub_kernel_dataset))
    kernel_dataloader = torch.utils.data.DataLoader(sub_kernel_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.threads)
    print('finish data load!')

    if os.path.exists(save_dir + 'I_2.npy'):
        z_theta = torch.from_numpy(np.load(save_dir + 'z_theta.npy')).to(device)
        I_2 = torch.from_numpy(np.load(save_dir + 'I_2.npy')).to(device)
    else:
        z_theta, I_2 = pre_compute(model, kernel_dataloader, save_dir, device, local_rank)
        torch.distributed.barrier()
        if local_rank == 0:
            np.save(save_dir + 'z_theta.npy', z_theta.cpu().detach().numpy()) ### mean matrix
            np.save(save_dir + 'I_2.npy', I_2.cpu().detach().numpy())   ### (variance matrix)^(-1/2)

    print('z_theta:', z_theta.shape, 'I_2:', I_2.shape)

    ################# extract average gradient per class
    num_classes = 1000
    for i in range(0, num_classes):
        cal_grad_per_class(model, kernel_dataloader, i, args.kernel, z_theta, I_2, save_dir, device)

    return
    

if __name__ == "__main__":
    args = set_args()
    # mp.spawn(main, args=(args,), nprocs=args.nproc_per_node)
    mp.set_start_method("spawn")

    processes = []
    try:
        for rank in range(args.nproc_per_node):
            p = mp.Process(target=main, args=(rank, args))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print('catch keyboardinterupterror')
        pid=os.getpid()
        os.popen('taskkill.exe /f /pid:%d'%pid) #在unix下无需此命令，但在windows下需要此命令强制退出当前进程
    except Exception as e:
        print(e)
    else:
        print('quit normally')







