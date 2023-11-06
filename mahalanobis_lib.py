from __future__ import print_function
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from scipy.spatial.distance import pdist, cdist, squareform
from visualize_utils import get_feas_by_hook
        

def sample_estimator(model, num_classes, train_loader, extract_module=['layer4']):
    """
    compute sample mean and precision (inverse of covariance)
    return: sample_class_mean: list of class mean
             precision: list of precisions
    """
    import sklearn.covariance

    model.eval()
    group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
    correct, total = 0, 0
    num_output = 1


    num_sample_per_class = np.empty(num_classes)
    num_sample_per_class.fill(0)
    list_features = []
    for i in range(num_output):
        temp_list = []
        for j in range(num_classes):
            temp_list.append(0)
        list_features.append(temp_list)

    for data, target in train_loader:
        total += data.size(0)
        # print(total)
        if total > 50000:
            break
        # data = data.cuda()
        data = Variable(data)
        data = data.cuda()
        out_features = []

        fea_hooks = get_feas_by_hook(model, extract_module=extract_module)  
        output = model(data)
        inter_features = fea_hooks[0].fea
        # print(inter_features.shape)
        # print(fsdfs)
        out_features.append(inter_features)
            
        # get hidden features
        for i in range(num_output):
            out_features[i] = out_features[i].view(out_features[i].size(0), out_features[i].size(1), -1)
            out_features[i] = torch.mean(out_features[i].data, 2)

        # compute the accuracy
        pred = output.data.max(1)[1]
        equal_flag = pred.eq(target.cuda()).cpu()
        correct += equal_flag.sum()

        # construct the sample matrix
        for i in range(data.size(0)):
            label = target[i]
            if num_sample_per_class[label] == 0:
                out_count = 0
                for out in out_features:
                    list_features[out_count][label] = out[i].view(1, -1)
                    out_count += 1
            else:
                out_count = 0
                for out in out_features:
                    list_features[out_count][label] \
                    = torch.cat((list_features[out_count][label], out[i].view(1, -1)), 0)
                    out_count += 1
            num_sample_per_class[label] += 1

    sample_class_mean = []
    out_count = 0
    feature_list = [out_features[0].size(1)]
    print('feature_list:', feature_list)
    for num_feature in feature_list:
        temp_list = torch.Tensor(num_classes, int(num_feature)).cuda()
        for j in range(num_classes):
            temp_list[j] = torch.mean(list_features[out_count][j], 0)
        sample_class_mean.append(temp_list)
        out_count += 1

    precision = []
    for k in range(num_output):
        X = 0
        for i in range(num_classes):
            if i == 0:
                X = list_features[k][i] - sample_class_mean[k][i]
            else:
                X = torch.cat((X, list_features[k][i] - sample_class_mean[k][i]), 0)

        # find inverse
        group_lasso.fit(X.cpu().numpy())
        temp_precision = group_lasso.precision_
        # temp_precision = torch.from_numpy(temp_precision).double().cuda()
        temp_precision = torch.from_numpy(temp_precision).float().cuda()
        precision.append(temp_precision)

    print('\n Training Accuracy:({:.2f}%)\n'.format(100. * correct / total))

    return sample_class_mean, precision



def get_Mahalanobis_score(inputs, model, num_classes=10, sample_mean=None, precision=None, magnitude=0.01, extract_module=['layer4']):
    num_output=1
    for layer_index in range(num_output):
        data = Variable(inputs, requires_grad = True)
        data = data.cuda()

        # out_features = model.intermediate_forward(data)
        fea_hooks = get_feas_by_hook(model, extract_module=extract_module)  
        output = model(data)
        out_features = fea_hooks[0].fea

        out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
        out_features = torch.mean(out_features, 2)

        gaussian_score = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[layer_index][i]
            zero_f = out_features.data - batch_sample_mean
            term_gau = -0.5*torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
            if i == 0:
                gaussian_score = term_gau.view(-1,1)
            else:
                gaussian_score = torch.cat((gaussian_score, term_gau.view(-1,1)), 1)

        # Input_processing
        sample_pred = gaussian_score.max(1)[1]
        batch_sample_mean = sample_mean[layer_index].index_select(0, sample_pred)
        zero_f = out_features - Variable(batch_sample_mean)
        pure_gau = -0.5*torch.mm(torch.mm(zero_f, Variable(precision[layer_index])), zero_f.t()).diag()
        loss = torch.mean(-pure_gau)
        loss.backward()

        gradient =  torch.ge(data.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2

        tempInputs = torch.add(data.data, -magnitude*gradient)

        # noise_out_features = model.intermediate_forward(Variable(tempInputs))
        fea_hooks = get_feas_by_hook(model, extract_module=extract_module)  
        output = model(Variable(tempInputs))
        noise_out_features = fea_hooks[0].fea

        noise_out_features = noise_out_features.view(noise_out_features.size(0), noise_out_features.size(1), -1)
        noise_out_features = torch.mean(noise_out_features, 2)
        noise_gaussian_score = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[layer_index][i]
            zero_f = noise_out_features.data - batch_sample_mean
            term_gau = -0.5*torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
            if i == 0:
                noise_gaussian_score = term_gau.view(-1,1)
            else:
                noise_gaussian_score = torch.cat((noise_gaussian_score, term_gau.view(-1,1)), 1)

        noise_gaussian_score, _ = torch.max(noise_gaussian_score, dim=1)

        noise_gaussian_score = np.asarray(noise_gaussian_score.cpu().numpy(), dtype=np.float32)
        if layer_index == 0:
            Mahalanobis_scores = noise_gaussian_score.reshape((noise_gaussian_score.shape[0], -1))
        else:
            Mahalanobis_scores = np.concatenate((Mahalanobis_scores, noise_gaussian_score.reshape((noise_gaussian_score.shape[0], -1))), axis=1)

    return Mahalanobis_scores


# this function is from https://github.com/xingjunm/lid_adversarial_subspace_detection
def merge_and_generate_labels(X_pos, X_neg):
    """
    merge positve and nagative artifact and generate labels
    return: X: merged samples, 2D ndarray
             y: generated labels (0/1): 2D ndarray same size as X
    """
    X_pos = np.asarray(X_pos, dtype=np.float32)
    X_pos = X_pos.reshape((X_pos.shape[0], -1))

    X_neg = np.asarray(X_neg, dtype=np.float32)
    X_neg = X_neg.reshape((X_neg.shape[0], -1))

    X = np.concatenate((X_pos, X_neg))
    y = np.concatenate((np.ones(X_pos.shape[0]), np.zeros(X_neg.shape[0])))
    y = y.reshape((X.shape[0], 1))

    return X, y


def load_characteristics(file_name):
    """
    Load the calculated scores
    return: data and label of input score
    """
    X, Y = None, None
    
    data = np.load(file_name)
    
    if X is None:
        X = data[:, :-1]
    else:
        X = np.concatenate((X, data[:, :-1]), axis=1)
    if Y is None:
        Y = data[:, -1] # labels only need to load once
         
    return X, Y


def block_split(X, Y, data):
    """
    Split the data training and testing
    :return: X (data) and Y (label) for training / testing
    """
    num_samples = X.shape[0]
    if data == 'cifar10':
        partition = 10000
    elif data == 'imagenet':
        partition = 50000
    
    X_norm, Y_norm = X[:partition], Y[:partition]
    X_adv, Y_adv = X[partition: :], Y[partition: :]
    num_train = 1000

    X_train = np.concatenate((X_norm[:num_train], X_adv[:num_train]))
    Y_train = np.concatenate((Y_norm[:num_train], Y_adv[:num_train]))

    X_test = np.concatenate((X_norm[num_train:], X_adv[num_train:]))
    Y_test = np.concatenate((Y_norm[num_train:], Y_adv[num_train:]))

    return X_train, Y_train, X_test, Y_test










