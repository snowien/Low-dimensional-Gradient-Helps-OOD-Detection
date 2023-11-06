# Low-dimensional-Gradient-Helps-OOD-Detection
This is the code for our paper: <https://arxiv.org/abs/2310.17163>
# Usage
## 1. Dataset preparation
Refer to <https://github.com/deeplearning-wisc/knn-ood>
## 2. Pre-trained model
please download here: <https://jbox.sjtu.edu.cn/v/list/self/1721407219938893851>
## 3. Code run
We have two steps when running our method: 
### 1. extract principal components using PCA or Average Gradient:
```bash github_PCA.sh start_k interval_k K sample_num  # eg: bash github_PCA.sh 0 5 200 50000```

```bash github_Average_gradient.sh K sample_num      # eg: bash github_Average_gradient.sh 1000 50000```
### 2. project gradients into the extracted low-dimensional subspace and employ them to detect OOD samples
```python github_main.py --batch_size 16 --model resnet18 --data cifar10 --kernel NFK --base_method msp --ood_data SVHN # for cifar10```

```python github_main.py --batch_size 6 --model resnet50 --data imagenet --kernel NFK --base_method knn --ood_data dtd # for imagenet```
# Citation
If you use our codebase, please cite our work:
```
@article{wu2023low,
  title={Low-Dimensional Gradient Helps Out-of-Distribution Detection},
  author={Wu, Yingwen and Li, Tao and Cheng, Xinwen and Yang, Jie and Huang, Xiaolin},
  journal={arXiv preprint arXiv:2310.17163},
  year={2023}
}
```
My email address is yingwen_wu@sjtu.edu.cn
