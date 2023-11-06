####### extract principle components using Power iteration method
# attention: Due to the limited storage of the GPU or CPU, we split "K" into several segments, each segment solves "interval_k" principal components
# For single GPU and multiple GPU, we have different code in our github_PCA.py, please choose one of them and comment out the other one when runing
# The runing command is as follows:
bash github_PCA.sh start_k interval_k K sample_num      # eg: bash github_PCA.sh 0 5 200 50000

####### extract principle components using Average gradient
bash github_Average_gradient.sh K sample_num      # eg: bash github_Average_gradient.sh 1000 50000


####### using low-dimensional gradient to detect OOD samples
# for cifar10:
python github_main.py --batch_size 16 --model resnet18 --data cifar10 --kernel NFK --base_method msp --ood_data SVHN 
# for imagenet:
python github_main.py --batch_size 6 --model resnet50 --data imagenet --kernel NFK --base_method knn --ood_data dtd

