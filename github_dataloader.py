from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
import torch
import torchvision
from torchvision import transforms
from easydict import EasyDict
from ylib.dataloader.tinyimages_80mn_loader import TinyImages
from ylib.dataloader.imagenet_loader import ImageNet
from ylib.dataloader.svhn_loader import SVHN

from ylib.dataloader.random_data import GaussianRandom, LowFreqRandom

imagesize = 32

transform_test = transforms.Compose([
    transforms.Resize((imagesize, imagesize)),
    transforms.CenterCrop(imagesize),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616)),
])

transform_train = transforms.Compose([
    transforms.Resize((imagesize, imagesize)),
    transforms.CenterCrop(imagesize),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616)),
])


transform_train_largescale = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

transform_test_largescale = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

kwargs = {'num_workers': 2, 'pin_memory': True}
num_classes_dict = {'CIFAR-100': 100, 'CIFAR-10': 10, 'imagenet': 1000}


class IN_DATA:
    def __init__(self, data_type, batch_size, threads, num_gpus):
        # mean, std = self._get_statistics()
        # print(mean, std)
        if data_type == 'cifar10':
            self.train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
            self.test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        elif data_type == 'imagenet':
            self.train_set = torchvision.datasets.ImageFolder('/home/datasets/ILSVRC2012/train',transform_train_largescale)
            self.test_set = torchvision.datasets.ImageFolder('/home/datasets/ILSVRC2012/val', transform_test_largescale)
        
        if num_gpus > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_set) 
            test_sampler = torch.utils.data.distributed.DistributedSampler(self.test_set) 
            self.train = torch.utils.data.DataLoader(self.train_set, batch_size=batch_size, num_workers=threads, sampler=train_sampler)
            self.test = torch.utils.data.DataLoader(self.test_set, batch_size=batch_size, num_workers=threads, sampler=test_sampler)
        else:
            self.train = torch.utils.data.DataLoader(self.train_set, batch_size=batch_size, shuffle=True, num_workers=threads)
            self.test = torch.utils.data.DataLoader(self.test_set, batch_size=batch_size, shuffle=False, num_workers=threads)
           


def get_loader_out(args, dataset=('tim', 'noise'), config_type='default', split=('train', 'val')):

    config = EasyDict({
        "default": {
            'transform_train': transform_train,
            'transform_test': transform_test,
            'transform_test_largescale': transform_test_largescale,
            'transform_train_largescale': transform_train_largescale,
            'batch_size': args.batch_size
        },
    })[config_type]
    train_ood_loader, val_ood_loader = None, None

    if 'train' in split:
        if dataset[0].lower() == 'imagenet':
            train_ood_loader = torch.utils.data.DataLoader(
                ImageNet(transform=config.transform_train),
                batch_size=config.batch_size, shuffle=True, **kwargs)
        elif dataset[0].lower() == 'tim':
            train_ood_loader = torch.utils.data.DataLoader(
                TinyImages(transform=config.transform_train),
                batch_size=config.batch_size, shuffle=True, **kwargs)

    if 'val' in split:
        val_dataset = dataset[1]
        batch_size = args.batch_size
        imagesize = 224 if args.data in {'imagenet'} else 32
        if val_dataset == 'SVHN':
            val_ood_loader = torch.utils.data.DataLoader(SVHN('./data/ood_data/svhn/', split='test', transform=transform_test, download=False),
                                                       batch_size=batch_size, shuffle=False,
                                                        num_workers=2)
        elif val_dataset == 'dtd': # Texture
            transform = config.transform_test_largescale if args.data in {'imagenet'} else config.transform_test
            val_ood_loader = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(root="./data/ood_data/dtd/images", transform=transform),
                                                       batch_size=batch_size, shuffle=False, num_workers=2)
        elif val_dataset == 'places365':
            dataset_temp = torchvision.datasets.ImageFolder(root="./data/ood_data/places365", transform=transform_test)
            val_ood_loader = torch.utils.data.DataLoader(dataset_temp,batch_size=batch_size, shuffle=False, num_workers=2)
            print('dataset_temp:', len(dataset_temp))
        elif val_dataset == 'CIFAR-100':
            val_ood_loader = torch.utils.data.DataLoader(torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test),
                                                       batch_size=batch_size, shuffle=True, num_workers=2)
        elif val_dataset == 'CIFAR-10':
            val_ood_loader = torch.utils.data.DataLoader(torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test),
                batch_size=batch_size, shuffle=True, num_workers=2)
            
        elif val_dataset == 'places50':
            dataset_temp = torchvision.datasets.ImageFolder("./data/ood_data/imagenet/Places", transform=config.transform_test_largescale)
            val_ood_loader = torch.utils.data.DataLoader(dataset_temp, batch_size=batch_size, shuffle=False, num_workers=2)
            print('dataset_temp:', len(dataset_temp))
        elif val_dataset == 'sun50':
            dataset_temp = torchvision.datasets.ImageFolder("./data/ood_data/imagenet/SUN", transform=config.transform_test_largescale)
            val_ood_loader = torch.utils.data.DataLoader(dataset_temp, batch_size=batch_size, shuffle=False, num_workers=2)
            print('dataset_temp:', len(dataset_temp))
        elif val_dataset == 'inat':
            dataset_temp = torchvision.datasets.ImageFolder("./data/ood_data/imagenet/iNaturalist", transform=config.transform_test_largescale)
            val_ood_loader = torch.utils.data.DataLoader(dataset_temp, batch_size=batch_size,shuffle=False,num_workers=2)
            print('dataset_temp:', len(dataset_temp))
        elif val_dataset == 'tim':
            dataset_temp = TinyImages(transform=transform_test)
            val_ood_loader = torch.utils.data.DataLoader(dataset_temp, batch_size=batch_size, shuffle=False, num_workers=2)
            print('dataset_temp:', len(dataset_temp))
        elif val_dataset == 'imagenet':
            val_ood_loader = torch.utils.data.DataLoader(
                torchvision.datasets.ImageFolder(os.path.join('dataset/imagenet', 'val'), config.transform_test_largescale),
                batch_size=config.batch_size, shuffle=False, **kwargs)
        elif val_dataset == 'noise':
            val_ood_loader = torch.utils.data.DataLoader(
                GaussianRandom(image_size=imagesize, data_size=10000),
                batch_size=batch_size, shuffle=False, num_workers=2)
            # val_ood_loader = torch.utils.data.DataLoader(
            #     GaussianRandom(image_size=imagesize, data_size=10000, transform=config.transform_test_largescale),
            #     batch_size=batch_size, shuffle=False, num_workers=2)
        elif val_dataset == 'lfnoise':
            val_ood_loader = torch.utils.data.DataLoader(
                LowFreqRandom(image_size=imagesize, data_size=10000),
                batch_size=batch_size, shuffle=False, num_workers=2)
        else:
            dataset_temp = torchvision.datasets.ImageFolder("./data/ood_data/{}".format(val_dataset), transform=transform_test)
            val_ood_loader = torch.utils.data.DataLoader(dataset_temp, batch_size=batch_size, shuffle=False, num_workers=2)
            print('dataset_temp:', len(dataset_temp))
            
    return EasyDict({
        "train_ood_loader": train_ood_loader,
        "val_ood_loader": val_ood_loader,
    })