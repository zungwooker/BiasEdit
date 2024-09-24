'''Modified from https://github.com/alinlab/LfF/blob/master/data/util.py'''
'''Modified from https://github.com/kakaoenterprise/BiasEnsemble'''

import os
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms as T
from glob import glob
from PIL import Image
import json
import gc

def load_json(json_path: str):
    if os.path.exists(json_path):
        with open(json_path, 'r') as file:
            try:
                json_file = json.load(file)
            except json.JSONDecodeError:
                raise RuntimeError("An error occurred while loading the existing json file.")
    else:
        raise RuntimeError(f".json does not exist.\nPath: {json_path}")
    
    return json_file


class IdxDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return (idx, *self.dataset[idx])


class ZippedDataset(Dataset):
    def __init__(self, datasets):
        super(ZippedDataset, self).__init__()
        self.dataset_sizes = [len(d) for d in datasets]
        self.datasets = datasets

    def __len__(self):
        return max(self.dataset_sizes)

    def __getitem__(self, idx):
        items = []
        for dataset_idx, dataset_size in enumerate(self.dataset_sizes):
            items.append(self.datasets[dataset_idx][idx % dataset_size])

        item = [torch.stack(tensors, dim=0) for tensors in zip(*items)]

        return item
    

class CMNISTDataset(Dataset):
    def __init__(self,
                 args,
                 root, 
                 split,
                 transform=None,
                 include_generated=False,
                 preproc_root=None):
        super(CMNISTDataset, self).__init__()
        self.args = args
        self.transform = transform
        self.root = root
        self.preproc_root = preproc_root

        if split=='train':
            self.align = glob(os.path.join(root, 'align', '*', '*'))
            self.conflict = glob(os.path.join(root, 'conflict', '*', '*'))
            self.data = self.align + self.conflict
            
            print("Origin align: ", len(self.align))
            print("Origin conflict: ", len(self.conflict))
            
        elif split=='valid':
            self.data = glob(os.path.join(root, split, '*'))
            
        elif split=='test':
            self.data = glob(os.path.join(root, '../test', '*', '*'))
            
        if include_generated: # just append generated samples(not exchanging)
            if split == 'train':
                generated_align = glob(os.path.join(preproc_root, 'align', '*', 'imgs', '*'))
                generated_conflict = glob(os.path.join(preproc_root, 'conflict', '*', 'imgs', '*'))
                generated = generated_align + generated_conflict
                self.data += generated
                
                print("Generated from align: ", len(generated_align))
                print("Generated from conflict: ", len(generated_conflict))
                del generated_align, generated_conflict, generated
                gc.collect()
            else:
                print("Wrong split on include_generated flag. Check your settings.")
                raise KeyError()
            
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        attr = torch.LongTensor([int(self.data[index].split('_')[-2]),int(self.data[index].split('_')[-1].split('.')[0])])
        image = Image.open(self.data[index]).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)
        
        return image, attr, self.data[index]


class bFFHQDataset(Dataset):   
    def __init__(self,
                 args,
                 root, 
                 split,
                 transform=None,
                 include_generated=False,
                 preproc_root=None):
        super(bFFHQDataset, self).__init__()
        self.args = args
        self.transform = transform
        self.root = root
        self.preproc_root = preproc_root

        if split=='train':
            self.align = glob(os.path.join(root, 'align', '*', '*'))
            self.conflict = glob(os.path.join(root, 'conflict', '*', '*'))
            self.data = self.align + self.conflict
            
            print("Origin align: ", len(self.align))
            print("Origin conflict: ", len(self.conflict))

        elif split=='valid':
            self.data = glob(os.path.join(os.path.dirname(root), split, '*'))

        elif split=='test':
            self.data = glob(os.path.join(os.path.dirname(root), split, '*'))
            data_conflict = []
            for path in self.data:
                target_label = path.split('/')[-1].split('.')[0].split('_')[1]
                bias_label = path.split('/')[-1].split('.')[0].split('_')[2]
                if target_label != bias_label:
                    data_conflict.append(path)
            self.data = data_conflict
            
        if include_generated: # just append generated samples(not exchanging)
            if split == 'train':
                generated_align = glob(os.path.join(preproc_root, 'align', '*', 'imgs', '*'))
                generated_conflict = glob(os.path.join(preproc_root, 'conflict', '*', 'imgs', '*'))
                generated = generated_align + generated_conflict
                self.data += generated
                
                print("Generated from align: ", len(generated_align))
                print("Generated from conflict: ", len(generated_conflict))
                del generated_align, generated_conflict, generated
                gc.collect()
            else:
                print("Wrong split on include_generated flag. Check your settings.")
                raise KeyError()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        attr = torch.LongTensor([int(self.data[index].split('_')[-2]),int(self.data[index].split('_')[-1].split('.')[0])])
        image = Image.open(self.data[index]).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)
        
        return image, attr, self.data[index]


# class BARDataset(Dataset):
#     def __init__(self,
#                  root, 
#                  split,
#                  transform=None,
#                  include_generated=False,
#                  preproc_root=None):
#         super(BARDataset, self).__init__()
#         self.transform = transform
#         self.root = root
#         self.preproc_root = preproc_root
        
#         if include_generated:
#             origin2gene_path = os.path.join(preproc_root, 'origin2gene.json')
#             self.origin2gene = load_json(origin2gene_path)

#         if split=='train':
#             self.align = glob(os.path.join(root, 'align', '*', '*'))
#             self.conflict = glob(os.path.join(root, 'conflict', '*', '*'))
#             self.data = self.align + self.conflict
    
#             if include_generated:
#                 self.generated_data = []
#                 for data in self.data:
#                     image_key = data.replace(root+'/', '')
#                     if not self.origin2gene[image_key]: continue
#                     tmp_gene_data = os.path.join(preproc_root, random.choice(self.origin2gene[image_key]))
#                     self.generated_data.append(tmp_gene_data)
#                 self.data += self.generated_data
            
#                 del self.origin2gene
#                 gc.collect()
                
#         elif split=='valid':
#             self.data = glob(os.path.join(root, '../valid', '*', '*'))
#         elif split=='test':
#             self.data = glob(os.path.join(root, '../test', '*', '*'))

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, index):
#         attr = torch.LongTensor(
#             [int(self.data[index].split('_')[-2]), int(self.data[index].split('_')[-1].split('.')[0])])
#         image = Image.open(self.data[index]).convert('RGB')
#         image_path = self.data[index]

#         if 'conflict' in image_path:
#             attr[1] = (attr[0] + 1) % 6
#         elif 'align' in image_path:
#             attr[1] = attr[0]

#         if self.transform is not None:
#             image = self.transform(image)
        
#         return image, attr, self.data[index]


# class DogCatDataset(Dataset):
#     def __init__(self,
#                  root, 
#                  split,
#                  transform=None,
#                  include_generated=False,
#                  preproc_root=None):
#         super(DogCatDataset, self).__init__()
#         self.transform = transform
#         self.root = root
#         self.preproc_root = preproc_root
        
#         if include_generated:
#             origin2gene_path = os.path.join(preproc_root, 'origin2gene.json')
#             self.origin2gene = load_json(origin2gene_path)

#         if split=='train':
#             self.align = glob(os.path.join(root, 'align', '*', '*'))
#             self.conflict = glob(os.path.join(root, 'conflict', '*', '*'))
#             self.data = self.align + self.conflict
    
#             if include_generated:
#                 self.generated_data = []
#                 for data in self.data:
#                     image_key = data.replace(root+'/', '')
#                     if not self.origin2gene[image_key]: continue
#                     tmp_gene_data = os.path.join(preproc_root, random.choice(self.origin2gene[image_key]))
#                     self.generated_data.append(tmp_gene_data)
#                 self.data += self.generated_data
            
#                 del self.origin2gene
#                 gc.collect()
                
#         elif split == "valid":
#             self.data = glob(os.path.join(root, split, '*'))
#         elif split == "test":
#             self.data = glob(os.path.join(root, "../test", '*', '*'))
        
#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, index):
#         attr = torch.LongTensor([int(self.data[index].split('_')[-2]),int(self.data[index].split('_')[-1].split('.')[0])])
#         image = Image.open(self.data[index]).convert('RGB')

#         if self.transform is not None:
#             image = self.transform(image)

#         return image, attr, self.data[index]


# class CIFAR10CDataset(Dataset):
#     def __init__(self,
#                  root, 
#                  split,
#                  transform=None,
#                  include_generated=False,
#                  preproc_root=None):
#         super(CIFAR10CDataset, self).__init__()
#         self.transform = transform
#         self.root = root
#         self.preproc_root = preproc_root
        
#         if include_generated:
#             origin2gene_path = os.path.join(preproc_root, 'origin2gene.json')
#             self.origin2gene = load_json(origin2gene_path)

#         if split=='train':
#             self.align = glob(os.path.join(root, 'align', '*', '*'))
#             self.conflict = glob(os.path.join(root, 'conflict', '*', '*'))
#             self.data = self.align + self.conflict
    
#             if include_generated:
#                 self.generated_data = []
#                 for data in self.data:
#                     image_key = data.replace(root+'/', '')
#                     if not self.origin2gene[image_key]: continue
#                     tmp_gene_data = os.path.join(preproc_root, random.choice(self.origin2gene[image_key]))
#                     self.generated_data.append(tmp_gene_data)
#                 self.data += self.generated_data
            
#                 del self.origin2gene
#                 gc.collect()
                
#         elif split=='valid':
#             self.data = glob(os.path.join(root, split, '*', '*'))
#         elif split=='test':
#             self.data = glob(os.path.join(root, '../test', '*', '*'))

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, index):
#         attr = torch.LongTensor([int(self.data[index].split('_')[-2]),int(self.data[index].split('_')[-1].split('.')[0])])
#         image = Image.open(self.data[index]).convert('RGB')

#         if self.transform is not None:
#             image = self.transform(image)
        
#         return image, attr, self.data[index]


transforms = {
    "cmnist": {
        "train": T.Compose([T.ToTensor()]),
        "valid": T.Compose([T.ToTensor()]),
        "test": T.Compose([T.ToTensor()])
        },
    "bar": {
        "train": T.Compose([T.Resize((224, 224)), T.ToTensor()]),
        "valid": T.Compose([T.Resize((224, 224)), T.ToTensor()]),
        "test": T.Compose([T.Resize((224, 224)), T.ToTensor()])
        },
    "bffhq": {
        "train": T.Compose([T.Resize((224, 224)), T.ToTensor()]),
        "valid": T.Compose([T.Resize((224, 224)), T.ToTensor()]),
        "test": T.Compose([T.Resize((224, 224)), T.ToTensor()])
        },
    "dogs_and_cats": {
        "train": T.Compose([T.Resize((224, 224)), T.ToTensor()]),
        "valid": T.Compose([T.Resize((224, 224)), T.ToTensor()]),
        "test": T.Compose([T.Resize((224, 224)), T.ToTensor()])
        },
    "cifar10c": {
        "train": T.Compose([T.ToTensor(),]),
        "valid": T.Compose([T.ToTensor(),]),
        "test": T.Compose([T.ToTensor(),])
        },
    }


transforms_preprcs = {
    "cmnist": {
        "train": T.Compose([T.ToTensor()]),
        "valid": T.Compose([T.ToTensor()]),
        "test": T.Compose([T.ToTensor()])
        },
    "bar": {
        "train": T.Compose([
            T.Resize((224, 224)),
            T.RandomCrop(224, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ]),
        "valid": T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]),
        "test": T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        },
    "bffhq": {
        "train": T.Compose([
            T.Resize((224,224)),
            T.RandomCrop(224, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]),
        "valid": T.Compose([
            T.Resize((224,224)),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]),
        "test": T.Compose([
            T.Resize((224,224)),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        },
    "dogs_and_cats": {
        "train": T.Compose([
            T.Resize((224, 224)),
            T.RandomCrop(224, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]),
        "valid": T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]),
        "test": T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]),
        },
    "cifar10c": {
        "train": T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]),
        "valid": T.Compose([
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]),
        "test": T.Compose([
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]),
        },
    }


def get_dataset(args,
                dataset_split, 
                transform_split,
                include_generated,
                use_preprocess=None):

    if use_preprocess:
        transform = transforms_preprcs[args.dataset][transform_split]
    else:
        transform = transforms[args.dataset][transform_split]

    dataset_split = "valid" if (dataset_split == "eval") else dataset_split
    
    if args.dataset == 'cmnist':
        root = args.data_dir + f"/cmnist/{args.pct}"
        preproc_root = args.preproc_dir + f"/cmnist/{args.pct}"
        dataset = CMNISTDataset(args=args,
                                root=root, 
                                split=dataset_split,
                                transform=transform,
                                include_generated=include_generated,
                                preproc_root=preproc_root)
        
    elif args.dataset == "bffhq":
        root = args.data_dir + f"/bffhq/{args.pct}"
        preproc_root = args.preproc_dir + f"/bffhq/{args.pct}"
        dataset = bFFHQDataset(args=args,
                               root=root, 
                               split=dataset_split,
                               transform=transform,
                               include_generated=include_generated,
                               preproc_root=preproc_root)
        
    # elif args.dataset == "bar":
    #     root = args.data_dir + f"/bar/{args.pct}"
    #     preproc_root = args.preproc_dir + f"/bar/{args.pct}"
    #     dataset = BARDataset(root=root, 
    #                          split=dataset_split,
    #                          transform=transform,
    #                          include_generated=include_generated,
    #                          preproc_root=preproc_root)
        
    # elif args.dataset == "dogs_and_cats":
    #     root = args.data_dir + f"/dogs_and_cats/{args.pct}"
    #     preproc_root = args.preproc_dir + f"/dogs_and_cats/{args.pct}"
    #     dataset = DogCatDataset(root=root, 
    #                             split=dataset_split,
    #                             transform=transform,
    #                             include_generated=include_generated,
    #                             preproc_root=preproc_root)
        
    # elif args.dataset == "cifar10c":
    #     root = args.data_dir + f"/cifar10c/{args.pct}"
    #     preproc_root = args.preproc_dir + f"/cifar10c/{args.pct}"
    #     dataset = CIFAR10CDataset(root=root, 
    #                               split=dataset_split,
    #                               transform=transform,
    #                               include_generated=include_generated,
    #                               preproc_root=preproc_root)
    else:
        print('Wrong dataset ...')
        print('Select one of them: cmnist, bffhq')
        print("bar, dogs_and_cats, cifar10c is not implemented!")
        import sys
        sys.exit(0)

    return dataset