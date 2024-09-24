import os

def makedir_preprocessed(args):
    data2num_class = {
        'cmnist': 10,
        'bffhq': 2,
        'dogs_and_cats': 2,
        'bar': 6,
        'cifar10c': 10,
        'waterbird': 2
        }

    data2pct = {
        'cmnist': ['0pct', '0.5pct', '1pct', '2pct', '5pct'],
        'bffhq': ['0pct', '0.5pct', '1pct', '2pct', '5pct'],
        'dogs_and_cats': ['0pct', '1pct', '5pct'],
        'bar': ['0pct', '1pct', '5pct'],
        'cifar10c': ['0pct', '0.5pct', '1pct', '2pct', '5pct'],
        'waterbird': ['0pct', '0.5pct', '1pct', '2pct', '5pct']
    }
    
    for pct in data2pct[args.dataset]:
        for class_num in range(data2num_class[args.dataset]):
            os.makedirs(f'{args.root}/{args.preproc}/{args.dataset}/{pct}/{class_num}', exist_ok=True) # edited align images
            os.makedirs(f'{args.root}/{args.preproc}/{args.dataset}/{pct}/tags/', exist_ok=True) # tags
    
    print(f"[Done] makedir: preprocessed dir has been made.")