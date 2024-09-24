from torchvision.transforms import Normalize, Compose, Resize, ToTensor, InterpolationMode


def convert_to_rgb(image):
    return image.convert("RGB")

def get_transform(dataset: str, image_size: int=224):
    if dataset == 'bffhq':
        return Compose([
            convert_to_rgb,
            Resize((image_size, image_size)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif dataset == 'cmnist':
        return Compose([
            convert_to_rgb,
            Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif dataset == 'bar':
        return Compose([
            convert_to_rgb,
            Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif dataset == 'dogs_and_cats':
        return Compose([
            convert_to_rgb,
            Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif dataset == 'cifar10c':
        return Compose([
            convert_to_rgb,
            Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])