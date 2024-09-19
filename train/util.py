'''Modified from https://github.com/alinlab/LfF/blob/master/util.py'''

import torch

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


class EMA:
    def __init__(self, label, num_classes=None, alpha=0.9, device=None):
        self.device = device
        self.label = label.to(self.device)
        self.alpha = alpha
        self.parameter = torch.zeros(label.size(0)).to(self.device)
        self.updated = torch.zeros(label.size(0)).to(self.device)
        self.num_classes = num_classes
        self.max = torch.zeros(self.num_classes).to(self.device)

    def update(self, data, index, curve=None, iter_range=None, step=None):
        if curve is None:
            self.parameter[index] = self.alpha * self.parameter[index] + (1 - self.alpha * self.updated[index]) * data
        else:
            alpha = curve ** -(step / iter_range)
            self.parameter[index] = alpha * self.parameter[index] + (1 - alpha * self.updated[index]) * data

        self.updated[index] = 1

    def max_loss(self, label):
        label_device = torch.tensor(label, dtype=torch.int).to(self.device)
        label_index = torch.where(self.label == label_device)[0]
        return self.parameter[label_index].max()


class Hook:
    def __init__(self, module, backward=False):
        self.feature = []
        if backward==False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output
        self.feature.append(output)

    def close(self):
        self.hook.remove()