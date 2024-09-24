'''
 * The Tag2Text Model
 * Written by Xinyu Huang
 * Edited by Jungwook Seo
'''

import os
import json
import glob
import gc
import itertools

import torch
from tqdm import tqdm
from rich.progress import track

from PIL import Image
from .ram.models import tag2text
from .ram import inference_tag2text as inference
from .ram import get_transform
        
        
class Bias2Tag():
    def __init__(self,
                 args,
                 class_name: dict[str: str],
                 image_size=224):
        self.args = args
        self.pretrained = os.path.join(args.pretrained, 'tag2text', 'tag2text_swin_14m.pth')
        self.image_size = image_size
        self.device = torch.device(f'cuda:{str(args.gpu_num)}' if torch.cuda.is_available() else 'cpu')
        self.tag2text_model = None
        self.class_name = class_name

    def load_model(self):
        self.tag2text_model = tag2text(pretrained=self.pretrained,
                                       image_size=self.image_size,
                                       vit='swin_b')
        self.tag2text_model.thres = self.args.tag2text_thres  # thres for tagging
        self.tag2text_model.eval()
        self.tag2text_model = self.tag2text_model.to(self.device)
        print(f"Tag2Text has been loaded. Device: {self.device}")

    def off_model(self):
        del self.tag2text_model
        torch.cuda.empty_cache()
        gc.collect()
        self.tag2text_model = None

    def generate_tag_json(self):
        # Load tag2text.
        if self.tag2text_model == None: self.load_model()

        # Generate tags info
        transform = get_transform(dataset=self.args.dataset,
                                  image_size=self.image_size)
        
        # For each class generate {class_idx}.json.
        for class_idx in self.class_name:
            save_json_path = os.path.join(self.args.root, 
                                          self.args.preproc,
                                          self.args.dataset, 
                                          self.args.percent,
                                          'tags', 
                                          f'{str(class_idx)}_tags.json')
            if os.path.exists(save_json_path):
                print(f"Tag file exists: {class_idx}")
                continue
            
            image_paths = glob.glob(os.path.join(self.args.root, 'benchmarks', self.args.dataset, self.args.percent, '*', class_idx, '*.*'))

            # Note that we do not use bias attribute during debiasing.
            json_dict = {}
            for image_path in track(image_paths, description=f"tags.json... | class_idx: {class_idx}"):
                bias_idx = image_path.split('/')[-1].split('_')[-1][0]
                image_id = image_path.split('/')[-1] # *.png or *.jpg
                
                # Inference tags and caption.
                image = transform(Image.open(image_path)).unsqueeze(0).to(self.device)
                res = inference(image, self.tag2text_model)

                # Append new data to json_dict.
                # json_dict is {class_idx}.json dictionary.
                json_dict[image_id] = {
                    "class_name": self.class_name[class_idx],
                    "class_idx": class_idx,
                    "bias_idx": bias_idx,
                    "biased": True if class_idx == bias_idx else False,
                    "bias_detected": None, # After computing 3 conditions.
                    "tags": {key: None for key in res[0].split(' | ')}, # not to modify official Tag2Text code.
                    "caption": res[2],
                    "tag2text_thres": self.args.tag2text_thres
                }

            with open(save_json_path, 'w') as file:
                json.dump(json_dict, file, indent=4)
                
        # Load off model.
        self.off_model()
        
        print("[Done] Tag2Text: tags.json files have been made.") 