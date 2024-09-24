import os
import torch
import json
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from PIL import Image
import itertools
import gc
from rich.progress import track
import glob
import shutil

class IP2P:
    def __init__(self,
                 args,
                 class_name: dict[str: str]):
        self.args = args
        self.class_name = class_name
        self.pretrained_path = os.path.join(args.pretrained_path, 'InstructPix2Pix')
        self.device = torch.device(f'cuda:{str(args.gpu_num)}' if torch.cuda.is_available() else 'cpu')
        self.pipe = None
        self.generator = torch.Generator(self.device).manual_seed(args.random_seed)
        
        
        itg_tag_stats_path = os.path.join(self.args.root_path, 
                                          self.args.preproc,
                                          self.args.dataset,
                                          self.args.conflict_ratio+'pct',
                                          'tag_stats.json')
        if os.path.exists(itg_tag_stats_path):
            with open(itg_tag_stats_path, 'r') as file:
                try:
                    itg_tag_stats = json.load(file)
                except json.JSONDecodeError:
                    raise RuntimeError("An error occurred while loading the existing json file.")
        else:
            raise RuntimeError(f"tag_stats.json does not exist.\nPath: {itg_tag_stats_path}")
        self.itg_tag_stats = itg_tag_stats
        
    def load_model(self):
        model_id = "timbrooks/instruct-pix2pix"
        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id,
                                                                           torch_dtype=torch.float16, 
                                                                           safety_checker=None,
                                                                           cache_dir=self.pretrained_path)
        self.pipe.to(self.device)
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.set_progress_bar_config(disable=True)
        
    def off_model(self):
        del self.pipe
        torch.cuda.empty_cache()
        gc.collect()
        self.pipe = None
        
    def resize_image(self, inout: str, image_path: str=None, image: Image.Image=None):
        if inout == 'in':
            image = Image.open(image_path)
            resized_image = image.resize((self.args.resize_in, self.args.resize_in), Image.BICUBIC)
            return resized_image
        elif inout == 'out':
            if self.args.dataset == 'cmnist':
                resized_image = image.resize((28, 28), Image.BICUBIC)
            elif self.args.dataset == 'bffhq':
                resized_image = image.resize((224, 224), Image.BICUBIC)
            elif self.args.dataset == 'bar':
                resized_image = image.resize((224, 224), Image.BICUBIC)
            elif self.args.dataset == 'dogs_and_cats':
                resized_image = image.resize((224, 224), Image.BICUBIC)
            elif self.args.dataset == 'cifar10c':
                resized_image = image.resize((32, 32), Image.BICUBIC)
            return resized_image
    
    def return_insts(self, class_name: str, bias_conflict_tags: list[str]):
        return [f"Turn {class_name} into {class_name} {tag}" for tag in bias_conflict_tags]
    
    def edit_images(self):
        if self.pipe == None: self.load_model()

        iter_class = self.args.edit_class_idx.split(',') if self.args.edit_class_idx else self.class_name
        for class_idx, bias_type in itertools.product(iter_class, ['align', 'conflict']):
            class_biases = self.itg_tag_stats[class_idx]['bias_tags']
            
            # Load tags.json
            tags_json_path = os.path.join(self.args.root_path, 
                                          self.args.preproc, 
                                          self.args.dataset, 
                                          self.args.conflict_ratio+'pct', 
                                          bias_type, class_idx, 'jsons', 
                                          'tags.json')
            if os.path.exists(tags_json_path):
                with open(tags_json_path, 'r') as file:
                    try:
                        tags_json = json.load(file)
                    except json.JSONDecodeError:
                        raise RuntimeError("An error occurred while loading the existing json file.")
            else:
                raise RuntimeError(f"tags.json does not exist.\nPath: {tags_json_path}")

            # Start Editing
            for image_id in track(tags_json, description=f"Editing... | class_idx: {class_idx}, bias: {bias_type}"):
                if not set(class_biases) & set(tags_json[image_id]['tags']):
                    # This sample is not biased; no need to edit it.
                    continue
                
                origin_image_path = os.path.join(self.args.root_path, 
                                                 'benchmarks', 
                                                 self.args.dataset, 
                                                 self.args.conflict_ratio+'pct', 
                                                 bias_type, class_idx, image_id)
                input_image = self.resize_image(inout='in', image_path=origin_image_path)
                input_insts = self.return_insts(class_name=self.class_name[class_idx],
                                                bias_conflict_tags=self.itg_tag_stats[class_idx]['bias_conflict_tags'])
                for inst in input_insts:
                    save_path = os.path.join(self.args.root_path,
                                             self.args.preproc, 
                                             self.args.dataset, 
                                             self.args.conflict_ratio+'pct', 
                                             bias_type, class_idx, 'imgs', 
                                             f"{inst.replace(' ', '-')}_{image_id}")
                    # If the image exists, continue
                    if os.path.exists(save_path):
                        print(f"File exists: {save_path}")
                        continue
                    
                    # If the image was generated before, just copy it.
                    prev_generated_img_path = os.path.join(self.args.root_path,
                                                           self.args.preproc, 
                                                           self.args.dataset, 
                                                           "*", # pct
                                                           bias_type, class_idx, 'imgs', 
                                                           f"{inst.replace(' ', '-')}_{image_id}")
                    prev_generated_images = glob.glob(prev_generated_img_path)
                    if len(prev_generated_images) > 0:
                        shutil.copy(prev_generated_images[0], save_path)
                        print(f"Copied: {save_path}")
                    else:
                        # Or not, generate it.
                        images = self.pipe(inst, image=input_image, 
                                           num_inference_steps=10, 
                                           image_guidance_scale=self.args.image_guidance_scale,
                                           generator=self.generator).images
                        out_image = self.resize_image(inout='out', image=images[0])
                        out_image.save(save_path, format='PNG')
                    
            print(f"[WIP] IP2P: Images have been edited. | class index: {class_idx} bias type: {bias_type}")
            
        self.off_model()
        if not self.args.edit_class_idx:
            print(f"[Done] IP2P: Images have been edited. | Whole dataset")
        else:
            print(f"[Done] IP2P: Images have been edited. | Class index: {self.args.edit_class_idx}")