from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import torch
import gc
import os
import json
import itertools
from tqdm import tqdm


class BartLargeMnli():
    def __init__(self,
                 args,
                 gpu_num: int,
                 dataset: str,
                 class_name: dict[str: str],
                 conflict_ratio: str,
                 root_path: str,
                 pretrained_path: str,
                 sim_thres: float,):
        self.args = args
        self.dataset = dataset
        self.class_name = class_name
        self.conflict_ratio = conflict_ratio
        self.sim_thres = sim_thres
        self.model = None
        self.tokenizer = None
        self.classifier = None
        self.device = torch.device(f'cuda:{str(gpu_num)}' if torch.cuda.is_available() else 'cpu')
        self.root_path = root_path
        self.pretrained_path = os.path.join(pretrained_path, 'bart') #pretrained_path + '/bart'
        
    def load_model(self):
        model_id = "facebook/bart-large-mnli"
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id,
                                                                        cache_dir=self.pretrained_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id,
                                                       cache_dir=self.pretrained_path)
        self.classifier = pipeline("zero-shot-classification", 
                                   model=self.model, 
                                   tokenizer=self.tokenizer, 
                                   device=self.device)
        print(f"Bart-large-mnli classifier has been loaded. Device: {self.device}")
        
    def off_model(self):
        del self.model
        del self.tokenizer
        del self.classifier
        torch.cuda.empty_cache()
        gc.collect()
        self.model = None
        self.tokenizer = None
        self.classifier = None
        
    def compute_sim(self, caption: str, tags: dict[str: None], multi_label=True):
        res = self.classifier(caption, list(tags.keys()), multi_label=multi_label)
        tags_scores = {tag: score for tag, score in zip(res['labels'], res['scores'])}
        
        # Filtered tags has less similiarity as sim_thres 
        filtered_tags_scores = {tag: score for tag, score in zip(res['labels'], res['scores']) if score <= self.sim_thres}
        
        # Label tags has more or equal similiarity as sim_thres; same meaning as user label.
        label_tags_scores = {tag: score for tag, score in zip(res['labels'], res['scores']) if score > self.sim_thres}
        
        return tags_scores, filtered_tags_scores, label_tags_scores
    
    def generate_filtered_json(self):       
        if self.classifier == None: self.load_model()
        
        # For each class and align/conflict, generate filtered.json.
        for label, bias in itertools.product(self.class_name, ['align', 'conflict']):
            # Load tag2text.json.
            tag2text_json_path = os.path.join(self.root_path, self.args.preproc, self.dataset, self.conflict_ratio+'pct', bias, label, 'jsons', 'tag2text.json')
            if os.path.exists(tag2text_json_path):
                with open(tag2text_json_path, 'r') as file:
                    try:
                        tag2text_json = json.load(file)
                    except json.JSONDecodeError:
                        raise RuntimeError("An error occurred while loading the existing json file.")
            else:
                raise RuntimeError(f"tag2text.json does not exist.\nPath: {tag2text_json_path}")
            
            # Compute similiarity between class_name and tags.
            # Assign filtered tags and lable tags.
            for image_id in tqdm(tag2text_json, desc=f"filtered.json... | label: {label}, bias: {bias}"):
                tags_scores, filtered_tags_scores, label_tags_scores = self.compute_sim(caption=f"A photo of {tag2text_json[image_id]['class_name']}",
                                                                                        tags=tag2text_json[image_id]['tags'])
                tag2text_json[image_id]['tags'] = tags_scores
                tag2text_json[image_id]['filtered_tags'] = filtered_tags_scores
                tag2text_json[image_id]['label_tags'] = label_tags_scores
                tag2text_json[image_id]['sim_thres'] = self.sim_thres
                
            # Save /{class}/{bias}/tag2text.json
            save_json_path = os.path.join(self.root_path, self.args.preproc, self.dataset, self.conflict_ratio+'pct', bias, label, 'jsons', 'filtered.json')
            with open(save_json_path, 'w') as file:
                json.dump(tag2text_json, file, indent=4)
            
        # Load off model.
        self.off_model()
        
        print("[Done] Bart: filtered.json files have been made.")