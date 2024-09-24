import itertools
import json
import os
import torch
import gc
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from tqdm import tqdm
import copy
from rich.progress import track
import transformers
import logging

logging.getLogger("transformers").setLevel(logging.ERROR)

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

class TagStats():
    def __init__(self, args, class_name: dict) -> None:
        self.args = args
        self.class_name = class_name
        self.device = torch.device(f'cuda:{str(args.gpu_num)}' if torch.cuda.is_available() else 'cpu')
        self.original_class_bias_stats = {}
        self.generated_class_bias_stats = {}
        self.origin2gene = {}
        
        
    def load_model(self):
        model_id = "facebook/bart-large-mnli"
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id,
                                                                        cache_dir=self.args.pretrained+'/bart').to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id,
                                                       cache_dir=self.args.pretrained+'/bart')
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
    
    
    def sort_tags(self, tags: dict[str: int]):
        # Sort tags descending order in frequency.
        sorted_tags = sorted(tags.items(), key=lambda item:item[1], reverse=True)
        return dict(sorted_tags)
    
    
    def sort_tags_(self, tags: dict[str, dict[str, int]]):
        # Sort tags descending order in frequency.
        sorted_tags = sorted(tags.items(), key=lambda item:item[1]['appeared'][0], reverse=True)
        return dict(sorted_tags)
        
        
    def generate_tag_stats(self):
        # For each class and align/conflict, generate tag_stats.json 
        # and integrated tag_stats.json by class.
        for class_idx in self.class_name:
            # Base architecture of tag_stats.json
            tag_stats = {'n_data': 0, 'tags': {}}
            
            # Load tags.json files for {class_idx, bias_type}
            tags_json_path = os.path.join(self.args.root, 
                                          self.args.preproc, 
                                          self.args.dataset, 
                                          self.args.percent, 
                                          'tags', 
                                          f'{class_idx}_tags.json')
            tags_json = load_json(tags_json_path)
            
            # Count tags.
            for image_id in tags_json:
                tag_stats['n_data'] += 1
                for tag in tags_json[image_id]['tags']:
                    if tag in tag_stats['tags']: tag_stats['tags'][tag] += 1
                    else: tag_stats['tags'][tag] = 1
                
            # Sort
            tag_stats['tags'] = self.sort_tags(tag_stats['tags'])
            for tag in tag_stats['tags']:
                tag_stats['tags'][tag] = [tag_stats['tags'][tag], tag_stats['tags'][tag]/tag_stats['n_data']] # [n_tags appeared, appearance ratio]
                
            # Save json.
            save_json_path = os.path.join(self.args.root, 
                                          self.args.preproc, 
                                          self.args.dataset, 
                                          self.args.percent, 
                                          'tags', 
                                          f'{class_idx}_stats.json')
            with open(save_json_path, 'w') as file:
                json.dump(tag_stats, file, indent=4)
                
                
    def integrate_tag_stats(self):
        # Integrate jsons by class.
        self.itg_tag_stats = {}
        for class_idx in self.class_name:
            self.itg_tag_stats[class_idx] = {'n_data': 0, 'tags': {}}
            
            # Load tags.json files
            tag_stats_json_path = os.path.join(self.args.root, 
                                               self.args.preproc, 
                                               self.args.dataset, 
                                               self.args.percent,
                                               'tags',
                                               f'{class_idx}_stats.json')
            tag_stats = load_json(tag_stats_json_path)
            
            # Combine results from align/conflict.
            self.itg_tag_stats[class_idx]['n_data'] = tag_stats['n_data']
            for tag in tag_stats['tags']:
                if tag in self.itg_tag_stats[class_idx]['tags']:
                    self.itg_tag_stats[class_idx]['tags'][tag]['appeared'] += tag_stats['tags'][tag][0]
                else:
                    self.itg_tag_stats[class_idx]['tags'][tag] = {'appeared': tag_stats['tags'][tag][0], 
                                                                  'cond1': None, 
                                                                  'cond2': None,
                                                                  'cond3': None}
            # Appearance ratio
            for tag in self.itg_tag_stats[class_idx]['tags']:
                n_appeared = self.itg_tag_stats[class_idx]['tags'][tag]['appeared']
                self.itg_tag_stats[class_idx]['tags'][tag]['appeared'] = [n_appeared, n_appeared/self.itg_tag_stats[class_idx]['n_data']]

                    
        # Save json.
        save_json_path = os.path.join(self.args.root, 
                                      self.args.preproc, 
                                      self.args.dataset, 
                                      self.args.percent,
                                      'tags',
                                      'tag_stats.json')
        with open(save_json_path, 'w') as file:
            json.dump(self.itg_tag_stats, file, indent=4)
                    
                    
    def condition_bias(self):
        # Bias Attributes Validation
        # Condition 1: Majority within class
        # Condition 2: Minority of classes
        # Condition 3: Non-class
        # Bias attribute(candidates) = Condition1 & Condition2 & Condition3
        
        # Check Condition. 1
        for class_idx in self.class_name:
            for tag in self.itg_tag_stats[class_idx]['tags']:
                appeared_ratio = self.itg_tag_stats[class_idx]['tags'][tag]['appeared'][-1]
                intra_corr_ratio = 1/len(self.class_name)
                condition_1 = 1 if appeared_ratio > intra_corr_ratio else 0
                self.itg_tag_stats[class_idx]['tags'][tag]['cond1'] = condition_1
        
        # Check Condition. 2
        for target_class_idx in self.class_name:
            for tag in self.itg_tag_stats[target_class_idx]['tags']:
                cond2_cnt = 0
                for subject_class_idx in self.class_name:
                    if tag in self.itg_tag_stats[subject_class_idx]['tags']:
                        if self.itg_tag_stats[subject_class_idx]['tags'][tag]['cond1'] == 1:
                            cond2_cnt += 1
                condition_2 = 1 if cond2_cnt <= len(self.class_name)/2 else 0
                self.itg_tag_stats[target_class_idx]['tags'][tag]['cond2'] = condition_2
                
        # Check Condition. 3
        self.load_model()
        for class_idx in self.class_name:
            tags = list(self.itg_tag_stats[class_idx]['tags'].keys())
            tags_sim = {}
            for tag in track(tags, description=f"Tag sim... | class_idx: {class_idx}"): 
                res = self.classifier(f"{self.class_name[class_idx]}", tag, multi_label=True)
                tag_sim = {res['labels'][0]: res['scores'][0]}
                tags_sim.update(tag_sim)
            for tag in track(tags, description=f"Cond3... | class_idx: {class_idx}"): 
                condition_3 = 1 if tags_sim[tag] < self.args.sim_thres else 0
                self.itg_tag_stats[class_idx]['tags'][tag]['cond3'] = condition_3
                self.itg_tag_stats[class_idx]['tags'][tag]['tag_sim'] = tags_sim[tag]

        # Collect n biases
        for class_idx in self.class_name:
            # Sort tags.  
            self.itg_tag_stats[class_idx]['tags'] = self.sort_tags_(self.itg_tag_stats[class_idx]['tags'])
            bias_tags = {}
            for tag in self.itg_tag_stats[class_idx]['tags']:
                if self.itg_tag_stats[class_idx]['tags'][tag]['cond1'] and self.itg_tag_stats[class_idx]['tags'][tag]['cond2'] and self.itg_tag_stats[class_idx]['tags'][tag]['cond3']:
                    bias_tags[tag] = self.itg_tag_stats[class_idx]['tags'][tag]
                if len(bias_tags) >= self.args.n_bias:
                    break
            self.itg_tag_stats[class_idx]['bias_tags'] = bias_tags
        
        # Save json.
        save_json_path = os.path.join(self.args.root, 
                                      self.args.preproc, 
                                      self.args.dataset, 
                                      self.args.percent,
                                      'tags',
                                      'tag_stats.json')
        with open(save_json_path, 'w') as file:
            json.dump(self.itg_tag_stats, file, indent=4)
        
        
    def mix_bias(self):
        biases = set()
        for class_idx in self.class_name:
            for bias_tag in self.itg_tag_stats[class_idx]['bias_tags']:
                biases.add(bias_tag)
        biases = list(biases)
                
        # Do not use biases that have the same meaning as a specific class as bias-conflict attributes.
        bias_class_scores = self.classifier([value for value in self.class_name.values()], biases, multi_label=True)
        confusing_biases = []
        for i in range(len(self.class_name)):
            confusing_biases += [label for score, label in zip(bias_class_scores[i]['scores'], bias_class_scores[i]['labels']) if score >= self.args.sim_thres]
        confusing_biases = set(confusing_biases)
                
        for class_idx in self.class_name:
            tmp_biases = list(biases)
            for bias_tag in self.itg_tag_stats[class_idx]['bias_tags']:
                if bias_tag in tmp_biases:
                    tmp_biases.remove(bias_tag)
            
            tmp_biases = list(set(tmp_biases) - confusing_biases)
            self.itg_tag_stats[class_idx]['bias_conflict_tags'] = tmp_biases
            
        save_json_path = os.path.join(self.args.root, 
                                      self.args.preproc, 
                                      self.args.dataset, 
                                      self.args.percent,
                                      'tags',
                                      'tag_stats.json')
        with open(save_json_path, 'w') as file:
            json.dump(self.itg_tag_stats, file, indent=4)
            
        self.off_model()
            
        print("[Done] Bias candidates: tag_stats.json files have been made.")
    
    
    def detect_biased_samples(self):
        # fill bias_detected as list of bias attrs
        itg_tag_stats_path = os.path.join(self.args.root, 
                                          self.args.preproc, 
                                          self.args.dataset, 
                                          self.args.percent,
                                          'tags',
                                          'tag_stats.json')
        self.itg_tag_stats = load_json(itg_tag_stats_path)
        
        for class_idx in self.class_name:
            # Load tags.json files for {class_idx, bias_type}
            tags_json_path = os.path.join(self.args.root, 
                                          self.args.preproc, 
                                          self.args.dataset, 
                                          self.args.percent, 
                                          'tags', 
                                          f'{class_idx}_tags.json')
            tags_json = load_json(tags_json_path)
            bias_attrs = set(self.itg_tag_stats[class_idx]['bias_tags'].keys())
            
            for sample in tags_json:
                sample_tags = set(tags_json[sample]['tags'].keys())
                bias_tags = list(sample_tags & bias_attrs)
                tags_json[sample]['bias_detected'] = bias_tags
                
            with open(tags_json_path, 'w') as file:
                json.dump(tags_json, file, indent=4)
    
            
    def class_bias_stats(self):
        itg_tag_stats_path = os.path.join(self.args.root, 
                                          self.args.preproc, 
                                          self.args.dataset, 
                                          self.args.percent,
                                          'tags',
                                          'tag_stats.json')
        self.itg_tag_stats = load_json(itg_tag_stats_path)
        
        # For original
        class_biases = [self.itg_tag_stats[class_idx]['bias_tags'] for class_idx in self.class_name]
        class_biases = set([bias for sublist in class_biases for bias in sublist])

        for class_idx in self.class_name:
            self.original_class_bias_stats[class_idx] = {
                bias: [] for bias in class_biases
            }
            self.original_class_bias_stats[class_idx]['none'] = []
            
            for bias_type in ['align', 'conflict']:
                # Original dataset path
                origin_prepath = os.path.join(bias_type, class_idx)
                # Load tags.json files for {class_idx, bias_type}
                tag_json_path = os.path.join(self.args.root, self.args.preproc, self.args.dataset, self.args.percent, bias_type, class_idx, 'jsons', 'tags.json')
                tags = load_json(tag_json_path)
                
                for image_id in tqdm(tags, desc=f"Original: {class_idx}, {bias_type}"):
                    none_flag = True
                    for bias in self.original_class_bias_stats[class_idx]:
                        if bias in tags[image_id]['tags']:
                            self.original_class_bias_stats[class_idx][bias].append(os.path.join(origin_prepath, image_id))
                            none_flag = False
                    if none_flag:
                        self.original_class_bias_stats[class_idx]['none'].append(os.path.join(origin_prepath, image_id))
                        
        # Save json.
        save_json_path = os.path.join(self.args.root, self.args.preproc, self.args.dataset, self.args.percent, 'original_class_bias_stats.json')
        with open(save_json_path, 'w') as file:
            json.dump(self.original_class_bias_stats, file, indent=4)
        
        # For generated
        class_biases = [self.itg_tag_stats[class_idx]['bias_conflict_tags'] for class_idx in self.class_name]
        class_biases = set([bias for sublist in class_biases for bias in sublist])
        
        for class_idx in self.class_name:
            self.generated_class_bias_stats[class_idx] = {
                bias: [] for bias in class_biases
            }
            self.generated_class_bias_stats[class_idx]['none'] = []

            bias = list(self.itg_tag_stats[class_idx]['bias_tags'].keys())[0]
            
            for bias_type in ['align', 'conflict']:
                # Original dataset path
                generated_prepath = os.path.join(bias_type, class_idx, 'imgs')
                # Load tags.json files for {class_idx, bias_type}
                tag_json_path = os.path.join(self.args.root, self.args.preproc, self.args.dataset, self.args.percent, bias_type, class_idx, 'jsons', 'tags.json')
                if os.path.exists(tag_json_path):
                    with open(tag_json_path, 'r') as file:
                        try:
                            tags = json.load(file)
                        except json.JSONDecodeError:
                            raise RuntimeError("An error occurred while loading the existing json file.")
                else:
                    raise RuntimeError(f"tag_stats.json does not exist.\nPath: {tag_json_path}")
                
                for image_id in tqdm(tags, desc=f"Generated: {class_idx}, {bias_type}"):
                    none_flag = True
                    if bias in tags[image_id]['tags']:
                        tmp_class_biases = copy.deepcopy(list(class_biases))
                        if bias in tmp_class_biases:
                            tmp_class_biases.remove(bias)
                        for bias_conflict_attr in tmp_class_biases:
                            self.generated_class_bias_stats[class_idx][bias_conflict_attr].append(os.path.join(generated_prepath, f"Turn-{self.class_name[class_idx]}-into-{self.class_name[class_idx]}-{bias_conflict_attr}_".replace(' ', '-')+image_id))
                    else:
                        self.generated_class_bias_stats[class_idx]['none'].append(os.path.join(generated_prepath, image_id))
        # Save json.
        save_json_path = os.path.join(self.args.root, self.args.preproc, self.args.dataset, self.args.percent, 'generated_class_bias_stats.json')
        with open(save_json_path, 'w') as file:
            json.dump(self.generated_class_bias_stats, file, indent=4)
            
        # origin2gene
        for class_idx in self.class_name:
            bias = list(self.itg_tag_stats[class_idx]['bias_tags'])[0]
            bias_conflict_attrs = self.itg_tag_stats[class_idx]['bias_conflict_tags']
            
            for bias_type in ['align', 'conflict']:
                # Original dataset path
                origin_prepath = os.path.join(bias_type, class_idx)
                
                # Load tags.json files for {class_idx, bias_type}
                tag_json_path = os.path.join(self.args.root, self.args.preproc, self.args.dataset, self.args.percent, bias_type, class_idx, 'jsons', 'tags.json')
                if os.path.exists(tag_json_path):
                    with open(tag_json_path, 'r') as file:
                        try:
                            tags = json.load(file)
                        except json.JSONDecodeError:
                            raise RuntimeError("An error occurred while loading the existing json file.")
                else:
                    raise RuntimeError(f"tag_stats.json does not exist.\nPath: {tag_json_path}")
                
                for image_id in tqdm(tags, desc=f"origin2gene: {class_idx}, {bias_type}"):
                    if bias in tags[image_id]['tags']:
                        tmp_paths = [os.path.join(origin_prepath, 'imgs', f"Turn-{self.class_name[class_idx]}-into-{self.class_name[class_idx]}-{bias_conflict_attr}_".replace(' ', '-')+image_id) for bias_conflict_attr in bias_conflict_attrs]
                        self.origin2gene[os.path.join(origin_prepath, image_id)] = tmp_paths
                    else:
                        self.origin2gene[os.path.join(origin_prepath, image_id)] = False
                    none_flag = True
                    
        # Save json.
        save_json_path = os.path.join(self.args.root, self.args.preproc, self.args.dataset, self.args.percent, 'origin2gene.json')
        with open(save_json_path, 'w') as file:
            json.dump(self.origin2gene, file, indent=4)