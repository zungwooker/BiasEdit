from collections import Counter
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb

import os

from data.util import get_dataset, IdxDataset
from module.loss import GeneralizedCELoss
from module.util import get_backbone
from util import *

import warnings
warnings.filterwarnings(action='ignore')
import copy


class LearnerBase(object):
    def __init__(self, args):
        self.args = args
        print("\n==============================================================")
        for arg, value in vars(args).items():
            print(f"{arg}: {value}")
        print("==============================================================\n")
        
        self.device = torch.device(''.join(['cuda:', args.gpu_num]) if args.device == 'cuda' else 'cpu')

        # Experiment settings for each dataset.
        self.data2model = {'cmnist': "MLP",
                           'bffhq': "ResNet18",
                           'dogs_and_cats': "ResNet18",
                           'bar': "ResNet18",
                           'cifar10c': "ResNet18",
                           }

        self.data2batch_size = {'cmnist': 256,
                                'bffhq': 64,
                                'dogs_and_cats': 64,
                                'bar': 64,
                                'cifar10c': 256,
                                }
        
        self.data2preprocess = {'cmnist': None,
                                'bffhq': True,
                                'dogs_and_cats':True,
                                'bar': True,
                                'cifar10c': True,
                                }
        
        self.data2lr = {'cmnist': 0.01,
                        'bffhq': 0.0001,
                        'dogs_and_cats': 0.0001,
                        'bar': 0.00001,
                        'cifar10c': 0.001,
                        }
        
        self.data2pretrained = {'cmnist': False,
                                'bffhq': False,
                                'dogs_and_cats': False,
                                'bar': True,
                                'cifar10c': False,
                                }

        self.model = self.data2model[self.args.dataset]
        self.batch_size = self.data2batch_size[self.args.dataset]
        self.preprocess = self.data2preprocess[self.args.dataset]
        self.lr = self.data2lr[self.args.dataset]
        self.pretrained = self.data2pretrained[self.args.dataset]
        
        
        # Dataset configuration
        self.train_dataset = get_dataset(
            args=args,
            dataset_split="train",
            transform_split="train",
            use_preprocess=self.preprocess,
            include_generated = False # Origin only
        )
        
        self.valid_dataset = get_dataset(
            args=args,
            dataset_split="valid",
            transform_split="valid",
            use_preprocess=self.preprocess,
            include_generated = False
        )
        
        self.test_dataset = get_dataset(
            args=args,
            dataset_split="test",
            transform_split="test",
            use_preprocess=self.preprocess,
            include_generated = False
        )
        
        train_target_attr = []
        for data in self.train_dataset.data:
            train_target_attr.append(int(data.split('_')[-2]))
        train_target_attr = torch.LongTensor(train_target_attr)

        attr_dims = []
        attr_dims.append(torch.max(train_target_attr).item() + 1)
        self.num_classes = attr_dims[0]

        self.train_dataset = IdxDataset(self.train_dataset)

        # Make loader
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            drop_last=True
        )

        self.pretrain_loader = DataLoader( # The loader that BCDs use for voting.
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False
        )

        self.valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )

        # Define model and optimizer
        self.model_b = get_backbone(self.model, self.num_classes, args=self.args, pretrained=self.pretrained).to(self.device)
        self.model_d = get_backbone(self.model, self.num_classes, args=self.args, pretrained=self.pretrained).to(self.device)

        self.optimizer_b = torch.optim.Adam(self.model_b.parameters(), lr=self.lr)
        self.optimizer_d = torch.optim.Adam(self.model_d.parameters(), lr=self.lr)

        # Define loss
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        print(f'self.criterion: {self.criterion}')

        self.bias_criterion = GeneralizedCELoss(q=args.q)
        print(f'self.bias_criterion: {self.bias_criterion}')
        
        self.sample_loss_ema_b = EMA(torch.LongTensor(train_target_attr), num_classes=self.num_classes, alpha=args.ema_alpha, device=self.device)
        self.sample_loss_ema_d = EMA(torch.LongTensor(train_target_attr), num_classes=self.num_classes, alpha=args.ema_alpha, device=self.device)
        print(f'alpha : {self.sample_loss_ema_d.alpha}')
        
        self.best_valid_acc_b, self.best_test_acc_b = 0., 0.
        self.best_valid_acc_d, self.best_test_acc_d = 0., 0.
        
        # Logging
        self.projcode = args.projcode
        self.run_name = args.run_name
        
        biasedit_dir = os.path.dirname(os.path.abspath(__file__))
        self.result_dir = os.path.join(biasedit_dir, 'log', args.dataset, self.projcode, self.run_name)
        os.makedirs(self.result_dir, exist_ok=True)
                    
        print('finished model initialization...')
    
    
    def wandb_switch(self, switch):
        if self.args.wandb and switch == 'start':
            wandb.init(
                project=self.projcode,
                name=self.run_name,
                config={
                },
                settings=wandb.Settings(start_method="fork"),
                tags = [f"pct_{self.args.pct}", f"seed_{self.args.seed}"]
            )
            wandb.define_metric("training/*", step_metric="Iter step")
            
            wandb.define_metric("train/*", step_metric="Iter step")
            wandb.define_metric("valid/*", step_metric="Iter step")
            wandb.define_metric("test/*", step_metric="Iter step")
                
        elif self.args.wandb and switch == 'finish':
            wandb.finish()
            
            
    # evaluation code for vanilla
    def evaluate(self, model, data_loader):
        model.eval()
        total_correct, total_num = 0, 0
        for data, attr, index in tqdm(data_loader, leave=False):
            label = attr[:, 0]
            data = data.to(self.device)
            label = label.to(self.device)

            with torch.no_grad():
                logit = model(data)
                pred = logit.data.max(1, keepdim=True)[1].squeeze(1)
                correct = (pred == label).long()
                total_correct += correct.sum()
                total_num += correct.shape[0]

        accs = total_correct/float(total_num)
        model.train()
        return accs
    

    def save_best(self, step):
        model_path = os.path.join(self.result_dir, "best_model_d.th")
        state_dict = {
            'steps': step,
            'state_dict': self.model_d.state_dict(),
            'optimizer': self.optimizer_d.state_dict(),
        }
        with open(model_path, "wb") as f:
            torch.save(state_dict, f)

        model_path = os.path.join(self.result_dir, "best_model_b.th")
        state_dict = {
            'steps': step,
            'state_dict': self.model_b.state_dict(),
            'optimizer': self.optimizer_b.state_dict(),
        }
        with open(model_path, "wb") as f:
            torch.save(state_dict, f)

        # print(f'{step} model saved...')


    def board_vanilla_acc(self, step, inference=None):
        # check label network
        valid_accs_d = self.evaluate(self.model_d, self.valid_loader)
        test_accs_d = self.evaluate(self.model_d, self.test_loader)

        if inference:
            print(f'test acc: {test_accs_d.item()}')
            import sys
            sys.exit(0)

        if step > 0:
            if valid_accs_d >= self.best_valid_acc_d:
                self.best_valid_acc_d = valid_accs_d

            if test_accs_d >= self.best_test_acc_d:
                self.best_test_acc_d = test_accs_d
                self.save_best(step)

        # print(f'valid_d: {valid_accs_d} || test_d: {test_accs_d} ')
        
        if self.args.wandb:
            wandb.log({
                    "Iter step": step,
                    "valid/accs_d": valid_accs_d,
                    "test/accs_d": test_accs_d,
                    "valid/best_acc_b": self.best_valid_acc_b,
                    "valid/best_acc_d": self.best_valid_acc_d,
                    "test/best_acc_b": self.best_test_acc_b,
                    "test/best_acc_d": self.best_test_acc_d,
                })


    def board_lff_acc(self, step, inference=None):
        # check label network
        valid_accs_b = self.evaluate(self.model_b, self.valid_loader)
        test_accs_b = self.evaluate(self.model_b, self.test_loader)

        valid_accs_d = self.evaluate(self.model_d, self.valid_loader)
        test_accs_d = self.evaluate(self.model_d, self.test_loader)

        if inference:
            print(f'test acc: {test_accs_d.item()}')
            import sys
            sys.exit(0)

        if step > 0:
            if valid_accs_b >= self.best_valid_acc_b:
                self.best_valid_acc_b = valid_accs_b

            if test_accs_b >= self.best_test_acc_b:
                self.best_test_acc_b = test_accs_b

            if valid_accs_d >= self.best_valid_acc_d:
                self.best_valid_acc_d = valid_accs_d

            if test_accs_d >= self.best_test_acc_d:
                self.best_test_acc_d = test_accs_d
                self.save_best(step)

        # print(f'valid_b: {valid_accs_b} || test_b: {test_accs_b} ')
        # print(f'valid_d: {valid_accs_d} || test_d: {test_accs_d} ')
        
        if self.args.wandb:
            wandb.log({
                    "Iter step": step,
                    "valid/accs_b": valid_accs_b,
                    "valid/accs_d": valid_accs_d,
                    "test/accs_b": test_accs_b,
                    "test/accs_d": test_accs_d,
                    "valid/best_acc_b": self.best_valid_acc_b,
                    "valid/best_acc_d": self.best_valid_acc_d,
                    "test/best_acc_b": self.best_test_acc_b,
                    "test/best_acc_d": self.best_test_acc_d,
                })
            

    def board_pretrain_best_acc(self, i, model_b, best_valid_acc_b, step):
        # check label network
        valid_accs_b = self.evaluate(model_b, self.valid_loader)

        # print(f'best: {best_valid_acc_b}, curr: {valid_accs_b}')

        if valid_accs_b > best_valid_acc_b:
            best_valid_acc_b = valid_accs_b

            ######### copy parameters #########
            self.best_model_b = copy.deepcopy(model_b)
            # print(f'early model {i}th saved...')

        return best_valid_acc_b


    def pretrain_b_ensemble_best(self, args):
        train_iter = iter(self.train_loader)
            
        index_dict, label_dict, gt_prob_dict = {}, {}, {}

        for i in range(self.args.num_bias_models):
            best_valid_acc_b = 0
            print(f'{i}th model working...')
            del self.model_b
            self.best_model_b = None
            self.model_b = get_backbone(self.model, self.num_classes, args=self.args, pretrained=self.pretrained, first_stage=True).to(self.device)
            self.optimizer_b = torch.optim.Adam(self.model_b.parameters(), lr=self.lr)
            
            iter_cnt = 0
            for step in tqdm(range(self.args.biased_model_train_iter)):
                try:
                    index, data, attr, _ = next(train_iter)
                except:
                    train_iter = iter(self.train_loader)
                    index, data, attr, _ = next(train_iter)

                data = data.to(self.device)
                attr = attr.to(self.device)
                label = attr[:, args.target_attr_idx]

                logit_b = self.model_b(data)
                loss_b_update = self.bias_criterion(logit_b, label)
                loss = loss_b_update.mean()

                self.optimizer_b.zero_grad()
                loss.backward()
                self.optimizer_b.step()
                
                iter_cnt += 1
                if iter_cnt % args.valid_freq == 0:
                    best_valid_acc_b = self.board_pretrain_best_acc(i, self.model_b, best_valid_acc_b, step)                    

            label_list, bias_list, pred_list, index_list, gt_prob_list, align_flag_list = [], [], [], [], [], []
            self.best_model_b.eval()
            
            for index, data, attr, _ in self.pretrain_loader:
                index = index.to(self.device)
                data = data.to(self.device)
                attr = attr.to(self.device)
                label = attr[:, args.target_attr_idx]
                bias_label = attr[:, args.bias_attr_idx]

                logit_b = self.best_model_b(data)
                prob = torch.softmax(logit_b, dim=-1)
                gt_prob = torch.gather(prob, index=label.unsqueeze(1), dim=1).squeeze(1)

                label_list += label.tolist()
                index_list += index.tolist()
                gt_prob_list += gt_prob.tolist()
                align_flag_list += (label == bias_label).tolist()

            index_list = torch.tensor(index_list)
            label_list = torch.tensor(label_list)
            gt_prob_list = torch.tensor(gt_prob_list)
            align_flag_list = torch.tensor(align_flag_list)

            align_mask = ((gt_prob_list > args.biased_model_softmax_threshold) & (align_flag_list == True)).long()
            conflict_mask = ((gt_prob_list > args.biased_model_softmax_threshold) & (align_flag_list == False)).long()
            mask = (gt_prob_list > args.biased_model_softmax_threshold).long()

            exceed_align = index_list[align_mask.nonzero().squeeze(1)]
            exceed_conflict = index_list[conflict_mask.nonzero().squeeze(1)]
            exceed_mask = index_list[mask.nonzero().squeeze(1)]

            model_index = i
            index_dict[f'{model_index}_exceed_align'] = exceed_align
            index_dict[f'{model_index}_exceed_conflict'] = exceed_conflict
            index_dict[f'{model_index}_exceed_mask'] = exceed_mask
            label_dict[model_index] = label_list
            gt_prob_dict[model_index] = gt_prob_list

            log_dict = {
                f"{model_index}_exceed_align": len(exceed_align),
                f"{model_index}_exceed_conflict": len(exceed_conflict),
                f"{model_index}_exceed_mask": len(exceed_mask),
            }

        exceed_mask = [(gt_prob_dict[i] > args.biased_model_softmax_threshold).long() for i in
                        range(self.args.num_bias_models)]
        exceed_mask_align = [
            ((gt_prob_dict[i] > args.biased_model_softmax_threshold) & (align_flag_list == True)).long() for i in
            range(self.args.num_bias_models)]
        exceed_mask_conflict = [
            ((gt_prob_dict[i] > args.biased_model_softmax_threshold) & (align_flag_list == False)).long() for i in
            range(self.args.num_bias_models)]

        mask_sum = torch.stack(exceed_mask).sum(dim=0)
        mask_sum_align = torch.stack(exceed_mask_align).sum(dim=0)
        mask_sum_conflict = torch.stack(exceed_mask_conflict).sum(dim=0)

        total_exceed_mask = index_list[(mask_sum >= self.args.agreement).long().nonzero().squeeze(1)]
        total_exceed_align = index_list[(mask_sum_align >= self.args.agreement).long().nonzero().squeeze(1)]
        total_exceed_conflict = index_list[(mask_sum_conflict >= self.args.agreement).long().nonzero().squeeze(1)]

        print(f'exceed mask length: {total_exceed_mask.size(0)}')
        curr_index_label = torch.index_select(label_dict[0].unsqueeze(1).to(self.device), 0,
                                              torch.tensor(total_exceed_mask).long().to(self.device))
        curr_align_index_label = torch.index_select(label_dict[0].unsqueeze(1).to(self.device), 0,
                                                    torch.tensor(total_exceed_align).long().to(self.device))
        curr_conflict_index_label = torch.index_select(label_dict[0].unsqueeze(1).to(self.device), 0,
                                                       torch.tensor(total_exceed_conflict).long().to(self.device))
        log_dict = {
            f"total_exceed_align": len(total_exceed_align),
            f"total_exceed_conflict": len(total_exceed_conflict),
            f"total_exceed_mask": len(total_exceed_mask),
        }

        total_exceed_mask = torch.tensor(total_exceed_mask)

        for key, value in log_dict.items():
            print(f"* {key}: {value}")
        print(f"* EXCEED DATA COUNT: {Counter(curr_index_label.squeeze(1).tolist())}")
        print(f"* EXCEED DATA (ALIGN) COUNT: {Counter(curr_align_index_label.squeeze(1).tolist())}")
        print(f"* EXCEED DATA (CONFLICT) COUNT: {Counter(curr_conflict_index_label.squeeze(1).tolist())}")
        
        return total_exceed_mask
    

    def train_vanilla(self, args): 
        print("training vanilla ours...")
        train_iter = iter(self.train_loader)

        iter_cnt = 0
        for step in tqdm(range(args.num_steps)):
            if iter_cnt == 0: 
                self.board_vanilla_acc(iter_cnt) # eval on init state

            try:
                index, data, attr, _ = next(train_iter)
            except:
                train_iter = iter(self.train_loader)
                index, data, attr, _ = next(train_iter)
                
            data = data.to(self.device)
            attr = attr.to(self.device)
            label = attr[:, args.target_attr_idx]

            logit_d = self.model_d(data)
            loss_d_update = self.criterion(logit_d, label)
            loss = loss_d_update.mean()

            self.optimizer_d.zero_grad()
            loss.backward()
            self.optimizer_d.step()

            iter_cnt += 1
            if iter_cnt % args.valid_freq == 0:
                self.board_vanilla_acc(iter_cnt)
    
        if args.num_steps % args.valid_freq != 0:
            self.board_vanilla_acc(iter_cnt) # eval on last iter


    def train_lff(self, args):
        print('Training LfF ours...')

        train_iter = iter(self.train_loader)

        iter_cnt = 0
        for step in tqdm(range(args.num_steps)):
            if iter_cnt == 0: 
                self.board_lff_acc(iter_cnt) # eval on init state
            
            try:
                index, data, attr, _ = next(train_iter)
            except:
                train_iter = iter(self.train_loader)
                index, data, attr, _ = next(train_iter)
                
            data = data.to(self.device)
            attr = attr.to(self.device)
            index = index.to(self.device)
            label = attr[:, args.target_attr_idx]

            logit_b = self.model_b(data)
            logit_d = self.model_d(data)

            loss_b = self.criterion(logit_b, label).detach()
            loss_d = self.criterion(logit_d, label).detach()

            if np.isnan(loss_b.mean().item()):
                raise NameError('loss_b')
            if np.isnan(loss_d.mean().item()):
                raise NameError('loss_d')

            # EMA sample loss
            self.sample_loss_ema_b.update(loss_b, index)
            self.sample_loss_ema_d.update(loss_d, index)

            # class-wise normalize
            loss_b = self.sample_loss_ema_b.parameter[index].clone().detach()
            loss_d = self.sample_loss_ema_d.parameter[index].clone().detach()

            if np.isnan(loss_b.mean().item()):
                raise NameError('loss_b_ema')
            if np.isnan(loss_d.mean().item()):
                raise NameError('loss_d_ema')

            label_cpu = label.cpu()

            for c in range(self.num_classes):
                class_index = np.where(label_cpu == c)[0]
                max_loss_b = self.sample_loss_ema_b.max_loss(c) + 1e-8
                max_loss_d = self.sample_loss_ema_d.max_loss(c)
                loss_b[class_index] /= max_loss_b
                loss_d[class_index] /= max_loss_d

            # re-weighting based on loss value / generalized CE for biased model
            loss_weight = loss_b / (loss_b + loss_d + 1e-8)

            if np.isnan(loss_weight.mean().item()):
                raise NameError('loss_weight')
            
            loss_b_update = self.bias_criterion(logit_b, label)
            loss_d_update = self.criterion(logit_d, label) * loss_weight.to(self.device)

            if np.isnan(loss_b_update.mean().item()):
                loss_b_update = torch.tensor(0, dtype=torch.float32)

            if np.isnan(loss_d_update.mean().item()):
                raise NameError('loss_d_update')

            loss = loss_b_update.mean() + loss_d_update.mean()

            self.optimizer_b.zero_grad()
            self.optimizer_d.zero_grad()
            loss.backward()
            self.optimizer_b.step()
            self.optimizer_d.step()
                    
            iter_cnt += 1
            if iter_cnt % args.valid_freq == 0:
                self.board_lff_acc(iter_cnt)
                
        if args.num_steps % args.valid_freq != 0:
            self.board_vanilla_acc(iter_cnt) # eval on last iter

           
    def train_lff_be(self, args):
        print('Training LfF with BiasEnsemble ours...')

        train_iter = iter(self.train_loader)
        train_num = len(self.train_dataset.dataset)

        # Mask for original dataset for BCDs.
        mask_index = torch.zeros(train_num, 1)
        self.conflicting_index = torch.zeros(train_num, 1)
        self.label_index = torch.zeros(train_num).long().to(self.device)
        
        #### BiasEnsemble ####
        # BCDs learns original dataset.
        pseudo_align_flag = self.pretrain_b_ensemble_best(args)
        mask_index[pseudo_align_flag] = 1
        
        del self.model_b
        self.model_b = get_backbone(self.model, self.num_classes, args=self.args, pretrained=self.pretrained, first_stage=True).to(self.device)
        self.optimizer_b = torch.optim.Adam(self.model_b.parameters(), lr=self.lr)

        iter_cnt = 0
        for step in tqdm(range(args.num_steps)):
            if iter_cnt == 0: 
                self.board_lff_acc(iter_cnt) # eval on init state

            # train main model
            try:
                index, data, attr, _ = next(train_iter)
            except:
                train_iter = iter(self.train_loader)
                index, data, attr, _ = next(train_iter)
                
            data = data.to(self.device)
            attr = attr.to(self.device)
            index = index.to(self.device)
            label = attr[:, args.target_attr_idx]

            logit_b = self.model_b(data)
            logit_d = self.model_d(data)

            loss_b = self.criterion(logit_b, label).detach()
            loss_d = self.criterion(logit_d, label).detach()

            if np.isnan(loss_b.mean().item()):
                raise NameError('loss_b')
            if np.isnan(loss_d.mean().item()):
                raise NameError('loss_d')

            # EMA sample loss
            self.sample_loss_ema_b.update(loss_b, index)
            self.sample_loss_ema_d.update(loss_d, index)

            # class-wise normalize
            loss_b = self.sample_loss_ema_b.parameter[index].clone().detach()
            loss_d = self.sample_loss_ema_d.parameter[index].clone().detach()

            if np.isnan(loss_b.mean().item()):
                raise NameError('loss_b_ema')
            if np.isnan(loss_d.mean().item()):
                raise NameError('loss_d_ema')

            label_cpu = label.cpu()

            for c in range(self.num_classes):
                class_index = np.where(label_cpu == c)[0]
                max_loss_b = self.sample_loss_ema_b.max_loss(c) + 1e-8
                max_loss_d = self.sample_loss_ema_d.max_loss(c)
                loss_b[class_index] /= max_loss_b
                loss_d[class_index] /= max_loss_d

            # re-weighting based on loss value / generalized CE for biased model
            loss_weight = loss_b / (loss_b + loss_d + 1e-8)

            curr_align_flag = torch.index_select(mask_index.to(self.device), 0, index)
            curr_align_flag = (curr_align_flag.squeeze(1) == 1)
            
            loss_b_update = self.criterion(logit_b[curr_align_flag], label[curr_align_flag])
            loss_d_update = self.criterion(logit_d, label) * loss_weight.to(self.device)

            if np.isnan(loss_b_update.mean().item()):
                loss_b_update = torch.tensor(0, dtype=torch.float32)

            if np.isnan(loss_d_update.mean().item()):
                loss_d_update = torch.tensor(0, dtype=torch.float32)

            loss = loss_b_update.mean() + loss_d_update.mean()

            self.optimizer_b.zero_grad()
            self.optimizer_d.zero_grad()
            loss.backward()
            self.optimizer_b.step()
            self.optimizer_d.step()
                    
            iter_cnt += 1
            if iter_cnt % args.valid_freq == 0:
                self.board_lff_acc(iter_cnt)
                
        if args.num_steps % args.valid_freq != 0:
            self.board_vanilla_acc(iter_cnt) # eval on last iter