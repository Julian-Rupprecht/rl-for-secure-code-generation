import numpy as np
import torch
from torch.utils.data import Dataset
import json
import os 
import pickle as pkl

class GenerationDataset(Dataset):
    def __init__(self, dataroot, output_dirs, logger, reposVul=True):
        self.dataroot = dataroot
        self.output_dirs = output_dirs
        self.gen_samples = []
        self.all_vuln_types = []
        self.all_compiler_results = []
        self.samples_info = []
        self.reposVul = reposVul
        self.logger = logger

        self.initialize()

    def initialize(self):
        gen_samples = []
        samples_info = []

        for output_dir in self.output_dirs:
            path = os.path.join(self.dataroot, output_dir, "gen_critic_scores.pkl")
            critic_score_fname = path

            if os.path.exists(critic_score_fname):
                gen_critic_scores = pkl.load(open(critic_score_fname, "rb"))
                samples, info = self.load_samples(gen_critic_scores)
                gen_samples.append(samples)
                samples_info.append(info)
            
        self.samples_info = samples_info
        self.gen_samples = gen_samples


    def load_samples(self, gen_critic_scores):
        samples = []
        info = []

        for idx, (prompt, code) in enumerate(zip(gen_critic_scores['prompt'], gen_critic_scores['code'])):
            samples.append((prompt, code))

            if self.reposVul:
                info.append((gen_critic_scores['gt_vuln_critic'][idx], gen_critic_scores['vuln_hidden_states'][idx], gen_critic_scores['gt_vuln_target'][idx]))
            else: 
                info.append((gen_critic_scores['gt_vuln_critic'][idx], gen_critic_scores['vuln_hidden_states'][idx], gen_critic_scores['gt_error'][idx]))
                
        return samples, info


    def sample_generation(self, item, info, z_score=False):
        input_ids, label_ids = zip(*item) 
        gt_vuln, vuln_logit, gt_error_or_vuln_target = zip(*info) 
    
        q_hats = torch.sigmoid(torch.tensor(vuln_logit, dtype=torch.float32))
        q_hats = torch.where(q_hats < 0.5, 1 - q_hats, q_hats)
        self.logger.info("jo")

        if self.reposVul:
            if z_score:
                advantage = torch.tensor(self.get_z_score_reward(gt_error_or_vuln_target))[:, None, None]
            else:
                advantage = torch.tensor(self.get_rewards(gt_error_or_vuln_target))[:, None, None] 
            
        else:
            advantage = torch.tensor(self.get_z_score_reward(gt_vuln, gt_error_or_vuln_target))[:, None, None]
       
        rewards = q_hats * advantage
        #rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        out_sample = {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'rewards': torch.tensor(rewards, dtype=torch.float32),
            'label_ids': torch.tensor(label_ids, dtype=torch.long)
        }

        return out_sample
    
    def __len__(self):
        return len(self.gen_samples)
    
    def __getitem__(self, idx):
        item_group = self.gen_samples[idx]
        info_group = self.samples_info[idx]

        return self.sample_generation(item_group, info_group)
        
    
    def get_z_score_reward(self, gt_vuln, gt_error=None):
        gt_vuln = np.array(gt_vuln)

        if not self.reposVul: 
            gt_error = np.array(gt_error)
        
        rewards = self.get_rewards(gt_vuln, gt_error)
        rewards = np.array(rewards)
        mean = np.mean(rewards)
        std = np.std(rewards)

        if std == 0:
            z_scores = np.zeros_like(rewards)
        else:
            z_scores = (rewards - mean) / std 

        return z_scores

    def get_rewards(self, gt_vuln, gt_error=None): 
        rewards = []

        if gt_error is None:
            for gt_v in gt_vuln:
                if gt_v >= 0.5:
                    reward = -0.5
                    self.logger.info(reward)
                else:
                    reward = 1.0

                rewards.append(reward)
        else: 
            for gt_v, gt_e in zip(gt_vuln, gt_error):

                if gt_v >= 0.5 and gt_e == 1:
                    reward = -1.0
                elif gt_v >= 0.5 and gt_e == 0:
                    reward = -0.6
                elif gt_v < 0.5 and gt_e == 1:
                    reward = -0.3
                else:
                    reward = 1.0
            
                rewards.append(reward)
                
        return rewards 