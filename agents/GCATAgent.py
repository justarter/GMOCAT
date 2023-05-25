import time
import logging
import random
import numpy as np
import copy as cp
import torch 
from util import *
from collections import defaultdict
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
from torch.distributions import Categorical

from .dataset import Dataset, collate_fn

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.action_masks = []
        self.dones = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.action_masks[:]
        del self.dones[:]

class GCATAgent(object):
    def __init__(self, env, fa, args):
        self.env = env
        self.fa = fa
        self.args = args
        self.device = torch.device('cuda')
        
        self.memory = Memory()
        self.logger = logging.getLogger(f'{args.FA}')

        self.all_AUC, self.all_ACC = defaultdict(list), defaultdict(list)
        self.best_AUC, self.best_ACC = None, None
        self.best_value = -1
        self.collate_fn = collate_fn(self.env.item_num)

        self.all_cov = defaultdict(list)

        self.know_map = env.know_map

    def train(self):
        for epoch in range(self.args.training_epoch):
            self.collecting_data_update_model("training", epoch)
            # self.collecting_data_update_model("validation", epoch)
            for iter in range(3):
                self.env.re_split_data(_id=iter+100)
                self.collecting_data_update_model("evaluation", epoch)
            # average evaluations
            self.logger.info("Mean Metrics: ")
            for step in self.args.ST:
                self.logger.info(str(step)+ 'AUC: ' + str(round(np.mean(self.all_AUC[step]),4))+", ACC: "+str(round(np.mean(self.all_ACC[step]),4)))
            for step in self.args.ST:
                tmp_auc = ','.join([str(x) for x in self.all_AUC[step]])
                tmp_acc = ','.join([str(x) for x in self.all_ACC[step]])
                self.logger.info(str(step)+ 'AUC: ' + tmp_auc +", ACC: "+ tmp_acc )
            # compare on AUC
            if np.mean(self.all_AUC[self.args.ST[-1]]) > self.best_value:
                self.best_value = np.mean(self.all_AUC[self.args.ST[-1]])
                self.best_AUC, self.best_ACC = self.all_AUC, self.all_ACC
            # best metric
            self.logger.info("Best Mean Metrics: ")
            for step in self.args.ST:
                self.logger.info(str(step)+ 'AUC: ' + str(round(np.mean(self.best_AUC[step]),4))+", ACC: "+str(round(np.mean(self.best_ACC[step]),4)))
            # empty
            self.all_AUC, self.all_ACC = defaultdict(list), defaultdict(list)
            # re-split
            self.env.re_split_data(_id=None)

        self.logger.info('\n'.join([str(round(np.mean(self.all_cov[i]),4)) for i in range(self.args.T)]))
        self.all_cov = defaultdict(list)

    def collecting_data_update_model(self, flag="training", epoch=0):
        if flag=="training":
            dataset = Dataset(self.env.get_records('training'))
            loader = torch.utils.data.DataLoader(
                dataset, collate_fn=self.collate_fn, batch_size=self.args.train_bs, num_workers=3, shuffle=True, drop_last=False)
        elif flag=="validation":
            dataset = Dataset(self.env.get_records('validation'))
            loader = torch.utils.data.DataLoader(
                dataset, collate_fn=self.collate_fn, batch_size=self.args.test_bs, num_workers=3, shuffle=False, drop_last=False)
        elif flag=="evaluation":
            # print('Begin eval')
            dataset = Dataset(self.env.get_records('evaluation'))
            loader = torch.utils.data.DataLoader(
                dataset, collate_fn=self.collate_fn, batch_size=self.args.test_bs, num_workers=3, shuffle=False, drop_last=False)
        else:
            self.logger.info('Data flag no exist')
            exit()

        # infos for all student's metric
        infos = {step:[] for step in self.args.ST}
        all_loss = 0
        actions_list = []
        for batch in loader:
            # rollout
            selected_users = batch['user_ids'] 
            state = self.env.reset_with_users(selected_users)
            action_mask = batch['mask'].long()
            train_mask = torch.zeros(action_mask.shape[0], self.env.item_num).long()
            done = False
            cnt_step = 0
            while not done:
                batch_question = cp.deepcopy(state['batch_question']) # B T+1
                batch_answer = cp.deepcopy(state['batch_answer'])
                pnt = [cnt_step for _ in range(len(selected_users))]
                batch_know, batch_know_num = self.get_know_num(batch_question)
                # print(batch_know.shape, batch_know_num.shape)
                data = {"p_rec":batch_question, "p_t":pnt, \
                         "a_rec":batch_answer, 'kn_rec':batch_know, 'kn_num':batch_know_num} 

                logits = self.fa.policy_old.predict(data)
                # add mask
                inf_mask = torch.clamp(torch.log(action_mask.float()), min=torch.finfo(torch.float32).min)
                logits = logits + inf_mask 
                action_probs = F.softmax(logits, dim=-1)
                dist = Categorical(action_probs)
                action = dist.sample() # sample, not for max

                if flag == 'training':
                    self.memory.actions.append(action) # tensor,B, 20
                    self.memory.logprobs.append(dist.log_prob(action))  # tensor, B ,20
                    self.memory.action_masks.append(cp.deepcopy(action_mask)) # tensor, B*N, 20
                    self.memory.states.extend([(batch_question[i],pnt[i],batch_answer[i]) for i in range(len(selected_users))]) # np, B, 20
                
                # update mask
                action_mask[range(len(action)), action] = 0
                train_mask[range(len(train_mask)), action] = 1
                # update state
                action = tensor_to_numpy(action)
                # rwd is (B, 3)
                if (epoch == self.args.training_epoch-1) and flag=='evaluation':
                    state_next, rwd, done, all_info, cov = self.env.step(action, True)
                    self.all_cov[cnt_step].append(cov)
                else:
                    state_next, rwd, done, all_info, _ = self.env.step(action, False)

                if flag == 'training':
                    self.memory.rewards.append(rwd) # np,B*3 ,20
                    self.memory.dones.append(done) # bool, 20

                state = state_next
                cnt_step += 1
                if cnt_step in self.args.ST:
                    infos[cnt_step].extend(all_info)


            # train policy
            if flag == "training":
                rewards = []
                discounted_reward = 0
                for reward, is_terminal in zip(reversed(self.memory.rewards), reversed(self.memory.dones)):
                    if is_terminal:
                        discounted_reward = 0
                    discounted_reward = reward + (self.args.gamma * discounted_reward)
                    rewards.insert(0, discounted_reward)
                rewards = np.concatenate(rewards, axis=0)
                rewards = torch.tensor(rewards, dtype=torch.float32).detach()
                # rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
                old_actions = torch.cat(self.memory.actions, dim=0).detach() # 20B,
                old_logprobs = torch.cat(self.memory.logprobs, dim=0).detach() # 20B,
                old_actionmask = torch.cat(self.memory.action_masks, dim=0).detach() # 20B, N
                # print(rewards.shape, old_actions.shape, old_logprobs.shape,old_actionmask.shape)
                ep_length = len(old_actions)
                batch_size = ep_length//4
                mini_batch_indicies = [np.arange(ep_length, dtype=np.int64)[i:i + batch_size] for i in np.arange(0, ep_length, batch_size)]
                for indices in mini_batch_indicies:
                    b_rewards = rewards[indices].to(self.device)
                    b_old_states = self.convert_state([self.memory.states[idx] for idx in indices])
                    b_old_actions = old_actions[indices].to(self.device)
                    b_old_logprobs = old_logprobs[indices].to(self.device)
                    b_old_actionmask = old_actionmask[indices].to(self.device)
                    loss = self.fa.optimize_model(b_old_states, b_old_actions, b_old_logprobs,b_old_actionmask,b_rewards)
                
                all_loss += loss
                # print('transfer weights')
                self.fa.transfer_weights()
                # self.logger.info('policy loss: ' + str(np.round(loss, 4)))
                
            # reset student 
            self.env.model.init_stu_emb()
            # clear memory
            self.memory.clear_memory()

        if flag == 'training':
            self.logger.info('policy loss: ' + str(np.round(all_loss/len(loader), 4)))

        # compute AUC ACC
        if flag != "training":
            for step in self.args.ST:
                pred, label = [], []
                for metric in infos[step]:
                    pred.append(metric['pred'])
                    label.append(metric['label'])
                pred = np.concatenate(pred, axis=0)
                label = np.concatenate(label,axis=0)
                
                pred_bin = np.where(pred > 0.5, 1, 0)
                ACC = np.sum(np.equal(pred_bin, label)) / len(pred_bin) 
                try:
                    AUC = roc_auc_score(label, pred)
                except ValueError:
                    AUC = -1
                self.all_ACC[step].append(ACC)
                self.all_AUC[step].append(AUC)

        self.logger.info(f'epoch: {epoch}, flag: {flag}')

    def convert_state(self, batch):
        p_r = np.array([item[0] for item in batch])
        pnt = np.array([item[1] for item in batch])
        a_rec = np.array([item[2] for item in batch])
        # print(p_r.shape,pnt.shape,a_rec.shape,kn_rec.shape)# (B ,T) (B,) (B,T)(B,T)
        batch_know, batch_know_num = self.get_know_num(p_r)
        data = {"p_rec":p_r, "p_t":pnt, "a_rec":a_rec, \
                    'kn_rec':batch_know, 'kn_num':batch_know_num} 
        return data 

    def get_know_num(self, batch_question):
        batch_know = []
        batch_know_num = []
        for i in range(len(batch_question)):
            concepts_embs = []
            concepts_num = []

            for qid in batch_question[i]:
                concepts_emb = [0.] * self.env.know_num
                concepts = self.env.know_map[qid]
                for concept in concepts:
                    concepts_emb[concept] = 1.0
                concepts_embs.append(concepts_emb)
                if qid ==0:
                    concepts_num.append(1)
                else:
                    concepts_num.append(len(self.env.know_map[qid]))

            concepts_embs = torch.Tensor(concepts_embs)
            batch_know.append(concepts_embs)
            concepts_num = torch.Tensor(concepts_num)
            batch_know_num.append(concepts_num)
        batch_know = torch.stack(batch_know) # B, T, N
        batch_know_num = torch.stack(batch_know_num) # B, T
        return batch_know, batch_know_num