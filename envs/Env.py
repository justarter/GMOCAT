import sys
import json
import os
import math
import torch
import random
import logging
import copy as cp
import numpy as np
from util import *
from collections import defaultdict


from .irt import IRTModel
from .ncd import NCDModel


class Env(object):
    def __init__(self, args):
        self.args = args

        self.T = args.T
        self.data_name = args.data_name
        self.CDM = args.CDM
        self.device = torch.device('cuda')
        self.seed = args.seed # for split train/test, sup/query

        self.rates = {}
        self.users = {}
        self.logger = logging.getLogger(f'{args.FA}')
        
        self.rates, self._item_num, self.know_map, self._know_num, self.know2item = self.load_data()
        self.setup_train_test()
        self.sup_rates, self.query_rates = self.split_data(_id=None)

        self.model, self.dataset = self.load_CDM()

    def split_data(self, _id):
        sup_rates, query_rates = {}, {}
        for u in range(self.user_num):
            all_items = list(self.rates[u].keys())
            if _id is None:
                np.random.shuffle(all_items)
            else:
                # fix seed
                random.Random(_id + u).shuffle(all_items)
            N = len(all_items)
           
            sup_rates[u] = {it: self.rates[u][it] for it in all_items[:-N//5]}
            query_rates[u] = {it: self.rates[u][it] for it in all_items[-N//5:]}
            
        return sup_rates, query_rates

    def re_split_data(self, _id):
        self.sup_rates, self.query_rates = self.split_data(_id)

    def get_records(self, flag):
        if flag == 'training':
            users = self.training
        elif flag == 'validation':
            users = self.validation
        elif flag == 'evaluation':
            users = self.evaluation
        else:
            exit()
        dataset = [{"user_id":uid, "q_ids":list(self.sup_rates[uid].keys()), 
                    "labels":list(self.sup_rates[uid].values())} for uid in users]
        return dataset

    def load_CDM(self):
        name = self.CDM
        if name == 'IRT':
            path = 'models/{}/{}_{}_{}.pt'.format(self.args.data_name, self.CDM, self.args.data_name, self.args.T)

            model = IRTModel(self.args, self.user_num, self.item_num, 1)
            model.adaptest_load(path)
            self.logger.info(path)

        elif name == 'NCD':
            path = 'models/{}/{}_{}_{}.pt'.format(self.args.data_name,self.CDM, self.args.data_name, self.args.T)

            model = NCDModel(self.args, self.user_num, self.item_num, self.know_num)
            model.adaptest_load(path)
            self.logger.info(path)
        else:
            self.logger.info('no CDM')
            exit()
        cdm_dataset = None
        return model, cdm_dataset

    def setup_train_test(self):
        N = self.user_num//10
        test_fold, valid_fold = 0, 1
        
        users_id = [idx for idx in range(self.user_num)]
        random.Random(self.args.seed).shuffle(users_id)

        self.evaluation = [uu for i,uu in enumerate(users_id) if i//N==test_fold]
        self.validation = [uu for i,uu in enumerate(users_id) if i//N==valid_fold]
        self.training = [uu for i, uu in enumerate(users_id) if i //
                     N != test_fold and i//N != valid_fold ]
        self.logger.info(f'train/val/test size: {len(self.training)}, {len(self.validation)},{len(self.evaluation)}')
    
    def load_data(self):
        with open(f'data/concept_map_{self.args.data_name}.json', encoding='utf8') as i_f:
            concept_data = json.load(i_f) 
        with open(f'data/train_task_{self.args.data_name}.json', encoding='utf8') as i_f:
            stus = json.load(i_f) 

        know2item = defaultdict(list)
        rates = {}
        items = set()
        user_cnt = -1
        know_map = {}
        know_map[0] = []  # the know of qid0 is empty
        for stu in stus:
            user_cnt += 1
            rates[user_cnt] = {}
            for qid, label in zip(stu['q_ids'], stu['labels']): 
                qid_pad = int(qid)+1
                rates[user_cnt][qid_pad] = int(label)
                items.add(qid_pad)
                know_map[qid_pad] = concept_data[str(qid)]
                for kk in concept_data[str(qid)]:
                    know2item[kk].append(qid_pad)
        max_itemid = max(items)

        knows = set()
        for know_list in know_map.values():
            knows.update(know_list)
        max_knowid = max(knows)
        
        return rates, max_itemid, know_map, max_knowid, know2item

    @property
    def user_num(self):
        return len(self.rates)

    @property
    def item_num(self):
        return self._item_num + 1
    
    @property
    def know_num(self):
        return self._know_num + 1
    

