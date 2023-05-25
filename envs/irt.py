import os
import time
import copy
import vegas
import logging
import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as data
from math import exp as exp
from sklearn.metrics import roc_auc_score
from scipy import integrate

class IRT(nn.Module):
    def __init__(self, num_students, num_questions, num_dim):
        super().__init__()
        self.num_dim = num_dim
        self.num_students = num_students
        self.num_questions = num_questions
        self.theta = nn.Embedding(self.num_students, self.num_dim) # stu embed
        self.alpha = nn.Embedding(self.num_questions, self.num_dim) # que embed
        self.beta = nn.Embedding(self.num_questions, 1)

        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, student_ids, question_ids):
        theta = self.theta(student_ids) # B D
        alpha = self.alpha(question_ids)
        beta = self.beta(question_ids) # B 1
        pred = (alpha * theta).sum(dim=1, keepdim=True) + beta
        pred = torch.sigmoid(pred) # B 1
        return pred
    
    def init_stu_emb(self):
        for name, param in self.named_parameters():
            if 'weight' in name and 'theta' in name:
                nn.init.xavier_normal_(param)

class IRTModel:
    def __init__(self, args, num_students, num_questions, num_dim):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda')
        self.model = IRT(num_students, num_questions, num_dim).to(self.device)
    
    @property
    def name(self):
        return 'Item Response Theory'

    def init_stu_emb(self):
        self.model.init_stu_emb()

    def cal_loss(self, sids, query_rates, know_map):
        device = self.device 
        real = []
        pred = []
        all_loss = np.zeros(len(sids))
        with torch.no_grad():
            self.model.eval()
            for idx, sid in enumerate(sids):
                question_ids = list(query_rates[sid].keys())
                student_ids = [sid] * len(question_ids)
                labels = [query_rates[sid][qid] for qid in question_ids]
                real.append(np.array(labels))

                student_ids = torch.LongTensor(student_ids).to(device)
                question_ids = torch.LongTensor(question_ids).to(device)
                labels = torch.FloatTensor(labels).to(device).float()
                output = self.model(student_ids, question_ids).view(-1)
                loss = self._loss_function(output, labels).cpu().detach().numpy()
                all_loss[idx] = loss.item()
                pred.append(np.array(output.tolist()))

            self.model.train()
        
        return all_loss, pred, real

    def train(self, train_data, lr, batch_size, epochs, path):
        device = self.device
        logger = logging.getLogger("Pretrain")
        logger.info('train on {}'.format(device))

        train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        best_loss = 1000000
        for ep in range(1, epochs + 1):
            loss = 0.0
            for cnt, (student_ids, question_ids, _, labels) in enumerate(train_loader):
                student_ids = student_ids.to(device)
                question_ids = question_ids.to(device)
                labels = labels.to(device).float()
                pred = self.model(student_ids, question_ids).view(-1)
                bz_loss = self._loss_function(pred, labels)
                optimizer.zero_grad()
                bz_loss.backward()
                optimizer.step()
                loss += bz_loss.data.float()
            loss /= len(train_loader)
            logger.info('Epoch [{}]: loss={:.5f}'.format(ep, loss))
            if loss < best_loss:
                best_loss = loss
                logger.info('Store model')
                self.adaptest_save(path)
            # theta = self.model.theta.weight.data.cpu().numpy()
            # print(np.max(theta), np.min(theta))
            # alpha = self.model.alpha.weight.data.cpu().numpy()
            # print(np.max(alpha), np.min(alpha))
                
                # if cnt % 50 == 0:
                #     logger.info('Epoch [{}] Batch [{}]: loss={:.5f}'.format(ep, cnt, loss / cnt))


    # for see
    def train_all(self, train_data, lr, batch_size, epochs, path):
        device = self.device
        logger = logging.getLogger("SEE")
        logger.info('train on {}'.format(device))

        train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        best_loss = 1000000
        for ep in range(1, epochs + 1):
            loss = 0.0
            for cnt, (student_ids, question_ids, _, labels) in enumerate(train_loader):
                student_ids = student_ids.to(device)
                question_ids = question_ids.to(device)
                labels = labels.to(device).float()
                pred = self.model(student_ids, question_ids).view(-1)
                bz_loss = self._loss_function(pred, labels)
                optimizer.zero_grad()
                bz_loss.backward()
                optimizer.step()
                loss += bz_loss.data.float()
            loss /= len(train_loader)
            logger.info('Epoch [{}]: loss={:.5f}'.format(ep, loss))
            
            theta = self.model.theta.weight.data.cpu().numpy()
            print('theta',np.max(theta), np.min(theta))
            alpha = self.model.alpha.weight.data.cpu().numpy()
            print('alpha',np.max(alpha), np.min(alpha))
        logger.info('Store Student')
        np.save(path, self.model.theta.weight.data.cpu().numpy())
        
     

    def update(self, tested_dataset, lr, epochs, batch_size):
        device = self.device
        optimizer = torch.optim.Adam(self.model.theta.parameters(), lr=lr)
        dataloader = torch.utils.data.DataLoader(tested_dataset, batch_size=batch_size, shuffle=True)
      
        for ep in range(1, epochs + 1):
            loss = 0.0
            # log_steps = 100
            for cnt, (student_ids, question_ids, _, labels) in enumerate(dataloader):
                student_ids = student_ids.to(device)
                question_ids = question_ids.to(device)
                labels = labels.to(device).float()
                pred = self.model(student_ids, question_ids).view(-1)
                bz_loss = self._loss_function(pred, labels)
                optimizer.zero_grad()
                bz_loss.backward()
                optimizer.step()
                loss += bz_loss.data.float()
                # if cnt % log_steps == 0:
                # print('Epoch [{}] Batch [{}]: loss={:.3f}'.format(ep, cnt, loss / cnt))


    def adaptest_save(self, path):
        model_dict = self.model.state_dict()
        model_dict = {k:v for k,v in model_dict.items() if 'alpha' in k or 'beta' in k}
        torch.save(model_dict, path)

    def adaptest_load(self, path):
        self.model.load_state_dict(torch.load(path), strict=False)
        self.model.to(self.device)

    def get_pred(self, user_ids, avail_questions, concept_map):
        device = self.device

        pred_all = {}

        with torch.no_grad():
            self.model.eval()
            for sid in user_ids:
                pred_all[sid] = {}
                question_ids =  list(avail_questions[sid])
                student_ids = [sid] * len(question_ids)
              
                student_ids = torch.LongTensor(student_ids).to(device)
                question_ids = torch.LongTensor(question_ids).to(device)
                output = self.model(student_ids, question_ids).view(-1).tolist()
                for i, qid in enumerate(list(avail_questions[sid])):
                    pred_all[sid][qid] = output[i]
            self.model.train()

        return pred_all

    def _loss_function(self, pred, real):
        return -(real * torch.log(0.0001 + pred) + (1 - real) * torch.log(1.0001 - pred)).mean()
    
    def get_alpha(self, question_id):
        """ get alpha of one question
        Args:
            question_id: int, question id
        Returns:
            alpha of the given question, shape (num_dim, )
        """
        return self.model.alpha.weight.data.cpu().numpy()[question_id]
    
    def get_beta(self, question_id):
        """ get beta of one question
        Args:
            question_id: int, question id
        Returns:
            beta of the given question, shape (1, )
        """
        return self.model.beta.weight.data.cpu().numpy()[question_id]
    
    def get_theta(self, student_id):
        """ get theta of one student
        Args:
            student_id: int, student id
        Returns:
            theta of the given student, shape (num_dim, )
        """
        return self.model.theta.weight.data.cpu().numpy()[student_id]

    def get_kli(self, student_id, question_id, n, pred_all):
        """ get KL information
        Args:
            student_id: int, student id
            question_id: int, question id
            n: int, the number of iteration
        Returns:
            v: float, KL information
        """
        if n == 0:
            return np.inf
        device = self.device 
        dim = self.model.num_dim
        sid = torch.LongTensor([student_id]).to(device)
        qid = torch.LongTensor([question_id]).to(device)
        theta = self.get_theta(sid) # (num_dim, )
        alpha = self.get_alpha(qid) # (num_dim, )
        beta = self.get_beta(qid)[0] # float value
        pred_estimate = pred_all[student_id][question_id]
        
        def kli(x):
            """ The formula of KL information. Used for integral.
            Args:
                x: theta of student sid
            """
            if type(x) == float:
                x = np.array([x])
            pred = np.matmul(alpha.T, x) + beta
            pred = 1 / (1 + np.exp(-pred))
            q_estimate = 1  - pred_estimate
            q = 1 - pred
            return pred_estimate * np.log(pred_estimate / pred) + \
                    q_estimate * np.log((q_estimate / q))
        c = 3 
        boundaries = [[theta[i] - c / np.sqrt(n), theta[i] + c / np.sqrt(n)] for i in range(dim)]
        if len(boundaries) == 1:
            # KLI
            v, err = integrate.quad(kli, boundaries[0][0], boundaries[0][1])
            return v
        # MKLI
        integ = vegas.Integrator(boundaries)
        result = integ(kli, nitn=10, neval=1000)
        return result.mean

    def get_fisher(self, student_id, question_id, pred_all):
        """ get Fisher information
        Args:
            student_id: int, student id
            question_id: int, question id
        Returns:
            fisher_info: matrix(num_dim * num_dim), Fisher information
        """
        device = self.device
        qid = torch.LongTensor([question_id]).to(device)
        alpha = self.model.alpha(qid).clone().detach().cpu()
        pred = pred_all[student_id][question_id]
        q = 1 - pred
        fisher_info = (q*pred*(alpha * alpha.T)).numpy()
        return fisher_info
    
    def expected_model_change(self, sid: int, qid: int, pred_all: dict, concept_map):
        """ get expected model change
        Args:
            student_id: int, student id
            question_id: int, question id
        Returns:
            float, expected model change
        """
        # epochs = self.args.cdm_epoch # epoch lr no effect
        epochs = 1
        # lr = self.args.cdm_lr
        lr = 0.01

        device = self.device 
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        for name, param in self.model.named_parameters():
            if 'theta' not in name:
                param.requires_grad = False

        original_weights = self.model.theta.weight.data.clone()

        student_id = torch.LongTensor([sid]).to(device)
        question_id = torch.LongTensor([qid]).to(device)
        correct = torch.LongTensor([1]).to(device).float()
        wrong = torch.LongTensor([0]).to(device).float()

        for ep in range(epochs):
            optimizer.zero_grad()
            pred = self.model(student_id, question_id)
            loss = self._loss_function(pred, correct)
            loss.backward()
            optimizer.step()

        pos_weights = self.model.theta.weight.data.clone()
        self.model.theta.weight.data.copy_(original_weights)

        for ep in range(epochs):
            optimizer.zero_grad()
            pred = self.model(student_id, question_id)
            loss = self._loss_function(pred, wrong)
            loss.backward()
            optimizer.step()

        neg_weights = self.model.theta.weight.data.clone()
        self.model.theta.weight.data.copy_(original_weights)

        for param in self.model.parameters():
            param.requires_grad = True

        pred = pred_all[sid][qid]
        return pred * torch.norm(pos_weights - original_weights).item() + \
               (1 - pred) * torch.norm(neg_weights - original_weights).item()
        


