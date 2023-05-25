import torch
import logging
import numpy as np
import torch.nn as nn
import torch.utils.data as data
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F

class NCD(nn.Module):
    def __init__(self, student_n, exer_n, knowledge_n, prednet_len1=128, prednet_len2=64):
        self.knowledge_dim = 10
        self.exer_n = exer_n
        self.emb_num = student_n
        self.prednet_input_len = self.knowledge_dim

        self.prednet_len1, self.prednet_len2 = prednet_len1, prednet_len2  # changeable

        super(NCD, self).__init__()

        # network structure
        self.student_emb = nn.Embedding(self.emb_num, self.knowledge_dim)
        self.k_difficulty = nn.Embedding(self.exer_n, self.knowledge_dim)
        self.e_discrimination = nn.Embedding(self.exer_n, 1)
        self.prednet_full1 = nn.Linear(self.prednet_input_len, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = nn.Linear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = nn.Linear(self.prednet_len2, 1)

        # initialization
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, stu_id, exer_id, kn_emb):
        # before prednet
        stu_emb = torch.sigmoid(self.student_emb(stu_id))
        k_difficulty = torch.sigmoid(self.k_difficulty(exer_id))
        e_discrimination = torch.sigmoid(self.e_discrimination(exer_id)) * 10
        # prednet
        # input_x = e_discrimination * (stu_emb - k_difficulty) * kn_emb
        input_x = e_discrimination * (stu_emb - k_difficulty)
        input_x = self.drop_1(torch.sigmoid(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.sigmoid(self.prednet_full2(input_x)))    
        output = torch.sigmoid(self.prednet_full3(input_x))

        return output

    def init_stu_emb(self):
        for name, param in self.named_parameters():
            if 'weight' in name and 'student' in name:
                nn.init.xavier_normal_(param)

    def apply_clipper(self):
        clipper = NoneNegClipper()
        self.prednet_full1.apply(clipper)
        self.prednet_full2.apply(clipper)
        self.prednet_full3.apply(clipper)

class NoneNegClipper(object):
    def __init__(self):
        super(NoneNegClipper, self).__init__()

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            a = torch.relu(torch.neg(w))
            w.add_(a)

class NCDModel:
    def __init__(self, args, num_students, num_questions, num_knowledges):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda')
        self.num_knowledges = num_knowledges
        self.model = NCD(num_students, num_questions, num_knowledges).to(self.device)
        self.loss_function = nn.BCELoss()

    @property
    def name(self):
        return 'Neural Cognitive Diagnosis'

    def init_stu_emb(self):
        self.model.init_stu_emb()

    def cal_loss(self, sids, query_rates, concept_map):
        device = self.device
        real = []
        pred = []
        all_loss = np.zeros(len(sids))
        with torch.no_grad():
            self.model.eval()
            for idx, sid in enumerate(sids):
                question_ids = list(query_rates[sid].keys())
                student_ids = [sid] * len(question_ids)
                concepts_embs = []
                for qid in question_ids:
                    concepts = concept_map[qid]
                    concepts_emb = [0.] * self.num_knowledges
                    for concept in concepts:
                        concepts_emb[concept] = 1.0
                    concepts_embs.append(concepts_emb)

                labels = [query_rates[sid][qid] for qid in question_ids]
                real.append(np.array(labels))

                student_ids = torch.LongTensor(student_ids).to(device)
                question_ids = torch.LongTensor(question_ids).to(device)
                concepts_embs = torch.Tensor(concepts_embs).to(device)
                labels = torch.LongTensor(labels).to(device)
                output = self.model(student_ids, question_ids, concepts_embs)
                loss = self._loss_function(output, labels).cpu().detach().numpy()
                all_loss[idx] = loss.item()
                pred.append(np.array(output.view(-1).tolist()))
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
            for cnt, (student_ids, question_ids, concepts_emb, labels) in enumerate(train_loader):
                student_ids = student_ids.to(device)
                question_ids = question_ids.to(device)
                concepts_emb = concepts_emb.to(device)
                labels = labels.to(device)
                pred = self.model(student_ids, question_ids, concepts_emb)
                bz_loss = self._loss_function(pred, labels)
                optimizer.zero_grad()
                bz_loss.backward()
                optimizer.step()
                self.model.apply_clipper()
                loss += bz_loss.data.float()
            loss /= len(train_loader)
            logger.info('Epoch [{}]: loss={:.5f}'.format(ep, loss))
    
            if loss < best_loss:
                best_loss = loss
                logger.info('Store model')
                self.adaptest_save(path)
                # if cnt % log_step == 0:
                #     logging.info('Epoch [{}] Batch [{}]: loss={:.5f}'.format(ep, cnt, loss / cnt))
    
    def _loss_function(self, pred, real):
        pred_0 = torch.ones(pred.size()).to(self.device) - pred
        output = torch.cat((pred_0, pred), 1)
        criteria = nn.NLLLoss()
        return criteria(torch.log(output), real)
    
    def adaptest_save(self, path):
        model_dict = self.model.state_dict()
        model_dict = {k:v for k,v in model_dict.items() if 'student' not in k}
        torch.save(model_dict, path)
    
    def adaptest_load(self, path):
        self.model.load_state_dict(torch.load(path), strict=False)
        self.model.to(self.device)
    
    def update(self, tested_dataset, lr, epochs, batch_size):
        device = self.device
        optimizer = torch.optim.Adam(self.model.student_emb.parameters(), lr=lr)
        dataloader = torch.utils.data.DataLoader(tested_dataset, batch_size=batch_size, shuffle=True)

        for ep in range(1, epochs + 1):
            loss = 0.0
            # log_steps = 100
            for cnt, (student_ids, question_ids, concepts_emb, labels) in enumerate(dataloader):
                student_ids = student_ids.to(device)
                question_ids = question_ids.to(device)
                labels = labels.to(device)
                concepts_emb = concepts_emb.to(device)
                pred = self.model(student_ids, question_ids, concepts_emb)
                bz_loss = self._loss_function(pred, labels)
                optimizer.zero_grad()
                bz_loss.backward()
                optimizer.step()
                self.model.apply_clipper()
                loss += bz_loss.data.float()
                # if cnt % log_steps == 0:
                    # print('Epoch [{}] Batch [{}]: loss={:.3f}'.format(ep, cnt, loss / cnt))
    
    def get_pred(self, user_ids, avail_questions, concept_map):
        device = self.device

        pred_all = {}
        with torch.no_grad():
            self.model.eval()
            for sid in user_ids:
                pred_all[sid] = {}
                question_ids =  list(avail_questions[sid])
                student_ids = [sid] * len(question_ids)
          
                concepts_embs = []
                for qid in question_ids:
                    concepts = concept_map[qid]
                    concepts_emb = [0.] * self.num_knowledges
                    for concept in concepts:
                        concepts_emb[concept] = 1.0
                    concepts_embs.append(concepts_emb)
                student_ids = torch.LongTensor(student_ids).to(device)
                question_ids = torch.LongTensor(question_ids).to(device)
                concepts_embs = torch.Tensor(concepts_embs).to(device)
                output = self.model(student_ids, question_ids, concepts_embs).view(-1).tolist()
                for i, qid in enumerate(list(avail_questions[sid])):
                    pred_all[sid][qid] = output[i]
            self.model.train()
        return pred_all

    def expected_model_change(self, sid: int, qid: int, pred_all: dict, concept_map):
        """ get expected model change
        Args:
            student_id: int, student id
            question_id: int, question id
        Returns:
            float, expected model change
        """
        # epochs = self.args.cdm_epoch
        epochs = 1
        # lr = self.args.cdm_lr
        lr = 0.01
        device = self.device
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        for name, param in self.model.named_parameters():
            if 'student' not in name:
                param.requires_grad = False

        original_weights = self.model.student_emb.weight.data.clone()

        student_id = torch.LongTensor([sid]).to(device)
        question_id = torch.LongTensor([qid]).to(device)
        concepts = concept_map[qid]
        concepts_emb = [0.] * self.num_knowledges
        for concept in concepts:
            concepts_emb[concept] = 1.0
        concepts_emb = torch.Tensor([concepts_emb]).to(device)
        correct = torch.LongTensor([1]).to(device)
        wrong = torch.LongTensor([0]).to(device)

        for ep in range(epochs):
            optimizer.zero_grad()
            pred = self.model(student_id, question_id, concepts_emb)
            loss = self._loss_function(pred, correct)
            loss.backward()
            optimizer.step()

        pos_weights = self.model.student_emb.weight.data.clone()
        self.model.student_emb.weight.data.copy_(original_weights)

        for ep in range(epochs):
            optimizer.zero_grad()
            pred = self.model(student_id, question_id, concepts_emb)
            loss = self._loss_function(pred, wrong)
            loss.backward()
            optimizer.step()

        neg_weights = self.model.student_emb.weight.data.clone()
        self.model.student_emb.weight.data.copy_(original_weights)

        for param in self.model.parameters():
            param.requires_grad = True

        # pred = self.model(student_id, question_id, concepts_emb).item()
        pred = pred_all[sid][qid]
        return pred * torch.norm(pos_weights - original_weights).item() + \
               (1 - pred) * torch.norm(neg_weights - original_weights).item()