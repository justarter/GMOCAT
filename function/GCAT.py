import math
import copy
import time
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.distributions import Categorical

from util import tensor_to_numpy

class ActorCritic(nn.Module):
    def __init__(self,  args, local_map):
        self.device = torch.device('cuda')
        self.args = args
        self.know_num = args.know_num
        self.item_num = args.item_num
        # self.emb_dim = args.know_num
        self.emb_dim = args.emb_dim
        
        self.directed_g = local_map['directed_g'].to(self.device)
        self.k_from_e = local_map['k_from_e'].to(self.device)
        self.e_from_k = local_map['e_from_k'].to(self.device)

        self.morl_weights = args.morl_weights
        super(ActorCritic, self).__init__()

        # network structure
        self.knowledge_emb = nn.Embedding(self.know_num, self.emb_dim)
        # idx=0 pad
        self.exercise_emb = nn.Embedding(self.item_num, self.emb_dim)
        self.ans_embed = nn.Embedding(3, self.emb_dim)
        if self.args.use_graph:
            # all index
            self.k_index = torch.LongTensor(list(range(self.know_num))).to(self.device)
            self.exer_index = torch.LongTensor(list(range(self.item_num))).to(self.device)
            fusion = Fusion(args, local_map)
            self.graphencoder = GraphEncoder(copy.deepcopy(fusion), args.graph_block)

        # self attention
        hidden_input_size = 3 * self.emb_dim

        if self.args.use_attention:
            c = copy.deepcopy
            attn = MultiHeadedAttention(args.n_head, hidden_input_size)
            ff = PositionwiseFeedForward(hidden_input_size, 2048, dropout=args.dropout_rate)
            self.encoder = Encoder(EncoderLayer(hidden_input_size, c(attn), c(ff), dropout=args.dropout_rate), args.n_block)
        
        # actor layer
        self.actor_layer = nn.Sequential(
            nn.Linear(hidden_input_size, args.latent_factor),
            nn.Tanh(),
            nn.Linear(args.latent_factor, self.item_num))

        # rl head
        # Accuracy RL Head
        self.acc_head = nn.Sequential(
            nn.Linear(hidden_input_size, args.latent_factor),
            nn.Tanh(),
            nn.Linear(args.latent_factor, 1))

        # Diversity RL Head
        self.div_head = nn.Sequential(
            nn.Linear(hidden_input_size, args.latent_factor),
            nn.Tanh(),
            nn.Linear(args.latent_factor, 1))

        # Novelty RL Head
        self.nov_head = nn.Sequential(
            nn.Linear(hidden_input_size, args.latent_factor),
            nn.Tanh(),
            nn.Linear(args.latent_factor, 1))

        # initialization
        for name, param in self.named_parameters():
            if param.dim() > 1:
                nn.init.xavier_normal_(param)

    def hidden_layer(self, p_rec, p_target, a_rec, kn_rec, kn_num):# kn_rec (B,L,num_k)
        if self.args.use_graph:
            exer_emb = self.exercise_emb(self.exer_index).to(self.device)
            kn_emb = self.knowledge_emb(self.k_index).to(self.device)
            # Fusion layer
            kn_emb2, exer_emb2 = self.graphencoder(kn_emb, exer_emb)
            # get batch exercise data
            batch_exer_emb = exer_emb2[p_rec]  # B L D
            batch_ans_emb = self.ans_embed(a_rec) # B L D
            batch_kn_emb = torch.matmul(kn_rec, kn_emb2) / kn_num.unsqueeze(-1) # B L D
        else:
            # get batch exercise data
            batch_exer_emb = self.exercise_emb(p_rec).to(self.device)  # B L D
            batch_ans_emb = self.ans_embed(a_rec) # B L D
            batch_kn_emb = torch.matmul(kn_rec, self.knowledge_emb.weight) / kn_num.unsqueeze(-1) # B L D
        # concat
        item_emb = torch.cat([batch_exer_emb, batch_ans_emb, batch_kn_emb], dim=-1)# B L 3D
        
        if self.args.use_attention:
            # atten
            src_mask = mask(p_rec, p_target + 1).unsqueeze(-2)# B 1 L
            atten_hidden = self.encoder(item_emb, src_mask) # B L 3D
        else:
            atten_hidden = item_emb # no atten

        att_mask = mask(p_rec, p_target + 1) # B L
        atten_hidden = atten_hidden.masked_fill(att_mask.unsqueeze(-1)==0,0)
        hidden_state = torch.sum(atten_hidden,dim=1) / torch.sum(att_mask,dim=-1,keepdim=True)
        
        # hidden_state = torch.sum(atten_hidden,dim=1)
        
        return hidden_state

    def predict(self, data):
        self.eval()
        with torch.no_grad():
            p_rec, p_target, a_rec, kn_rec, kn_num = data['p_rec'], \
                     data['p_t'], data['a_rec'], data['kn_rec'], data['kn_num']
            p_rec, p_target, a_rec, kn_rec, kn_num = \
                                    torch.LongTensor(p_rec).to(self.device), \
                                    torch.LongTensor(p_target).to(self.device), \
                                    torch.LongTensor(a_rec).to(self.device), \
                                    kn_rec.to(self.device), kn_num.to(self.device)
            hidden_state = self.hidden_layer(p_rec, p_target, a_rec, kn_rec, kn_num)
            logits = self.actor_layer(hidden_state).detach().cpu()
        # self.train()

        return logits

    def forward(self):
        raise NotImplementedError

    def evaluate(self, p_rec, p_target, a_rec, kn_rec, kn_num, action, action_mask): # logprob, value, entropy
        hidden_state = self.hidden_layer(p_rec, p_target, a_rec, kn_rec, kn_num)
        logits = self.actor_layer(hidden_state)
        inf_mask = torch.clamp(torch.log(action_mask.float()),
                               min=torch.finfo(torch.float32).min)
        logits = logits + inf_mask
        action_probs = F.softmax(logits, dim=-1)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        acc_v = self.acc_head(hidden_state) # B 1
        div_v = self.div_head(hidden_state)
        nov_v = self.nov_head(hidden_state)
        morl_v = torch.cat([acc_v, div_v, nov_v], dim=1)# B 3
        return action_logprobs, morl_v, dist_entropy

class GCAT:
    def __init__(self, args, local_map):
        self.args = args
        self.lr = args.learning_rate
        self.betas = (0.9,0.999)
        self.morl_weights = args.morl_weights

        self.device = torch.device('cuda')
        self.eps_clip = 0.2

        self.policy = ActorCritic(args, local_map).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr, betas=self.betas)
        self.policy_old = ActorCritic(args, local_map).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()

    def update(self,p_rec, p_target, a_rec, kn_rec, kn_num, 
                    b_old_actions, b_old_logprobs, b_old_actionmask, b_rewards):

        logprobs, state_values, dist_entropy = self.policy.evaluate(
            p_rec, p_target, a_rec, kn_rec, kn_num, b_old_actions, b_old_actionmask)

        # Finding the ratio (pi_theta / pi_theta__old):
        ratios = torch.exp(logprobs - b_old_logprobs.detach())

        # Finding Surrogate Loss:
        advantages = b_rewards - state_values.detach() # B 3
        advantages =  advantages[:, 0] * self.morl_weights[0] + \
                    advantages[:, 1] * self.morl_weights[1] + \
                    advantages[:, 2] * self.morl_weights[2]# B
        advantages = torch.squeeze(advantages)
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()- 0.01*dist_entropy.mean()
        # actor_loss = -torch.min(surr1, surr2).mean()
        
        critic_loss = torch.square(state_values-b_rewards) * 0.5 # B 3
        critic_loss =  critic_loss[:, 0] * self.morl_weights[0] + \
                     critic_loss[:, 1] * self.morl_weights[1] + \
                     critic_loss[:, 2] * self.morl_weights[2]
        critic_loss = critic_loss.mean()
        loss = actor_loss + critic_loss
        # print(actor_loss, critic_loss)
        # take gradient step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Copy new weights into old policy:
        # self.policy_old.load_state_dict(self.policy.state_dict())
        return tensor_to_numpy(loss)

    def optimize_model(self, data, b_old_actions, b_old_logprobs,b_old_actionmask, b_rewards):
        self.policy.train()
        p_rec, p_target, a_rec, kn_rec, kn_num = \
            data['p_rec'], data['p_t'], data['a_rec'], data['kn_rec'], data['kn_num']
        p_rec, p_target, a_rec, kn_rec, kn_num = \
                            torch.LongTensor(p_rec).to(self.device), torch.LongTensor(p_target).to(self.device), \
                            torch.LongTensor(a_rec).to(self.device), kn_rec.to(self.device), kn_num.to(self.device)
        loss = self.update(p_rec, p_target, a_rec, kn_rec, kn_num, 
                    b_old_actions, b_old_logprobs,b_old_actionmask, b_rewards)
        return loss

    # def transfer_weights(self):
    #     W = self.policy.get_weights()
    #     tgt_W = self.target_policy.get_weights()
    #     for i in range(len(W)):
    #         tgt_W[i] = self.tau * W[i] + (1 - self.tau) * tgt_W[i]
    #     self.target_policy.set_weights(tgt_W)

    def transfer_weights(self):
        self.policy_old.load_state_dict(self.policy.state_dict())

    @classmethod
    def create_model(cls, config, local_map):
        model = cls(config, local_map)
        # device = torch.device('cuda')
        return model

# GAT layer
class GraphLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim):
        super(GraphLayer, self).__init__()
        self.g = g
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)} 

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, h):
        z = self.fc(h)
        self.g.ndata['z'] = z
        self.g.apply_edges(self.edge_attention)
        self.g.update_all(self.message_func, self.reduce_func)
        return self.g.ndata.pop('h')

class Fusion(nn.Module):
    def __init__(self, args, local_map):
        self.device = torch.device('cuda')
        self.know_num = args.know_num
        self.item_num = args.item_num
        self.emb_dim = args.emb_dim

        # graph structure
        self.directed_g = local_map['directed_g'].to(self.device)
        self.k_from_e = local_map['k_from_e'].to(self.device)
        self.e_from_k = local_map['e_from_k'].to(self.device)

        super(Fusion, self).__init__()

        # self.emb_dim = args.know_num
        self.directed_gat = GraphLayer(self.directed_g, self.emb_dim, self.emb_dim) # in_dim out_dim
        
        self.k_from_e = GraphLayer(self.k_from_e, self.emb_dim, self.emb_dim)  # src: e
        self.e_from_k = GraphLayer(self.e_from_k, self.emb_dim, self.emb_dim)  # src: k

        self.relation_attn = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim),
            nn.Tanh(),
            nn.Linear(self.emb_dim, 1, bias=False),
        )


    def forward(self, kn_emb, exer_emb):
        # aggregrated emb
        k_directed = self.directed_gat(kn_emb)

        e_k_graph = torch.cat((exer_emb, kn_emb), dim=0)
        k_from_e_graph = self.k_from_e(e_k_graph) # e2k
        e_from_k_graph = self.e_from_k(e_k_graph) # k2e

        # update concepts
        A = k_directed
        B = k_from_e_graph[self.item_num:]

        score1 = self.relation_attn(A)
        score2 = self.relation_attn(B)

        score = F.softmax(torch.cat([score1, score2], dim=1), dim=1) 
        kn_emb = score[:, 0].unsqueeze(1) * A + score[:, 1].unsqueeze(1) * B

        # updated exercises 
        exer_emb = e_from_k_graph[0: self.item_num] # N D

        return kn_emb, exer_emb


def mask(src, s_len): # set useful as 1
    if type(src) == torch.Tensor:
        mask = torch.zeros_like(src)
        for i in range(len(src)):
            mask[i, :s_len[i]] = 1
    return mask # B max_len

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class GraphEncoder(nn.Module):
    def __init__(self, layer, N):
        super(GraphEncoder, self).__init__()
        self.layers = clones(layer, N)
        
    def forward(self, kn_emb, exer_emb):
        for layer in self.layers:
            kn_emb, exer_emb = layer(kn_emb, exer_emb)
        return kn_emb, exer_emb

class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    
    if mask is not None: # B 1 1 h_k
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))