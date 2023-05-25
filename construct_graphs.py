import json
import numpy as np
from collections import Counter
import json

def dump_json(path_, data):
    with open(path_, 'w') as fh:
        json.dump(data, fh, indent=2)
    return data

def build_local_map(name):
    concept_file = f'data/concept_map_{name}.json'
    if name == 'assist2009':
        exer_n = 17751+1
    if name == 'junyi':
        exer_n = 2835+1
    if name == '3_4':
        exer_n = 948+1

    temp_list = []
    with open(concept_file, encoding='utf8') as f:
        concept_map = json.load(f)
    k_from_e = '' # e(src) to k(dst)
    e_from_k = '' # k(src) to k(dst)
    
    for qid in concept_map:
        # has id=0 question for pad
        exer_id = int(qid) + 1
        for k in concept_map[str(qid)]:
            if (str(exer_id) + '\t' + str(k + exer_n)) not in temp_list or (str(k + exer_n) + '\t' + str(exer_id)) not in temp_list:
                k_from_e += str(exer_id) + '\t' + str(k + exer_n) + '\n'
                e_from_k += str(k + exer_n) + '\t' + str(exer_id) + '\n'
                temp_list.append((str(exer_id) + '\t' + str(k + exer_n)))
                temp_list.append((str(k + exer_n) + '\t' + str(exer_id)))

    path = f'graph_data/{name}/'

    with open(path+'k_from_e.txt', 'w') as f:
        f.write(k_from_e)
    with open(path+'e_from_k.txt', 'w') as f:
        f.write(e_from_k)

def constructDependencyMatrix(name):
    data_file = f'data/train_task_{name}.json'
    concept_file = f'data/concept_map_{name}.json'
    if name == 'assist2009':
        knowledge_n = 123 # num of knowledge
    if name == 'junyi':
        knowledge_n = 40
    if name == '3_4':
        knowledge_n = 86

    edge_dic_deno = {}
    with open(data_file, encoding='utf8') as i_f:
        data = json.load(i_f)
    
    with open(concept_file, encoding='utf8') as f:
        concept_map = json.load(f)
    # Calculate correct matrix
    knowledgeCorrect = np.zeros([knowledge_n, knowledge_n])
    for student in data:
        if student['log_num'] < 2:
            continue
        q_ids, labels = student['q_ids'], student['labels']
        for log_i in range(student['log_num']-1):
            if labels[log_i] * labels[log_i+1] == 1:
                for ki in concept_map[str(q_ids[log_i])]:
                    for kj in concept_map[str(q_ids[log_i+1])]:
                        if ki != kj:
                            # n_{ij}
                            knowledgeCorrect[ki][kj] += 1.0
                            # n_{i*}, calculate the number of correctly answering i
                            if ki in edge_dic_deno.keys():
                                edge_dic_deno[ki] += 1
                            else:
                                edge_dic_deno[ki] = 1

    s = 0
    c = 0
    # Calculate transition matrix
    knowledgeDirected = np.zeros([knowledge_n, knowledge_n])
    for i in range(knowledge_n):
        for j in range(knowledge_n):
            if i != j and knowledgeCorrect[i][j] > 0:
                    knowledgeDirected[i][j] = float(knowledgeCorrect[i][j]) / edge_dic_deno[i]
                    s += knowledgeDirected[i][j]
                    c += 1
    o = np.zeros([knowledge_n, knowledge_n])
    min_c = 100000000
    max_c = 0
    for i in range(knowledge_n):
        for j in range(knowledge_n):
            if knowledgeCorrect[i][j] > 0 and i != j:
                min_c = min(min_c, knowledgeDirected[i][j])
                max_c = max(max_c, knowledgeDirected[i][j])
    s_o = 0
    l_o = 0
    for i in range(knowledge_n):
        for j in range(knowledge_n):
            if knowledgeCorrect[i][j] > 0 and i != j:
                o[i][j] = (knowledgeDirected[i][j] - min_c) / (max_c - min_c)
                l_o += 1
                s_o += o[i][j]
    
    # avg = 0.02
    if name == 'assist2009':
        avg = s_o / l_o #total / count
        avg *= avg
        avg *= avg
    elif name == '3_4':
        avg = s_o / l_o #total / count # 0.02
        # avg =0.02
    else:
        avg = s_o / l_o #total / count
        # avg *= avg
        # avg *= avg
        
    print(avg)
    # avg is threshold
    graph = ''
    # edge = np.zeros([knowledge_n, knowledge_n])
    for i in range(knowledge_n):
        for j in range(knowledge_n):
            if o[i][j] >= avg:
                graph += str(i) + '\t' + str(j) + '\n'
                # edge[i][j] = 1
    path = f'graph_data/{name}/'
    with open(path+'knowledgeGraph.txt', 'w') as f:
        f.write(graph)

def process_edge(name):
    K_Directed = ''
    K_Undirected = ''
    edge = []
    path = f'graph_data/{name}/'
    with open(path+'knowledgeGraph.txt', 'r') as f:
        for i in f.readlines():
            i = i.replace('\n', '').split('\t')
            src = i[0]
            tar = i[1]
            edge.append((src, tar))
    visit = []
    for e in edge:
        if e not in visit:
            if (e[1],e[0]) in edge:
                K_Undirected += str(e[0] + '\t' + e[1] + '\n')
                visit.append(e)
                visit.append((e[1],e[0]))
            else:
                K_Directed += str(e[0] + '\t' + e[1] + '\n')
                visit.append(e)

    with open(path+'K_Directed.txt', 'w') as f:
        f.write(K_Directed)
    with open(path+'K_Undirected.txt', 'w') as f:
        f.write(K_Undirected)
    all = len(visit)
    print(all)

def nov_reward(dataset):
    data_file = f'data/train_task_{dataset}.json'
    with open(data_file, encoding='utf8') as i_f:
        data = json.load(i_f)

    all_questions = []
    for student in data:
        q_ids = student['q_ids']
        all_questions.extend(q_ids)
    print(min(all_questions))
    # Get Novel Items
    all_pairs = Counter(all_questions).items()
    item_freqs = [pair[1] for pair in all_pairs]

    threshold = np.quantile(item_freqs, q=0.9)
    print(threshold)
 
    less_popular_items = []
    for pair in all_pairs:
        if pair[1] <= threshold:
            less_popular_items.append(pair[0])
    print('number of less popular items is: ', len(less_popular_items))

    # Binary Novelty Reward System
    binary_nov_reward= {}
    for pair in all_pairs:
        if pair[1] <= threshold:
            binary_nov_reward[str(pair[0])] = 1
        else:
            binary_nov_reward[str(pair[0])] = 0
    
    dump_json(f'data/nov_reward_{dataset}.json', binary_nov_reward)

if __name__ == '__main__':
    name = 'assist2009'
    build_local_map(name)
    constructDependencyMatrix(name)
    process_edge(name)
    nov_reward(name)