import os
import numpy as np
import pandas as pd
import numpy as np
import json
import time
from multiprocessing import Pool
import argparse
from collections import defaultdict

question_map = {}
question_meta, subject_metadata, df, question_meta_1, question_meta_3, = {}, {}, {}, {}, {}

def open_json(path_):
    with open(path_) as fh:
        data = json.load(fh)
    return data


def dump_json(path_, data):
    with open(path_, 'w') as fh:
        json.dump(data, fh, indent=2)
    return data

def f_eedi(user_id):
    global df, question_map
    user_df = df[df.UserId == user_id].sort_values('DateAnswered')
    q_ids, labels = [], [] 
    q_ids_set = set()
    for _, row in user_df.iterrows():
        q_id = str(row['QuestionId'])
        if q_id in q_ids_set:
            continue
        q_ids_set.add(q_id)
        q_ids.append(question_map[q_id])
        ans = 1 if row['IsCorrect'] else 0
        labels.append(ans)
    out = {'user_id': int(user_id), 'q_ids': q_ids, 'labels': labels, 'log_num':len(labels)}
    return out

def featurize_eedi(dataset):
    global question_map, df
    TRAIN_DATA = 'raw_data/train_task_'+dataset+'.csv'
    ANSWER_DATA = 'raw_data/answer_metadata_task_'+dataset+'.csv'

    # AnswerId,DateAnswered,Confidence,GroupId,QuizId,SchemeOfWorkId
    answer_df = pd.read_csv(ANSWER_DATA)[
        ['AnswerId', 'DateAnswered']]
    answer_df['DateAnswered'] = pd.to_datetime(
        answer_df['DateAnswered'], errors='coerce')
    print(answer_df.shape)

    # QuestionId,UserId,AnswerId,IsCorrect,CorrectAnswer,AnswerValue
    train_df = pd.read_csv(TRAIN_DATA)[
        ['QuestionId','UserId','AnswerId','IsCorrect']]
    print('train_df shape: ', train_df.shape)
    # print(train_df.isnull().values.any())

    # get answer id info for train
    train_merged_df = pd.merge(train_df, answer_df, on='AnswerId')
    print(train_merged_df.shape)
    print(train_merged_df.isnull().values.any())

    df = train_merged_df
    print(df.dtypes) #int64, int64,int64,int64
    
    user_ids = df['UserId'].unique()
    problems = df['QuestionId'].unique()
   
    for p in problems:
        question_map[str(p)] = len(question_map)
    with Pool(30) as p:
        results = p.map(f_eedi, user_ids)

    q2k = defaultdict(list)
    k2n = {}
    with open('raw_data/question_metadata_task_'+dataset+'.csv', 'r') as fp:
        lines = fp.readlines()[1:]
        for line in lines:
            line = line.strip('\n')
            words = line.split(',')
            q_id = int(words[0])
            if str(q_id) in question_map:
                qq = question_map[str(q_id)]

                subjects = eval(eval(','.join(words[1:])))
                for kk in subjects:
                    if kk not in k2n:
                        k2n[kk] = len(k2n)
                    q2k[qq].append(k2n[kk]) 
    
    bad_interactions = [len(d['q_ids']) for d in results if len(d['q_ids']) < 40]
    results = [d for d in results if len(d['q_ids']) >= 40]
    interactions = [len(d['q_ids']) for d in results]
    
    print('Number of eedi User: ', len(results))
    print('Number of eedi Interactions: ', sum(interactions))
    print('Ignored eedi Interactions: ', sum(bad_interactions))
    print('Number of Problems eedi:', len(question_map))
    print('Number of Knowledge eedi:', len(k2n))

    dump_json(f'data/train_task_{dataset}.json', results)
    dump_json(f'data/question_map_{dataset}.json', question_map)
    dump_json(f'data/concept_map_{dataset}.json', q2k)


def f_junyi(user_id):
    global df, question_map
    user_df = df[df.user_id == user_id].sort_values('Time')
    q_ids, labels = [], []
    q_ids_set = set()
    for _, row in user_df.iterrows():
        q_id = str(row['exercise'])
        if q_id in q_ids_set:
            continue
        if row['correct']=='CORRECT':
            ans = 1 
        elif row['correct']=='INCORRECT':
            ans = 0
        else:
            continue
        q_ids_set.add(q_id)
        q_ids.append(question_map[q_id])
        labels.append(ans)
    out = {'user_id': int(user_id), 'q_ids': q_ids, 'labels': labels, 'log_num':len(labels)}
    return out

def featurize_junyi():
    global question_map, df
    df = pd.read_csv('raw_data/junyi_ProblemLog_for_PSLC.txt',delimiter='\t',
        usecols=['Anon Student Id', 'Time', 'Outcome', 'Problem Name', 'KC (Topic)']).dropna()
    print(df.dtypes) # 
    df = df.rename(columns={
                            'Anon Student Id': 'user_id',#247547
                            'Problem Name': 'exercise',# 2835
                            'Outcome': 'correct', # 3
                            # 'KC (Exercise)':'skill1', # 722
                            'KC (Topic)' : 'skill', # 40
                            # 'KC (Area)': 'skill3' # 9
                            })

    user_ids = df['user_id'].unique()
    problems = df['exercise'].unique()
    for p in problems:
        question_map[str(p)] = len(question_map)
   
    with Pool(30) as p:
        results = p.map(f_junyi, user_ids)

    bad_interactions = [len(d['q_ids']) for d in results if len(d['q_ids']) < 50]
    results = [d for d in results if len(d['q_ids']) >= 50]
    interactions = [len(d['q_ids']) for d in results]

    table = df.loc[:, ['exercise', 'skill']].drop_duplicates()
    q2k = {}
    k2n = {}
    for _,row in table.iterrows():
        qid = str(question_map[str(row['exercise'])])
        kid = str([row['skill']])
        if kid not in k2n:
            k2n[kid] = len(k2n)
        q2k[qid] = [k2n[kid]]

    print('Number of Junyi User: ', len(results))
    print('Number of Junyi Interactions: ', sum(interactions))
    print('Ignored Junyi Interactions: ', sum(bad_interactions))
    print('Number of Problems Junyi:', len(question_map))
    print('Number of Knowledge Assist:', len(k2n))

    dump_json('data/train_task_junyi.json', results)
    dump_json('data/question_map_junyi.json', question_map) 
    dump_json('data/concept_map_junyi.json', q2k)



def f_assist2009(uuid):
    global df, question_map
    user_df = df[df.user_id == uuid].sort_values('order_id')
    q_ids, labels = [], []
    q_ids_set = set()
    for _, row in user_df.iterrows():
        q_id = str(row['problem_id'])
        if q_id in q_ids_set:
            continue
        q_ids_set.add(q_id)
        q_ids.append(question_map[q_id])
        ans = 1 if row['correct'] else 0
        labels.append(ans)
    out = {'user_id': int(uuid), 'q_ids': q_ids, 'labels': labels, 'log_num':len(labels)}
    return out

def featurize_assist2009():
    global question_map, df
    df = pd.read_csv('raw_data/assist09.csv', encoding = 'ISO-8859-1', dtype={'skill_id': str}, low_memory=False,
                     usecols=['order_id', 'user_id', 'problem_id', 'skill_id', 'correct']).dropna()
    print(df.dtypes)# int int int str int

    user_ids = df['user_id'].unique()
    problems = df['problem_id'].unique()
   
    for p in problems:
        question_map[str(p)] = len(question_map)
    with Pool(30) as p:
        results = p.map(f_assist2009, user_ids)
    q2k = defaultdict(list)
    k2n = {}
    table = df.loc[:, ['problem_id', 'skill_id']].drop_duplicates()
    for i, row in table.iterrows():
        qq = question_map[str(row['problem_id'])]
        for kk in list(set(map(int, str(row['skill_id']).split('_')))):
            if kk not in k2n:
                k2n[kk] = len(k2n)
            q2k[qq].append(k2n[kk]) 
    
    bad_interactions = [len(d['q_ids']) for d in results if len(d['q_ids']) < 40]
    results = [d for d in results if len(d['q_ids']) >= 40]
    interactions = [len(d['q_ids']) for d in results]

    print('Number of Assist User: ', len(results))
    print('Number of Assist Interactions: ', sum(interactions))
    print('Ignored Assist Interactions: ', sum(bad_interactions))
    print('Number of Problems Assist:', len(question_map))
    print('Number of Knowledge Assist:', len(k2n))

    dump_json('data/train_task_assist2009.json', results)
    dump_json('data/question_map_assist2009.json', question_map)
    dump_json('data/concept_map_assist2009.json', q2k)
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ML')
    parser.add_argument('--dataset', type=str,
                        default='assist2009', help='dataset name')
    params = parser.parse_args()
    if params.dataset == 'junyi':
        featurize_junyi()
    if params.dataset == 'assist2009':
        featurize_assist2009()
    if params.dataset == 'eedi':
        featurize_eedi(dataset='3_4')
