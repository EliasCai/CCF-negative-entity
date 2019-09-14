# -*- coding: UTF-8 -*-

from tqdm import tqdm
import pandas as pd
import numpy as np
import re, os, codecs
from sklearn.model_selection import train_test_split

TRAIN_PATH = '../data/Train_Data.csv'
TEST_PATH = '../data/Test_Data.csv'
STOPWORD_PATH = '../input/'
TEST_SIZE = 0.2
        
        
        
def get_traindata():
    with codecs.open(TRAIN_PATH, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    lines = [l.strip().split(',') for l in lines][1:]
    # sents = [split_sent(l[2]) for l in lines]  # 分句

    match_sents = []
    origin_sents = []
    entity_list = []
    negative_list = []
    key_entity_list = []
    id_list = []
    text_list = []
    for line in lines:
        if len(line) > 6:
            line[2] = ';'.join(line[2:-3])
            line = line[:3] + line[-3:]
        id, title, text, entities, negative, key_entity = line
        entities = entities.split(';')
        key_entities = key_entity.split(';')
        id_list.append(id)
        entity_list.append(entities)
        negative_list.append(negative)
        key_entity_list.append(key_entities)
        text_list.append(text)
        
    return id_list, entity_list, key_entity_list, text_list
        
def get_testdata():
    with codecs.open(TEST_PATH, 'r',encoding='utf-8') as f:
        lines = f.readlines()

    lines = [l.strip().split(',') for l in lines][1:]
    # sents = [split_sent(l[2]) for l in lines]  # 分句

    match_sents = []
    origin_sents = []
    entity_list = []
    # negative_list = []
    # key_entity_list = []
    id_list = []
    text_list = []
    for line in lines:
        if len(line) > 4:
            line[2] = ';'.join(line[2:-1])
            line = line[:3] + line[-1:]
        id, title, text, entities = line
        entities = entities.split(';')
        id_list.append(id)
        entity_list.append(entities)
        text_list.append(text)

        
    return id_list, entity_list, text_list
        
        
def test01():

    tr_id, tr_entity, tr_key, tr_text = get_traindata()
    te_id, te_entity, te_text = get_testdata()
    
    tr_data, te_data = [], []
    for id, entity, key, text in zip(tr_id, tr_entity, tr_key, tr_text):
        for e in entity:    
            tr_data.append((id, e, int(e in key), text))
            
    for id, entity, text in zip(te_id, te_entity, te_text):
        for e in entity:    
            te_data.append((id, e, text))    
            
    tr_data = pd.DataFrame(tr_data, columns = ['id','entity','neg','text'])
    te_data = pd.DataFrame(te_data, columns = ['id','entity','text'])
    tr_entity_map = tr_data.groupby('entity').agg({'neg':'mean','id':'count'}).sort_values('id', ascending=False)
    
    # neg, cnt = tr_entity_map.loc['京东白条']
    return te_id, te_entity, te_text, tr_entity_map
    
    
def test02():
    te_id, te_entity, te_text, tr_entity_map = test01()
    with codecs.open('../output/result_0870.txt','r',encoding='utf-8') as f:
        pred_lines = f.readlines()
        
    pred_lines = test03(te_id, te_entity, pred_lines, te_text, tr_entity_map)
    
    pred_data = []
    for idx, (id, te_e, line) in enumerate(zip(te_id, te_entity, pred_lines[1:])):
         
        pred_id, pred_neg, pred_entity = line.strip().split(',')

        assert id == pred_id
        for e in te_e:    
            pred_data.append((id, e, int(e in pred_entity)))
            
    pred_data = pd.DataFrame(pred_data, columns = ['id','entity','neg'])
    pred_entity_map = pred_data.groupby('entity').agg({'neg':'mean','id':'count'}).sort_values('id', ascending=False)
    
    # pred_lines = test03(te_id, te_entity, pred_lines, te_text, pred_entity_map)
    with codecs.open('../output/result_0914.txt','w',encoding='utf-8') as f:
        for text in pred_lines:
            f.write(text)

            
def test03(te_id, te_entity, pred_lines, te_text, entity_map):
    cnt = 0
    for idx, (id, te_e, line, text) in enumerate(zip(te_id, te_entity, pred_lines[1:], te_text)):
         
        pred_id, pred_neg, pred_entity = line.strip().split(',')

        assert id == pred_id
        
        neg_entity, post_entity = [], []
        pred_e = pred_entity.split(';')
        for e in te_e:
            if e in entity_map.index:
                neg, n = entity_map.loc[e]
                if neg > 0.8 and n > 10 :
                    neg_entity.append(e)
                    if e not in pred_e:
                        # if e in text:
                            cnt += 1
                if neg < 0.1 and n > 10:
                    post_entity.append(e)
                    if e in pred_e:
                        cnt += 1
                        pass
                        
        if set(neg_entity) > set(pred_e):
            new_e = [e for e in te_e if e in (set(neg_entity) | set(pred_e))]
            new_e = ';'.join(new_e)
            pred_neg = '1'
            text = ','.join([pred_id,pred_neg,new_e]) + '\n'
            pred_lines[idx + 1] = text
            
        if len(post_entity)>0 and set(post_entity) < set(pred_e):
            new_e = [e for e in te_e if e in (set(pred_e) - set(post_entity))]
            
            if len(new_e) == 0:
                pass
            
            new_e = ';'.join(new_e)
            text = ','.join([pred_id,pred_neg,new_e]) + '\n'
            pred_lines[idx + 1] = text
        
    print(cnt)
    return pred_lines
            
if __name__ == "__main__":
    test02()
    

    