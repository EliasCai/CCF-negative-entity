import torch
import torch.nn as nn
from torch.optim import Adam
import pandas as pd
# from pyltp import SentenceSplitter
from pytorch_pretrained_bert import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from pytorch_pretrained_bert.modeling import BertModel

import numpy as np
import re, os, codecs
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

TRAIN_PATH = '../data/Train_Data.csv'
TEST_PATH = '../data/Test_Data.csv'
STOPWORD_PATH = '../input/'
TEST_SIZE = 0.2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# https://www.datafountain.cn/competitions/353/datasets

def clean_text(x):
    x = re.sub('\?{2,}', '', x)
    x = re.sub('\u3000', '', x)
    x = re.sub(' ', '', x)
    return x


def split_sent(x):
    return re.split('[,，.。！!；;]', x)


def find_corr_sentence():
    with codecs.open(TEST_PATH, 'r',encoding='utf-8') as f:
        lines = f.readlines()

    lines = [l.strip().split(',') for l in lines][1:]
    sents = [split_sent(l[2]) for l in lines]  # 分句

    match_sents = []
    origin_sents = []
    entity_list = []
    # negative_list = []
    # key_entity_list = []
    id_list = []
    for line in lines:
        if len(line) > 4:
            line[2] = ';'.join(line[2:-1])
            line = line[:3] + line[-1:]
        id, title, text, entities = line
        entities = entities.split(';')
        id_list.append(id)
        entity_list.append(entities)
        if text == title:
            text = clean_text(text)
            sents = split_sent(text)
            origin_sents.append(text)
        else:
            text = clean_text(text)
            title = clean_text(title)
            sents = split_sent(title + ';' + text)
            origin_sents.append(title + ';' + text)
        match_sent = []
        # origin_sents.append(title+';' + text)
        for sent in sents:
            for entity in entities:
                if entity in sent:
                    match_sent.append(sent)
                    break

        match_sents.append(';'.join(match_sent))

    print(pd.Series([len(l) for l in match_sents]).describe())
    print(pd.Series([len(l) for l in origin_sents]).describe())

    return (
        id_list,
        origin_sents,
        match_sents,
        entity_list
    )


def generate_bert_input(tokenizer, seq_id, seq_text, seq_entity, maxlen=300):

    seq_entity = [e[:3] for e in seq_entity]
    seq_token, seq_mask, seq_segment = [], [], []
    for st, se in zip(seq_text, seq_entity):

        # cls = np.zeros((3,), dtype=np.int32)
        len_entity = len(se)
        t = st[: maxlen - len_entity]
        input_t = ['[CLS]'] + tokenizer.tokenize(t)
        segment = [0] * len(input_t)
        for idx, e in enumerate(se):
            e_tok = tokenizer.tokenize(e)
            input_t += ['[SEP]'] + e_tok
            segment += [0] + [1] * len(e_tok)

        # seq_cls.append(cls.reshape((1, -1)))
        input_t += ['[SEP]']
        segment += [1]
        seq_token.append(tokenizer.convert_tokens_to_ids(input_t))
        seq_mask.append(np.ones((len(input_t),), dtype=np.int32))
        seq_segment.append(segment)

    # seq_cls = np.vstack(seq_cls)
    seq_token = pad_sequences(
        seq_token, maxlen=maxlen, padding='post', truncating='post'
    )
    seq_mask = pad_sequences(seq_mask, maxlen=maxlen, padding='post', truncating='post')
    seq_segment = pad_sequences(
        seq_segment, maxlen=maxlen, padding='post', truncating='post'
    )
    return seq_token, seq_mask, seq_segment


def data_to_loader(tokenizer, bs=32):

    seq_id, _, seq_text, seq_entity = find_corr_sentence()
    seq_token, seq_mask, seq_segment = generate_bert_input(tokenizer, seq_id, seq_text, seq_entity)

    # te_id = torch.LongTensor(seq_id)
    te_token = torch.LongTensor(seq_token)
    te_mask = torch.LongTensor(seq_mask)
    te_segment = torch.LongTensor(seq_segment)

    valid_data = TensorDataset(te_token, te_mask, te_segment)
    valid_sampler = SequentialSampler(valid_data)
    valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=bs)

    return valid_dataloader,seq_id, seq_entity
        
def test_epoch(model, data_loader):
    # max_grad_norm = 1.0
    # optimizer = get_opt(model, True)
    # criterion = nn.CrossEntropyLoss(ignore_index=0)
    criterion = nn.CrossEntropyLoss()
    
     # TRAIN loop
    model.eval()
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    pbar = tqdm(data_loader)  
    preds, labels = [], []
    for step, batch in enumerate(pbar):
        # add batch to gpu

        batch = tuple(t.to(device) for t in batch)
        b_token, b_mask, b_segment = batch
        with torch.no_grad():
            
            logits = model(b_token, b_mask, b_segment)

        # track train loss
        desc = 'testing'
        pbar.set_description(desc)
        
        p = logits.detach().cpu().numpy()
        p = np.argmax(p, axis=2)
        preds.append(p)
    return preds
    
    
class TokenNet(nn.Module):
    def __init__(self, bert_base, finetuning=False):
        super().__init__()

        self.bert = bert_base

        self.dropout = nn.Dropout(0.3)

        self.fc = nn.Linear(768, 3)

        self.finetuning = finetuning

    def forward(self, input_ids, segment_ids, input_mask):

        if self.training and self.finetuning:
            # print("->bert.train()")
            self.bert.train()
            # encoded_layers, _ = self.bert(x)
            # model_bert(all_input_ids, all_segment_ids, all_input_mask)
            encoded_layers, _ = self.bert(input_ids, segment_ids, input_mask)
            enc = encoded_layers[-1]
        else:
            self.bert.eval()
            with torch.no_grad():
                # encoded_layers, _ = self.bert(x)
                encoded_layers, _ = self.bert(input_ids, segment_ids, input_mask)
                enc = encoded_layers[-1]

        enc = self.dropout(enc)

        logits = self.fc(enc)
        return logits[:,:3,:]


        
def generate_output(seq_id, seq_entity, preds):
    # id,negative,key_entity
    # 2,1,草根投资;石头理财
    with codecs.open('../output/result_0910.txt','w',encoding='utf-8') as f:
        f.write('id,negative,key_entity\n')
        for id, entity, pred in tqdm(zip(seq_id, seq_entity, preds)):
            negative = '1' if max(pred) ==2 else '0'
            key_entity = ';'.join([e for e,p in zip(entity, pred) if p==2])
            text = ','.join([id,negative,key_entity])
            f.write(text + '\n')
        
if __name__ == "__main__":
    
    BERT_PT_PATH = "../checkpoint/chinese_L-12_H-768_A-12/"
    tokenizer = BertTokenizer.from_pretrained(BERT_PT_PATH, do_lower_case=True)
    bert_base = BertModel.from_pretrained(BERT_PT_PATH)
    
    test_loader, seq_id, seq_entity = data_to_loader(tokenizer)
    model = TokenNet(bert_base, True)
    
    path_model_bert = os.path.join("../checkpoint", "bert_best.pt")
    if os.path.exists(path_model_bert):
        res = torch.load(path_model_bert)
    bert_base.load_state_dict(res["bert_base"])

    path_model_view = os.path.join("../checkpoint", "model_best.pt")
    if os.path.exists(path_model_view):
        res = torch.load(path_model_view)
    model.load_state_dict(res["model"])
    model.to(device)
    preds = test_epoch(model, test_loader)
    preds = np.vstack(preds)
    generate_output(seq_id, seq_entity, preds)
    
    
    
