# -*- coding: UTF-8 -*-
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

TRAIN_PATH = "../data/Train_Data.csv"
TEST_PATH = "../data/Test_Data.csv"
STOPWORD_PATH = "../input/"
TEST_SIZE = 0.2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# https://www.datafountain.cn/competitions/353/datasets


def clean_text(x):
    x = re.sub("\?{2,}", "", x)
    x = re.sub("\u3000", "", x)
    x = re.sub(" ", "", x)
    return x


def split_sent(x):
    return re.split("[,，.。！!；;#]", x)


def find_corr_sentence():
    with codecs.open(TRAIN_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()

    lines = [l.strip().split(",") for l in lines][1:]
    sents = [split_sent(l[2]) for l in lines]  # 分句

    match_sents = []
    origin_sents = []
    entity_list = []
    negative_list = []
    key_entity_list = []
    id_list = []
    for line in lines:
        if len(line) > 6:
            line[2] = ";".join(line[2:-3])
            line = line[:3] + line[-3:]
        id, title, text, entities, negative, key_entity = line
        entities = entities.split(";")
        key_entities = key_entity.split(";")
        id_list.append(id)
        entity_list.append(entities)
        negative_list.append(negative)
        key_entity_list.append(key_entities)
        if text == title:
            text = clean_text(text)
            sents = split_sent(text)
            origin_sents.append(text)
        else:
            text = clean_text(text)
            title = clean_text(title)
            sents = split_sent(title + ";" + text)
            origin_sents.append(title + ";" + text)
        match_sent = []
        # origin_sents.append(title+';' + text)
        for sent in sents:
            for entity in entities:
                if entity in sent:
                    match_sent.append(sent)
                    break

        match_sents.append(";".join(match_sent))

    # print(pd.Series([len(l) for l in match_sents]).describe())
    # print(pd.Series([len(l) for l in origin_sents]).describe())

    return (
        id_list,
        origin_sents,
        match_sents,
        entity_list,
        negative_list,
        key_entity_list,
    )


def generate_bert_input(
    tokenizer, seq_id, seq_text, seq_entity, seq_negative, seq_key, maxlen=300
):

    seq_entity = [e[:3] for e in seq_entity]
    seq_token, seq_mask, seq_segment, seq_cls = [], [], [], []
    for st, sk, se in zip(seq_text, seq_key, seq_entity):

        cls = np.zeros((3,), dtype=np.int32)
        len_entity = len(se)
        t = st[: maxlen - len_entity]
        input_t = ["[CLS]"] + tokenizer.tokenize(t)
        segment = [0] * len(input_t)
        for idx, e in enumerate(se):
            if e in sk:
                cls[idx] = 1
            else:
                cls[idx] = 0
            e_tok = tokenizer.tokenize(e)
            input_t += ["[SEP]"] + e_tok
            segment += [0] + [1] * len(e_tok)

        seq_cls.append(cls.reshape((1, -1)))
        input_t += ["[SEP]"]
        segment += [1]
        seq_token.append(tokenizer.convert_tokens_to_ids(input_t))
        seq_mask.append(np.ones((len(input_t),), dtype=np.int32))
        seq_segment.append(segment)

    seq_cls = np.vstack(seq_cls)
    seq_token = pad_sequences(
        seq_token, maxlen=maxlen, padding="post", truncating="post"
    )
    seq_mask = pad_sequences(seq_mask, maxlen=maxlen, padding="post", truncating="post")
    seq_segment = pad_sequences(
        seq_segment, maxlen=maxlen, padding="post", truncating="post"
    )
    return seq_token, seq_mask, seq_segment, seq_cls


def data_to_loader(tokenizer, bs=32):

    seq_id, _, seq_text, seq_entity, seq_negative, seq_key = find_corr_sentence()
    seq_token, seq_mask, seq_segment, seq_cls = generate_bert_input(
        tokenizer, seq_id, seq_text, seq_entity, seq_negative, seq_key
    )

    idx = np.random.permutation(range(seq_token.shape[0]))
    tr_idx, te_idx = train_test_split(idx, test_size=TEST_SIZE)

    tr_token, tr_mask, tr_segment, tr_labels = (
        seq_token[tr_idx],
        seq_mask[tr_idx],
        seq_segment[tr_idx],
        seq_cls[tr_idx],
    )

    te_token, te_mask, te_segment, te_labels = (
        seq_token[te_idx],
        seq_mask[te_idx],
        seq_segment[te_idx],
        seq_cls[te_idx],
    )

    tr_token = torch.LongTensor(tr_token)
    tr_mask = torch.LongTensor(tr_mask)
    tr_segment = torch.LongTensor(tr_segment)
    tr_labels = torch.LongTensor(tr_labels)

    te_token = torch.LongTensor(te_token)
    te_mask = torch.LongTensor(te_mask)
    te_segment = torch.LongTensor(te_segment)
    te_labels = torch.LongTensor(te_labels)

    train_data = TensorDataset(tr_token, tr_mask, tr_segment, tr_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=bs)

    valid_data = TensorDataset(te_token, te_mask, te_segment, te_labels)
    valid_sampler = SequentialSampler(valid_data)
    valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=bs)

    return train_dataloader, valid_dataloader


def get_opt(model, finetune=True):

    if finetune:
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "gamma", "beta"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay_rate": 0.01,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay_rate": 0.0,
            },
        ]
    else:
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
    optimizer = Adam(optimizer_grouped_parameters, lr=3e-5)
    # optimizer = Adam(model.parameters(), lr=0.0001)
    return optimizer


def train_epoch(model, data_loader):
    max_grad_norm = 1.0
    optimizer = get_opt(model, True)
    # criterion = nn.CrossEntropyLoss(ignore_index=0)
    criterion = nn.CrossEntropyLoss()

    # TRAIN loop
    model.train()
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    pbar = tqdm(data_loader)
    for step, batch in enumerate(pbar):
        # add batch to gpu

        optimizer.zero_grad()
        batch = tuple(t.to(device) for t in batch)
        b_token, b_mask, b_segment, b_labels = batch

        logits = model(b_token, b_mask, b_segment)

        loss = criterion(logits.reshape(-1, logits.shape[-1]), b_labels.view(-1))

        loss.backward()
        # track train loss
        tr_loss += loss.item()
        nb_tr_examples += b_token.size(0)
        nb_tr_steps += 1
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(
            parameters=model.parameters(), max_norm=max_grad_norm
        )
        # update parameters
        optimizer.step()

        desc = "train loss-%.3f" % (tr_loss / nb_tr_steps)
        pbar.set_description(desc)


def eval_epoch(model, data_loader, tokenizer):

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
        b_token, b_mask, b_segment, b_labels = batch
        with torch.no_grad():

            logits = model(b_token, b_mask, b_segment)

        loss = criterion(logits.reshape(-1, logits.shape[-1]), b_labels.view(-1))

        # track train loss
        tr_loss += loss.item()
        nb_tr_examples += b_token.size(0)
        nb_tr_steps += 1

        desc = "eval loss-%.3f" % (tr_loss / nb_tr_steps)
        pbar.set_description(desc)

        p = logits.detach().cpu().numpy()
        p = np.argmax(p, axis=2)
        preds.append(p)
        labels.append(b_labels.detach().cpu().numpy())
    F1, F2, score = cal_metrics(preds, labels)
    return F1, F2, score


def cal_metrics(preds, labels):

    TP1, FN1, FP1 = 0, 0, 0
    TP2, FN2, FP2 = 0, 0, 0
    for pred, label in zip(preds, labels):
        idx1 = np.arange(pred.shape[0])
        idx2 = np.arange(pred.shape[0] * pred.shape[1])

        true_neg = idx1[label.max(1) == 1]
        pred_neg = idx1[pred.max(1) == 1]
        TP1 += len(set(true_neg) & set(pred_neg))
        FN1 += len(set(true_neg) - set(pred_neg))
        FP1 += len(set(pred_neg) - set(true_neg))

        true_key = idx2[label.reshape(-1) == 1]
        pred_key = idx2[pred.reshape(-1) == 1]
        TP2 += len(set(true_key) & set(pred_key))
        FN2 += len(set(true_key) - set(pred_key))
        FP2 += len(set(pred_key) - set(true_key))
    P2 = TP2 / (TP2 + FP2)
    R2 = TP2 / (TP2 + FN2)
    F2 = 2 * P2 * R2 / (P2 + R2)

    P1 = TP1 / (TP1 + FP1)
    R1 = TP1 / (TP1 + FN1)
    F1 = 2 * P1 * R1 / (P1 + R1)

    # print(F1, F2, 0.4*F1+0.6*F2)

    return F1, F2, 0.4 * F1 + 0.6 * F2


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
        return logits[:, :3, :]


if __name__ == "__main__":
    BERT_PT_PATH = "../checkpoint/chinese_L-12_H-768_A-12/"
    tokenizer = BertTokenizer.from_pretrained(BERT_PT_PATH, do_lower_case=True)
    bert_base = BertModel.from_pretrained(BERT_PT_PATH)

    train_loader, valid_loader = data_to_loader(tokenizer)
    model = TokenNet(bert_base, True)
    model.to(device)
    max_score = 0.5
    for epoch in range(5):

        train_epoch(model, train_loader)
        print("\n")
        F1, F2, score = eval_epoch(model, valid_loader, tokenizer)
        print("\n")
        print("epoch: %d" % epoch, F1, F2, score, "\n")
        if score > max_score:
            # max_score = score
            state = {"bert_base": bert_base.state_dict()}
            torch.save(state, os.path.join("../checkpoint", "bert_best.pt"))
            state = {"model": model.state_dict()}
            torch.save(state, os.path.join("../checkpoint", "model_best.pt"))
