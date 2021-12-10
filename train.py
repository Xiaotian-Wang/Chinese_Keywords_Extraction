from pathlib import Path
import re

def read_data(file_path):
    file_path = Path(file_path)

    raw_text = file_path.read_text(encoding='utf8').strip()
    raw_docs = re.split(r'\n\t?\n', raw_text)
    token_docs = []
    tag_docs = []
    for doc in raw_docs:
        tokens = []
        tags = []
        for line in doc.split('\n'):
            token, tag = line.split('\t')
            tokens.append(token)
            tags.append(tag)
        token_docs.append(tokens)
        tag_docs.append(tags)

    return token_docs, tag_docs

texts, tags = read_data('tagged_data.txt')

from sklearn.model_selection import train_test_split
train_texts, val_texts, train_tags, val_tags = train_test_split(texts, tags, test_size=.2)
val_texts,test_texts, val_tags, test_tags = train_test_split(val_texts, val_tags, test_size=0.5)
unique_tags = set(tag for doc in tags for tag in doc)
tag2id = {tag: id for id, tag in enumerate(unique_tags)}
id2tag = {id: tag for tag, id in tag2id.items()}

import numpy as np

import torch

class KeywordDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# ============================

for i in range(len(train_tags)):
    for j in range(len(train_tags[i])):
        train_tags[i][j]=tag2id[train_tags[i][j]]

for i in range(len(val_tags)):
    for j in range(len(val_tags[i])):
        val_tags[i][j]=tag2id[val_tags[i][j]]

for i in range(len(test_tags)):
    for j in range(len(test_tags[i])):
        test_tags[i][j]=tag2id[test_tags[i][j]]


from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
# tokenizer = BertTokenizer.from_pretrained('pretrained_model/chinese-roberta-wwm-ext')

train_encodings = tokenizer(train_texts, is_split_into_words=True, padding=True)
val_encodings = tokenizer(val_texts, is_split_into_words=True, padding=True)
test_encodings = tokenizer(test_texts, is_split_into_words=True, padding=True)

for i in range(train_encodings.data['input_ids'].__len__()):
    num_zeros = train_encodings.data['input_ids'][i].__len__()-train_tags[i].__len__()
    temp = np.hstack((np.array(train_tags[i]), (np.zeros(num_zeros, dtype=int)+tag2id['O'])))
    train_tags[i] = list(temp)

for i in range(val_encodings.data['input_ids'].__len__()):
    num_zeros = val_encodings.data['input_ids'][i].__len__()-val_tags[i].__len__()
    temp = np.hstack((np.array(val_tags[i]), (np.zeros(num_zeros, dtype=int)+tag2id['O'])))
    val_tags[i] = list(temp)

for i in range(test_encodings.data['input_ids'].__len__()):
    num_zeros = test_encodings.data['input_ids'][i].__len__()-test_tags[i].__len__()
    temp = np.hstack((np.array(test_tags[i]), (np.zeros(num_zeros, dtype=int)+tag2id['O'])))
    test_tags[i] = list(temp)


train_dataset = KeywordDataset(train_encodings, train_tags)
val_dataset = KeywordDataset(val_encodings, val_tags)
test_dataset = KeywordDataset(test_encodings, test_tags)


from transformers import BertForTokenClassification

from transformers import BertModel

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# model = BertForTokenClassification.from_pretrained('bert-base-chinese', num_labels=len(unique_tags),return_dict=True)
model = BertModel.from_pretrained('hfl/chinese-roberta-wwm-ext', num_labels=len(unique_tags),return_dict=False,)
# outputs = model(input_ids, attention_mask=attention_mask)

from torchcrf import CRF

# 768 for bert-base-chinese
linear_model = torch.nn.Linear(768, 3).to(device)
model_crf = CRF(len(unique_tags)).to(device)

class BertCrf(torch.nn.Module):
    def __init__(self):
        super(BertCrf, self).__init__()
        self.bert = model
        self.linear = linear_model
        self.crf = model_crf

    def forward(self, x, attention_mask, tags):
        x = self.bert(x, attention_mask=attention_mask)[0]
        x = self.linear(x)
        x = self.crf(x, tags=tags)
        return x

    def decode(self, x, attention_mask):
        x = self.bert(x, attention_mask=attention_mask)[0]
        x = self.linear(x)
        x = self.crf.decode(x)
        return x

from torch.utils.data import DataLoader
from transformers import AdamW

model2 = BertCrf().to(device)

"""
# This block makes the model features based, 
# Only tuning the last five layers.

for name, param in model2.named_parameters():
    if name.startswith('bert'):
        param.requires_grad = False
        
"""

# model2.to(device)
model2.train()

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

optim = AdamW(model.parameters(), lr=5e-6, weight_decay=0)

from tqdm import tqdm

input_ids_val = torch.tensor(val_dataset.encodings['input_ids']).to(device)
attention_mask_val = torch.tensor(val_dataset.encodings['attention_mask']).to(device)
tags_val = torch.tensor(val_dataset.labels).to(device)

for epoch in range(30):
    for batch in tqdm(train_loader):
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model2(input_ids, attention_mask=attention_mask, tags=labels)
        # loss = outputs[0]
        loss = outputs*torch.tensor(-1)
        loss.backward()
        optim.step()
    print(epoch, loss.tolist())

    outputs_val = model2(input_ids_val, attention_mask=attention_mask_val, tags=tags_val)
    loss_val = outputs_val*torch.tensor(-1)
    print('validation loss:', loss_val.tolist())

model2.eval()

print("finished")

"""
x = val_dataset[10]
predicted = torch.tensor(model2.decode(x.get('input_ids').unsqueeze(0).to(device), attention_mask=x.get('attention_mask').unsqueeze(0).to(device)))
predicted.to(device).t()-x.get('labels').unsqueeze(0).to(device)
"""

def make_wordlist(texts, tags_original):
    wordlist = []
    tags = [id2tag[int(item)] for item in tags_original[0]]
    for i in range(len(tags)):
        if tags[i] == 'B_keyword':
            length = 1
            while(i+length < len(tags) and tags[i+length]=='I_keyword'):
                length += 1
            wordlist.append(''.join(texts[i:i+length]))
    return wordlist

"""
texts = [tokenizer.decode(item) for item in x['input_ids']]
while '[ C L S ]' in texts:
    texts.remove('[ C L S ]')


predicted_list = list(set(make_wordlist(texts, predicted.to(device).t())))
target_list = list(set(make_wordlist(texts, x.get('labels').unsqueeze(0).to(device))))


temp = [item for item in predicted_list if item in target_list]

precision = len(temp)/len(predicted_list)
recall = len(temp)/len(target_list)
print('precision:', precision)
print('recall:', recall)
print('f1 score:', 2*precision*recall/(precision+recall))
"""

def get_envaluations(val_dataset):
    precision_list = []
    recall_list = []
    f1_score_list = []

    correct = []
    predicted_num = []
    target_num = []

    for data_sample in val_dataset:

        predicted = torch.tensor(model2.decode(data_sample.get('input_ids').unsqueeze(0).to(device),
                                               attention_mask=data_sample.get('attention_mask').unsqueeze(0).to(device)))
        predicted.to(device).t() - data_sample.get('labels').unsqueeze(0).to(device)
        texts = [tokenizer.decode(item) for item in data_sample['input_ids']]
        while '[ C L S ]' in texts:
            texts.remove('[ C L S ]')

        predicted_list = list(set(make_wordlist(texts, predicted.to(device).t())))
        while '[ S E P ]' in predicted_list:
            predicted_list.remove('[ S E P ]')

        while '[ C L S ]' in predicted_list:
            predicted_list.remove('[ C L S ]')

        while '[ U N K ]' in predicted_list:
            predicted_list.remove('[ U N K ]')

        target_list = list(set(make_wordlist(texts, data_sample.get('labels').unsqueeze(0).to(device))))

        temp = [item for item in predicted_list if item in target_list]

        precision = len(temp) / len(predicted_list)
        recall = len(temp) / len(target_list)
        if (precision+recall) > 0:
            f1_score = 2*precision*recall/(precision+recall)
        else:
            f1_score = 0

        correct.append(len(temp))
        target_num.append(len(target_list))
        predicted_num.append(len(predicted_list))

        precision_list.append(precision)
        recall_list.append(recall)
        f1_score_list.append(f1_score)

    # print('precision:', np.mean(precision_list))
    # print('recall:', np.mean(recall_list))
    # print('f1_score:', np.mean(f1_score_list))

    print('precision: ', np.sum(correct)/np.sum(predicted_num))
    print('recall: ', np.sum(correct)/np.sum(target_num))

    print('f1 score:', 2*np.sum(correct)/(np.sum(predicted_num)+np.sum(target_num)))
    return precision_list, recall_list, f1_score_list

a, b, c = get_envaluations(train_dataset)
a, b, c = get_envaluations(val_dataset)
a, b, c = get_envaluations(test_dataset)

'''
import numpy as np
test_sample = val_dataset[19]
with torch.no_grad():
    output = model(input_ids=test_sample['input_ids'].unsqueeze(0).to(device),attention_mask=test_sample['attention_mask'].unsqueeze(0).to(device))

output = output[0].squeeze(0)

mylabel = list()

for i in range(output.size()[0]):
    mylabel.append(output[i].argmax().tolist())

thelabel = test_sample['labels'].tolist()

print(np.vstack((np.array(mylabel),np.array(thelabel))))
print(tokenizer.decode(test_sample['input_ids']))

'''

