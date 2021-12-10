import torch
from torchcrf import CRF
from transformers import BertModel
from transformers import BertTokenizer

class BertCrf(torch.nn.Module):
    def __init__(self):
        super(BertCrf, self).__init__()
        self.unique_label_num = 3
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.bert = BertModel.from_pretrained('hfl/chinese-roberta-wwm-ext', num_labels=self.unique_label_num,return_dict=False,)
        # self.bert = BertModel.from_pretrained('pretrained_model/chinese-roberta-wwm-ext', num_labels=self.unique_label_num,return_dict=False,)
        self.linear = torch.nn.Linear(768, self.unique_label_num).to(self.device)
        self.crf = CRF(self.unique_label_num).to(self.device)

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


model = BertCrf()
model.load_state_dict(torch.load('trained_model.pth'))
model.eval()

# tokenizer = BertTokenizer.from_pretrained('pretrained_model/chinese-roberta-wwm-ext')
tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

tag2id = {'O': 0, 'I_keyword': 1, 'B_keyword': 2}
id2tag = {0: 'O', 1: 'I_keyword', 2: 'B_keyword'}

def make_wordlist(texts, tags_original):
    wordlist = []
    tags = [id2tag[int(item)] for item in tags_original[0]]
    for i in range(len(tags)):
        if tags[i] == 'B_keyword':
            length = 1
            while(i+length < len(tags) and tags[i+length]=='I_keyword'):
                length += 1
            wordlist.append(''.join(texts[i:i+length]))
    return list(set(wordlist))

model.to(device)

def extract_from_text(the_text):
    tokenized = tokenizer(the_text)
    tokenized['input_ids'] = torch.tensor(tokenizer.encode(list(the_text))).to(device)
    tokenized['token_type_ids'] = torch.zeros(len(tokenized['input_ids'])).to(device)
    tokenized['attention_mask'] = torch.ones(len(tokenized['input_ids'])).to(device)
    predicted = torch.tensor(model.decode(tokenized.get('input_ids').unsqueeze(0).to(device),
                                           attention_mask=tokenized.get('attention_mask').unsqueeze(0).to(device)))

    return make_wordlist(the_text, predicted.t())


train_texts = [''.join(item) for item in train_texts]
word_list = []
predicted_list = []
for i in range(len(train_texts)):
    word_list.append(make_wordlist(texts=list(train_texts[i]), tags_original=[train_tags[i]]))

for i in range(len(train_texts)):
    predicted_list.append(jieba.analyse.textrank(train_texts[i]))

correct = []
target = []
predicted = []

for i in range(len(train_texts)):
    intersection = [item for item in word_list[i] if item in predicted_list[i]]
    correct.append(len(intersection))
    target.append(len(word_list[i]))
    predicted.append(len(predicted_list[i]))