import pickle
import numpy as np
import torch
import torch.nn as nn
import math

# LSTM Model 
# Load the state model and then combine with model here

class LSTMLanguageModel(nn.Module):
    def __init__(self,vocab_size, hid_dim, emb_dim, num_layers,dropout_rate):
        super().__init__()
        self.num_layers = num_layers
        self.hid_dim = hid_dim
        self.emb_dim = emb_dim

        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hid_dim, 
                            num_layers=num_layers, dropout=dropout_rate, 
                            batch_first=True) #dropout is applied to the output of each LSTM layer except the last layer
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hid_dim, vocab_size)

        self.init_weights()

    def init_weights(self):
        init_range_emb = 0.1
        init_range_other = 1/math.sqrt(self.hid_dim)
        self.embedding.weight.data.uniform_(-init_range_emb, init_range_emb)
        self.fc.weight.data.uniform_(-init_range_other, init_range_other)
        self.fc.bias.data.zero_()
        for i in range(self.num_layers):
            self.lstm.all_weights[i][0] = torch.FloatTensor(self.emb_dim,
                self.hid_dim).uniform_(-init_range_other, init_range_other) #We
            self.lstm.all_weights[i][1] = torch.FloatTensor(self.hid_dim,
                self.hid_dim).uniform_(-init_range_other, init_range_other) #Wh

    def init_hidden(self, batch_size, device):
        hidden = torch.zeros(self.num_layers, batch_size, self.hid_dim).to(device)
        cell = torch.zeros(self.num_layers, batch_size, self.hid_dim).to(device)
        return hidden, cell

    def detach_hidden(self, hidden):
        hidden, cell = hidden
        hidden  = hidden.detach() # not to be used for gradient calculation
        cell = cell.detach()
        return hidden, cell

    def forward(self, src, hidden):
        #src: [batch_size, seq_len]

        embedded = self.dropout(self.embedding(src)) 
        # embedding: [batch_size, seq_len, emb_dim]

        output, hidden = self.lstm(embedded, hidden)
        # output: [batch_size, seq_len, hid_dim]
        # hidden: [num_layers * direction, seq_len, hid_dim]

        output = self.dropout(output)
        prediction = self.fc(output)
        # prediction: [batch_size, seq_len, vocab_size]
        return prediction, hidden
    
def generate(prompt, max_seq_len, temperature, model, tokenizer, vocab, device, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    model.eval()
    tokens = tokenizer(prompt)
    indices = [vocab[t] for t in tokens]
    batch_size = 1
    hidden = model.init_hidden(batch_size, device)
    with torch.no_grad():
        for i in range(max_seq_len):
            src = torch.LongTensor([indices]).to(device)
            prediction, hidden = model(src, hidden)
            
            #prediction: [batch size, seq len, vocab size]
            #prediction[:, -1]: [batch size, vocab size] #probability of last vocab
            
            probs = torch.softmax(prediction[:, -1] / temperature, dim=-1)  
            prediction = torch.multinomial(probs, num_samples=1).item()    
            
            while prediction == vocab['<unk>']: #if it is unk, we sample again
                prediction = torch.multinomial(probs, num_samples=1).item()

            if prediction == vocab['<eos>']:    #if it is eos, we stop
                break

            indices.append(prediction) #autoregressive, thus output becomes input

    itos = vocab.get_itos()
    tokens = [itos[i] for i in indices]
    return tokens


def get_generate(prompt, max_seq_len, temperatures, model, tokenizer, vocab, device, seed):
    stack = {}
    for temperature in temperatures:
        generation = generate(prompt, max_seq_len, temperature, model, tokenizer, 
                            vocab, device, seed)
        stack[temperature] = " ".join(generation)
    return stack