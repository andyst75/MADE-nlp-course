from random import sample
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from IPython.display import clear_output
import numpy as np
import matplotlib.pyplot as plt

BATCH_SIZE = 64

class TextGenerate:
    
    def __init__(self, token_to_idx: dict, idx_to_token: dict, device: torch.device):
        self.token_to_idx = token_to_idx
        self.idx_to_token = idx_to_token
        self.device = device

    def to_matrix(self, texts, max_len=None, pad=None, dtype='int32', batch_first=True):
        """Casts a list of texts into rnn-digestable matrix"""

        
        if pad is None:
            pad = self.token_to_idx['\n']

        max_len = max_len or max(map(len, texts))
        token_ids = np.zeros([len(texts), max_len], dtype) + pad

        for i in range(len(texts)):
            row = [self.token_to_idx[c] for c in texts[i]]
            token_ids[i, :len(row)] = row

        if not batch_first: # convert [batch, time] into [time, batch]
            token_ids = np.transpose(token_ids)

        return token_ids

    def get_random_batch(self, texts, max_len, batch_size=BATCH_SIZE):
        batch = self.to_matrix(sample(texts, batch_size), max_len=max_len)
        batch = torch.tensor(batch, dtype=torch.int64)
        return batch

    def train_model(self, model, opt, lr_scheduler, criterion,
                    texts, n_iter=1_000, batch_size=BATCH_SIZE, plot=True):

        loss_history = []
        max_len = max([len(x) for x in texts])
        
        model.train()
        for i in range(n_iter):
            batch = self.get_random_batch(texts, max_len, batch_size).to(self.device)

            logit_seq = model(batch)
            loss = criterion(logit_seq[:, :-1].contiguous().view(-1, len(self.token_to_idx)), batch[:, 1:].contiguous().view(-1))
            loss_history.append(loss.data.detach().cpu().numpy())

            loss.backward()
            opt.step()
            opt.zero_grad()
            if not lr_scheduler is None:
                lr_scheduler.step()

            if plot and (i+1) % 100 == 0:
                clear_output(True)
                plt.plot(loss_history, label='loss')
                plt.legend()
                plt.show()
        return loss_history[-1]

    def eval_model(self, model, criterion, texts, max_len):
        model.eval()
        with torch.no_grad():
            data = self.to_matrix(texts, max_len=max_len)
            data = torch.tensor(data, dtype=torch.int64).to(self.device)
            logit_seq = model(data).to(self.device)
            loss = criterion(logit_seq[:, :-1].contiguous().view(-1, num_tokens),
                         data[:, 1:].contiguous().view(-1))
            return loss.item()

    def generate_sample(self, model, seed_phrase, max_length, temperature=1.0):
        '''
        The function generates text given a phrase of length at least SEQ_LENGTH.
        :param seed_phrase: prefix characters. The RNN is asked to continue the phrase
        :param max_length: maximum output length, including seed_phrase
        :param temperature: coefficient for sampling.  higher temperature produces more chaotic outputs, 
            smaller temperature converges to the single most likely output.

        Be careful with the model output. This model waits logits (not probabilities/log-probabilities)
        of the next symbol.
        '''

        if hasattr(model, 'rnn'):
            rnn = model.rnn
            hid_state = torch.zeros(rnn.num_layers, 1, rnn.hidden_size).to(self.device)
        elif hasattr(model, 'lstm'):
            rnn = model.lstm
            hid_state = (torch.zeros(rnn.num_layers, 1, rnn.hidden_size).to(self.device),
                        torch.zeros(rnn.num_layers, 1, rnn.hidden_size).to(self.device))
        else:
            raise NotImplementedError
        
        x_sequence = [self.token_to_idx[token] for token in seed_phrase]
        x_sequence = torch.tensor([x_sequence], dtype=torch.int64).to(self.device)

        model.eval()
        with torch.no_grad():
            if len(seed_phrase) > 1:
                    out, hid_state = model(x_sequence[..., :-1], hid_state)

            for _ in range(max_length - len(seed_phrase)):
                out, hid_state = model(x_sequence[..., -1:], hid_state)
                p_next = F.softmax(out / temperature, dim=-1).detach().cpu().numpy().squeeze()

                next_ix = np.random.choice(len(self.token_to_idx), p=p_next)
                next_ix = torch.tensor([[next_ix]], dtype=torch.int64).to(self.device)
                x_sequence = torch.cat([x_sequence, next_ix], dim=1)

        return ''.join([self.idx_to_token[ix] for ix in x_sequence.data.detach().cpu().numpy().squeeze()])
