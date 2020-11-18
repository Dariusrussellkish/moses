#!/usr/bin/env python
# coding: utf-8

# In[1]:


from torch.utils.data import DataLoader
from moses import CharVocab, StringDataset
from moses.char_rnn import CharRNN
from moses.char_rnn import config as CharRNNConfig
import moses
import torch
from torch import autograd
import rdkit.Chem as chem


# In[2]:


import torch
import torch.nn as nn
import torch.optim as optim

from tqdm.auto import tqdm
from torch.nn.utils.rnn import pad_sequence

from moses.interfaces import MosesTrainer
from moses.utils import CharVocab, Logger


def anti_model_rnn_loss(loss, alpha=8.0):
    mask = (loss == 0.0)
    loss = torch.exp(-1 * loss)
    loss = 1. - loss + 1.e-7
    loss = torch.log(loss)
    loss = -alpha * loss
    loss[mask] = 0.0
    return loss


class CharRNNTrainer(MosesTrainer):

    def __init__(self, config):
        self.config = config

    def _train_epoch(self, model, tqdm_data, criterion, optimizer=None):
        if optimizer is None:
            model.eval()
        else:
            model.train()

        postfix = {'loss': 0,
                   'running_loss': 0}

        for i, ((prevs, nexts, lens), (nprevs, nnexts, nlens)) in enumerate(tqdm_data):
            prevs = prevs.to(model.device)
            nexts = nexts.to(model.device)
            lens = lens.to(model.device)
            
            nprevs = nprevs.to(model.device)
            nnexts = nnexts.to(model.device)
            nlens = nlens.to(model.device)
            
            outputs, _, _ = model(prevs, lens)
            noutputs, _, _ = model(nprevs, nlens)

            loss = criterion(outputs.view(-1, outputs.shape[-1]),
                             nexts.view(-1))
            
            nloss = criterion(noutputs.view(-1, noutputs.shape[-1]),
                             nnexts.view(-1))
            
            nloss = anti_model_rnn_loss(nloss)
            loss = torch.mean(loss)
            nloss = torch.mean(nloss)
            loss = loss + nloss
            
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            
#             if optimizer is not None:
#                 optimizer.zero_grad()
#                 nloss.backward()
#                 optimizer.step()

            postfix['loss'] = loss.item() + nloss.item()
            postfix['running_loss'] += (loss.item() + nloss.item() -
                                        postfix['running_loss']) / (i + 1)
            tqdm_data.set_postfix(postfix)

        postfix['mode'] = 'Eval' if optimizer is None else 'Train'
        return postfix

    def _train(self, model, train_loader, negative_loader, val_loader=None, logger=None):
        def get_params():
            return (p for p in model.parameters() if p.requires_grad)

        device = model.device
        criterion = nn.CrossEntropyLoss(reduction='none')
        optimizer = optim.Adam(get_params(), lr=self.config.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer,
                                              self.config.step_size,
                                              self.config.gamma)

        model.zero_grad()
        for epoch in range(self.config.train_epochs):

            tqdm_data = tqdm(zip(train_loader, negative_loader),
                             desc='Training (epoch #{})'.format(epoch), total=min(len(train_loader), len(negative_loader)))
            postfix = self._train_epoch(model, tqdm_data, criterion, optimizer)
            if logger is not None:
                logger.append(postfix)
                logger.save(self.config.log_file)

            if val_loader is not None:
                tqdm_data = tqdm(val_loader,
                                 desc='Validation (epoch #{})'.format(epoch))
                postfix = self._train_epoch(model, tqdm_data, criterion)
                if logger is not None:
                    logger.append(postfix)
                    logger.save(self.config.log_file)

            if (self.config.model_save is not None) and (epoch % self.config.save_frequency == 0):
                model = model.to('cpu')
                torch.save(
                    model.state_dict(),
                    self.config.model_save[:-3]+'_{0:03d}.pt'.format(epoch)
                )
                model = model.to(device)

            scheduler.step()

    def get_vocabulary(self, data):
        return CharVocab.from_data(data)

    def get_collate_fn(self, model):
        device = self.get_collate_device(model)

        def collate(data):
            data.sort(key=len, reverse=True)
            tensors = [model.string2tensor(string, device=device)
                       for string in data]

            pad = model.vocabulary.pad
            prevs = pad_sequence([t[:-1] for t in tensors],
                                 batch_first=True, padding_value=pad)
            nexts = pad_sequence([t[1:] for t in tensors],
                                 batch_first=True, padding_value=pad)
            lens = torch.tensor([len(t) - 1 for t in tensors],
                                dtype=torch.long, device=device)
            return prevs, nexts, lens

        return collate

    def fit(self, model, train_data, negative_data, val_data=None):
        logger = Logger() if self.config.log_file is not None else None

        train_loader = self.get_dataloader(model, train_data, shuffle=True)
        negative_loader = self.get_dataloader(model, negative_data, shuffle=True)
        
        val_loader = None if val_data is None else self.get_dataloader(
            model, val_data, shuffle=False
        )

        self._train(model, train_loader, negative_loader, val_loader, logger)
        return model


# In[3]:


import pandas as pd


# In[4]:


df = pd.read_csv('data/ngram_smiles.csv')


# In[5]:


neg_trigrams = df.loc[(df['n'] == 3) | (df['n'] == 4)]['smiles'].values


# In[6]:


torch.cuda.is_available()


# In[7]:


torch.device(0)


# In[8]:


torch.cuda.is_initialized()


# In[ ]:


train = moses.get_dataset('train')
neg = neg_trigrams

config = CharRNNConfig.get_config()
config.log_file = None
config.n_workers = 1
config.n_batch = 2048
config.model_save = 'crnn_am_8.pth'
config.save_frequency = 20
config.train_epochs = 40
trainer = CharRNNTrainer(config)
crnn = CharRNN(CharVocab.from_data(train), config)
crnn.load_state_dict(torch.load('crnn_am_8._040.pt'))
crnn.cuda()
trainer.fit(crnn, train, neg)


# In[ ]:


neg_trigrams


# In[ ]:


crnn.cpu()


# In[ ]:


test_vals = crnn.sample(100_000)


# In[ ]:


with open('AM_8_results.json', 'w') as fh:
    fh.write(str(moses.get_all_metrics(test_vals)))


# In[ ]:




