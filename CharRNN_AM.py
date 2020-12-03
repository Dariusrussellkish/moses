#!/usr/bin/env python
import os

import pandas as pd
import torch

import moses
from moses.char_rnn import CharRNN
from moses.char_rnn_am import CharRNNTrainer
from moses.char_rnn_am import config as CharRNNConfig
from moses.utils import CharVocab

from ax.service.managed_loop import optimize


df = pd.read_csv('data/ngram_smiles.csv')
neg_trigrams = df.loc[(df['n'] == 3) | (df['n'] == 4)]['smiles'].values

assert torch.cuda.is_available()
torch.device(0)

train = moses.get_dataset('train')
neg = neg_trigrams


def train_evaluate(parameters=None):
    if parameters is None:
        parameters = {}
    config = CharRNNConfig.get_config()
    config.log_file = None
    config.n_workers = 1
    config.n_batch = 128
    config.model_save = os.path.join('crnn_am', 'crnn_am_8.pth')
    config.save_frequency = 5
    config.train_epochs = parameters.get('train_epochs', 20)
    config.alpha = parameters.get('alpha', 1.)
    config.num_layers = parameters.get('num_layers', config.num_layers)
    config.hidden = parameters.get('hidden', config.hidden)
    config.dropout = parameters.get('dropout', config.dropout)
    trainer = CharRNNTrainer(config)
    crnn = CharRNN(CharVocab.from_data(train), config)
    crnn.cuda()
    trainer.fit_am(crnn, train, neg, parameters=parameters)
    crnn.cpu()
    test_vals = crnn.sample(30_000)
    metrics = moses.get_all_metrics(test_vals)
    return metrics['valid'] * metrics['Novelty']


best_parameters, values, experiment, model = optimize(
    parameters=[
        {'name': 'lr', 'type': 'range', 'bounds': [1e-8, 5e-1], 'log_scale': True},
        {'name': 'alpha', 'type': 'range', 'bounds': [0.1, 8.], 'log_scale': False},
        {'name': 'step_size', 'type': 'range', 'bounds': [10, 60], 'log_scale': False},
        {'name': 'gamma', 'type': 'range', 'bounds': [0.01, 0.5], 'log_scale': False},
        # {'name': 'num_layers', 'type': 'range', 'bounds': [1, 6], 'log_scale': False},
        # {'name': 'hidden', 'type': 'range', 'bounds': [256, 1024], 'log_scale': False},
        # {'name': 'dropout', 'type': 'range', 'bounds': [0.01, 0.5], 'log_scale': False},
    ],
    evaluation_function=train_evaluate,
    objective_name="Novelty*Diversity"
)

print(best_parameters)
means, covariances = values
print(means)
print(covariances)
