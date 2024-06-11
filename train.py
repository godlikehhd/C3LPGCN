import csv
import math
import os
import sys
import copy
import random
import logging
import argparse
import time

import torch
import pandas as pd
import torch.nn as nn
import numpy as np
from sklearn import metrics
from criterion_new import *
from time import strftime, localtime
from torch.utils.data import DataLoader
from transformers import AdamW, BertModel, BertConfig
# from transformers.models.bert.modeling_bert import BertModel, BertConfig, BertForMaskedLM
from models.clptgcn import CLPTGCN
from models.dualGCN import DualGCNBertClassifier
from models.SSEGCN import SSEGCNBertClassifier
from data_utils.data_utils import Tokenizer4BertGCN, ABSAGCNData

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


def adjust_learning_rate(optimizer, epoch, opt):
    """Decay the learning rate based on schedule"""
    bert_lr = opt.bert_lr
    if opt.cos:  # cosine lr schedule
        bert_lr *= 0.5 * (1. + math.cos(math.pi * epoch / opt.num_epoch))
    else:  # stepwise lr schedule
        for milestone in opt.schedule:
            bert_lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = bert_lr


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


import re


def func(s):
    s = re.sub(r'\d+.', '', s)
    return s


class Instructor:
    ''' Model training and evaluation '''

    def __init__(self, opt):
        self.opt = opt

        tokenizer = Tokenizer4BertGCN(opt.max_length, opt.pretrained_bert_name)
        self.opt.tokenizer = tokenizer
        self.opt.config = BertConfig.from_pretrained(opt.pretrained_bert_name)
        bert = BertModel.from_pretrained(opt.pretrained_bert_name)
        self.model = opt.model_class(bert, self.opt).to(opt.device)

        trainset = ABSAGCNData(opt.dataset_file['train'], tokenizer, opt, istrain=True, do_augment=True)

        testset = ABSAGCNData(opt.dataset_file['test'], tokenizer, opt, istrain=False)

        self.train_dataloader = DataLoader(dataset=trainset, batch_size=opt.batch_size, shuffle=True)
        self.test_dataloader = DataLoader(dataset=testset, batch_size=opt.batch_size)

        if opt.device.type == 'cuda':
            logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(self.opt.device.index)))
        self._print_args()

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params

        logger.info(
            'n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        logger.info('training arguments:')

        # params_list = [n for n, p in list(self.model.named_parameters())]
        # params_list = set(map(func, params_list))
        # print(params_list)

        for arg in vars(self.opt):
            logger.info('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _reset_params(self):
        other = ['lstm', 'W', 'dense_mask', 'dense_aspect', 'cls', 'dense_sent', 'w_v']
        for n, p in self.model.named_parameters():
            name = n.split('.')
            if len(list(set(other) & set(name))) != 0:
                if p.requires_grad:
                    if len(p.shape) > 1:
                        self.opt.initializer(p)  # xavier_uniform_
                    else:
                        pass
            else:
                pass

    def get_bert_optimizer(self, model):
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        diff_part = ["bert.embeddings", "bert.encoder"]

        if self.opt.diff_lr:
            logger.info("layered learning rate on")
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in model.named_parameters() if
                               not any(nd in n for nd in no_decay) and any(nd in n for nd in diff_part)],
                    "weight_decay": self.opt.weight_decay,
                    "lr": self.opt.bert_lr
                },
                {
                    "params": [p for n, p in model.named_parameters() if
                               any(nd in n for nd in no_decay) and any(nd in n for nd in diff_part)],
                    "weight_decay": 0.0,
                    "lr": self.opt.bert_lr
                },
                {
                    "params": [p for n, p in model.named_parameters() if
                               not any(nd in n for nd in no_decay) and not any(nd in n for nd in diff_part)],
                    "weight_decay": self.opt.weight_decay,
                    "lr": self.opt.learning_rate
                },
                {
                    "params": [p for n, p in model.named_parameters() if
                               any(nd in n for nd in no_decay) and not any(nd in n for nd in diff_part)],
                    "weight_decay": 0.0,
                    "lr": self.opt.learning_rate
                },
            ]
            optimizer = AdamW(optimizer_grouped_parameters, eps=self.opt.adam_epsilon)

        else:
            logger.info("bert learning rate on")
            params = list(self.model.named_parameters())
            no_decay = ['bias,', 'LayerNorm']
            other = ['lstm', 'W', 'dense_mask', 'dense_aspect', 'cls', 'dense_sent', 'w_v']
            no_main = no_decay + other

            param_group = [
                {'params': [p for n, p in params if not any(nd in n for nd in no_main)],
                 'weight_decay': self.opt.weight_decay,
                 'lr': self.opt.bert_lr},
                {'params': [p for n, p in params if
                            not any(nd in n for nd in other) and any(nd in n for nd in no_decay)], 'weight_decay': 0,
                 'lr': self.opt.bert_lr},
                {'params': [p for n, p in params if any(nd in n for nd in other) and any(nd in n for nd in no_decay)],
                 'weight_decay': 0, 'lr': self.opt.learning_rate},
                {'params': [p for n, p in params if
                            any(nd in n for nd in other) and not any(nd in n for nd in no_decay)],
                 'weight_decay': self.opt.weight_decay,
                 'lr': self.opt.learning_rate},
            ]

            optimizer = AdamW(param_group, lr=self.opt.bert_lr, eps=self.opt.adam_epsilon)

        return optimizer

    def _train(self, criterion, optimizer, max_test_acc_overall, contrastiveLoss_sentiment):

        max_test_acc = 0
        max_f1 = 0
        global_step = 0
        model_path = ''

        for epoch in range(self.opt.num_epoch):
            logger.info('>' * 60)
            logger.info('epoch: {}'.format(epoch))
            n_correct, n_total = 0, 0
            for i_batch, sample_batched in enumerate(self.train_dataloader):

                global_step += 1
                self.model.train()
                optimizer.zero_grad()
                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                targets = sample_batched['polarity'].to(self.opt.device)
                logits, logits_pos, scores, label_id, loss_kl, loss_cl, loss_dl, pooled_out = self.model(inputs)
                loss_scl = contrastiveLoss_sentiment(pooled_out, targets)
                if self.opt.dataset != 'restaurant':
                    label_id = label_id.view(-1)
                    scores = scores.view(-1, 30522)
                    loss = criterion(logits, targets) + loss_cl * self.opt.l1  + \
                           loss_dl * self.opt.l6 + loss_kl * self.opt.l3 + loss_scl * self.opt.l4 + criterion(scores, label_id)*self.opt.l5
                else:
                    loss = criterion(logits, targets) + loss_cl * self.opt.l1 + \
                           loss_dl * self.opt.l6 + loss_kl * self.opt.l3 + loss_scl * self.opt.l4 
                loss.backward()
                optimizer.step()

                if global_step % self.opt.log_step == 0:
                    n_correct += (torch.argmax(logits, -1) == targets).sum().item()
                    n_total += len(logits)
                    train_acc = n_correct / n_total
                    test_acc, f1 = self._evaluate()
                    if test_acc > max_test_acc:
                        max_test_acc = test_acc
                        if test_acc > max_test_acc_overall:
                            if not os.path.exists('state_dict'):
                                os.mkdir('state_dict')
                            model_path = 'state_dict/{}_{}_acc_{:.4f}_f1_{:.4f}'.format(self.opt.model_name,
                                                                                        self.opt.dataset, test_acc, f1)
                            self.best_model = copy.deepcopy(self.model)
                            logger.info('>> saved: {}'.format(model_path))
                    if f1 > max_f1:
                        max_f1 = f1
                    logger.info(
                        'loss: {:.4f}, acc: {:.4f}, test_acc: {:.4f}, f1: {:.4f}'.format(
                            loss.item(), train_acc, test_acc, f1))
        return max_test_acc, max_f1, model_path

    def _evaluate(self, show_results=False):
        # switch model to evaluation mode
        self.model.eval()
        n_test_correct, n_test_total = 0, 0
        targets_all, outputs_all = None, None

        pooled_out_all = None
        with torch.no_grad():
            for batch, sample_batched in enumerate(self.test_dataloader):
                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                targets = sample_batched['polarity'].to(self.opt.device)
                logits, logits_pos, scores, label_id, loss_kl, loss_cl, loss_dl, pooled_out = self.model(inputs)
                n_test_correct += (torch.argmax(logits, -1) == targets).sum().item()
                n_test_total += len(logits)
                targets_all = torch.cat((targets_all, targets), dim=0) if targets_all is not None else targets
                outputs_all = torch.cat((outputs_all, logits), dim=0) if outputs_all is not None else logits
                pooled_out_all = torch.cat((pooled_out_all, pooled_out),
                                           dim=0) if pooled_out_all is not None else pooled_out


        test_acc = n_test_correct / n_test_total
        f1 = metrics.f1_score(targets_all.cpu(), torch.argmax(outputs_all, -1).cpu(), labels=[0, 1, 2], average='macro')

        labels = targets_all.data.cpu()
        predic = torch.argmax(outputs_all, -1).cpu()
        if show_results:
            report = metrics.classification_report(labels, predic, digits=4)
            confusion = metrics.confusion_matrix(labels, predic)
            return report, confusion, test_acc, f1

        return test_acc, f1

    def _test(self):

        self.model = self.best_model
        self.model.eval()
        test_report, test_confusion, acc, f1 = self._evaluate(show_results=True)
        logger.info('accuracy:')
        logger.info(acc)
        logger.info('f1_score:')
        logger.info(f1)
        logger.info("Precision, Recall and F1-Score...")
        logger.info(test_report)
        logger.info("Confusion Matrix...")
        logger.info(test_confusion)

    def run(self, ):
        criterion = nn.CrossEntropyLoss(ignore_index=-1)
        optimizer = self.get_bert_optimizer(self.model)
        max_f1_overall = 0
        max_test_acc_overall = 0
        self._reset_params()
        contrastiveLoss_sentiment = CL_sentiment()
        max_test_acc, max_f1, model_path = self._train(criterion, optimizer, max_test_acc_overall,
                                                       contrastiveLoss_sentiment)
        logger.info('max_test_acc: {0}, max_f1: {1}'.format(max_test_acc, max_f1))
        max_test_acc_overall = max(max_test_acc, max_test_acc_overall)
        max_f1_overall = max(max_f1, max_f1_overall)
        torch.save(self.best_model.state_dict(), model_path)
        logger.info('#' * 60)
        logger.info('max_test_acc_overall:{}'.format(max_test_acc_overall))
        logger.info('max_f1_overall:{}'.format(max_f1_overall))
        self._test()


def main():
    model_classes = {

        'clpt': CLPTGCN,
        'dual': DualGCNBertClassifier,
        'ssegcn': SSEGCNBertClassifier
    }

    dataset_files = {
        'restaurant': {
            'train': '.\dataset/Restaurants_allennlp/train.json',
            'test': '.\dataset/Restaurants_allennlp/test.json',
        },
        'laptop': {
            'train': '.\dataset\Laptops_allennlp/train.json',
            'test': '.\dataset\Laptops_allennlp/train.json'
        },
        'twitter': {
            'train': '.\dataset/Tweets_allennlp/train.json',
            'test': '.\dataset/Tweets_allennlp/test.json',
        },

    }

    input_cols = {
        'clpt': ['text_bert_indices_mask', 'text_bert_indices_pos', 'text_bert_indices_neg_1',
                 'text_bert_indices_neg_2', 'bert_segments_ids_aul', 'attention_mask_aul', 'adj_matrix', 'src_mask',
                 'aspect_mask',
                 'label_id', 'loc_mask'],
        'ssegcn': ['text_bert_indices', 'bert_segments_ids', 'attention_mask', 'asp_start', 'asp_end', 'src_mask',
                       'aspect_mask', 'short_mask'],
        'dual': ['text_bert_indices', 'bert_segments_ids', 'attention_mask', 'asp_start', 'asp_end',
                        'adj_matrix', 'src_mask', 'aspect_mask']
    }

    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal_,
        'orthogonal_': torch.nn.init.orthogonal_,
    }

    optimizers = {
        'adadelta': torch.optim.Adadelta,
        'adagrad': torch.optim.Adagrad,
        'adam': torch.optim.Adam,
        'adamax': torch.optim.Adamax,
        'asgd': torch.optim.ASGD,
        'rmsprop': torch.optim.RMSprop,
        'sgd': torch.optim.SGD,
    }


    # Hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='clpt', type=str, help=', '.join(model_classes.keys()))
    parser.add_argument('--dataset', default='laptop', type=str, help=', '.join(dataset_files.keys()))
    parser.add_argument('--optimizer', default='adam', type=str, help=', '.join(optimizers.keys()))
    parser.add_argument('--initializer', default='xavier_uniform_', type=str, help=', '.join(initializers.keys()))
    parser.add_argument('--learning_rate', default=1e-4, type=float)
    parser.add_argument('--num_epoch', default=4, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--log_step', default=5, type=int)
    parser.add_argument('--hidden_dim', type=int, default=768, help='GCN mem dim.')
    parser.add_argument('--num_layers', type=int, default=1, help='Num of GCN layers.')
    parser.add_argument('--polarities_dim', default=3, type=int, help='3')

    parser.add_argument('--input_dropout', type=float, default=0.2, help='Input dropout rate.')
    parser.add_argument('--gcn_dropout', type=float, default=0.1, help='GCN layer dropout rate.')
    parser.add_argument('--lower', default=True, help='Lowercase all words.')
    parser.add_argument('--direct', default=False, help='directed graph or undirected graph')
    parser.add_argument('--loop', default=True)
    parser.add_argument('--bidirect', default=True, help='Do use bi-RNN layer.')
    parser.add_argument('--rnn_hidden', type=int, default=50, help='RNN hidden state size.')
    parser.add_argument('--rnn_layers', type=int, default=3, help='Number of RNN layers.')
    parser.add_argument('--rnn_dropout', type=float, default=0.1, help='RNN dropout rate.')
    parser.add_argument('--attention_heads', default=2, type=int, help='number of multi-attention heads')
    parser.add_argument('--max_length', default=85, type=int)
    parser.add_argument('--device', default=None, type=str, help='cpu, cuda')
    parser.add_argument('--seed', default=1000, type=int)
    parser.add_argument("--weight_decay", default=1e-3, type=float, help="Weight deay if we apply some.")
    parser.add_argument('--parseadj', default=False, action='store_true', help='dependency probability')
    parser.add_argument('--parsehead', default=False, action='store_true', help='dependency tree')
    parser.add_argument('--cuda', default='0', type=str)
    # * bert
    parser.add_argument('--pretrained_bert_name', default='bert-base-uncased', type=str)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument('--bert_dim', type=int, default=768)
    parser.add_argument('--bert_dropout', type=float, default=0.0, help='BERT dropout rate.')
    parser.add_argument('--diff_lr', default=False, action='store_true')
    parser.add_argument('--bert_lr', default=3e-5, type=float)
    parser.add_argument('--l1', default=0.1, type=float, help='pcl_loss')
    parser.add_argument('--l3', default=0.1, type=float, help='kl_loss')
    parser.add_argument('--l4', default=0.3, type=float, help='scl_loss')
    parser.add_argument('--l5', default=0.01, type=float, help='pt_loss')
    parser.add_argument('--l6', default=0.1, type=float, help='cl_loss')
    parser.add_argument('--temp', default=0.05, type=float, help='tempture of pcl')

    opt = parser.parse_args()
    opt.model_class = model_classes[opt.model_name]
    opt.dataset_file = dataset_files[opt.dataset]
    opt.inputs_cols = input_cols[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]

    print("choice cuda:{}".format(opt.cuda))
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if opt.device is None else torch.device(
        opt.device)

    # set random seed
    setup_seed(opt.seed)

    if not os.path.exists('log'):
        os.makedirs('log', mode=0o777)
    log_file = '{}-{}-{}.log'.format(opt.model_name, opt.dataset, strftime("%Y-%m-%d_%H_%M_%S", localtime()))
    logger.addHandler(logging.FileHandler("%s/%s" % ('log', log_file)))

    ins = Instructor(opt)
    ins.run()


if __name__ == '__main__':
    main()
