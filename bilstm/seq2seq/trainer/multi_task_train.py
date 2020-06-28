#!/user/bin/env python3
# -*- utf-8 -*-
# author shengqiong.wu

from __future__ import division
import logging
import os
import random
import tqdm
import time

import torch
import torchtext
from torch import optim
import torch.nn as nn
from sklearn.metrics import classification_report
from sklearn import metrics

import seq2seq
from seq2seq.evaluator.multi_task_evaluator import Evaluator
from seq2seq.loss.loss import NLLLoss
from seq2seq.optim.optim import Optimizer
from seq2seq.util.checkpoint import Checkpoint
from seq2seq.util.utils import *


class MultiTaskTrainer(object):
    """ The SupervisedTrainer class helps in setting up a training framework in a
    supervised setting.
    Args:
        expt_dir (optional, str): experiment Directory to store details of the experiment,
            by default it makes a folder in the current directory to store the details (default: `experiment`).
        loss (seq2seq.loss.loss.Loss, optional): loss for training, (default: seq2seq.loss.NLLLoss)
        batch_size (int, optional): batch size for experiment, (default: 64)
        checkpoint_every (int, optional): number of batches to checkpoint after, (default: 100)
    """
    def __init__(self, config, expt_dir='experiment', loss=NLLLoss(), norm_batch_size=64, class_batch_size=15,
                 random_seed=123, print_every=100, logger=logging.getLogger(__name__), use_cuda=True):
        self._trainer = "Simple Trainer"
        self.config = config
        self.random_seed = random_seed
        if random_seed is not None:
            random.seed(random_seed)
            torch.manual_seed(random_seed)
        self.loss = loss
        self.class_loss = nn.CrossEntropyLoss()
        self.norm_loss = nn.CrossEntropyLoss()
        self.adv_loss = nn.CrossEntropyLoss()
        self.evaluator = Evaluator(loss=self.loss, batch_size=norm_batch_size)
        self.optimizer = None
        # self.checkpoint_every = checkpoint_every
        self.print_every = print_every

        if not os.path.isabs(expt_dir):
            expt_dir = os.path.join(os.getcwd(), expt_dir)
        self.expt_dir = expt_dir
        if not os.path.exists(self.expt_dir):
            os.makedirs(self.expt_dir)
        self.norm_batch_size = norm_batch_size
        self.class_batch_size = class_batch_size

        self.logger = logger
        self.use_cuda = use_cuda
        self.best_fb_f1 = 0.0
        self.best_tw_f1 = 0.0

    def _train_batch(self, input_variable, c_inputs, target_variable, model, teacher_forcing_ratio, train_type=None):

        model.zero_grad()
        if train_type == 'pre_train':
            # loss = self.loss
            # Forward propagation
            (decoder_outputs, decoder_hidden, other), share_feature, private_feature, feature = model(input_variable, c_inputs,
                                                                                                      target_variable,
                                                                                                      teacher_forcing_ratio=teacher_forcing_ratio,
                                                                                                      train_type=train_type)
            norm_loss = 0
            for step, step_output in enumerate(decoder_outputs):
                batch_size = target_variable.size(0)
                # loss.eval_batch(step_output.contiguous().view(batch_size, -1), target_variable[:, step + 1])
                norm_loss = norm_loss + self.norm_loss(step_output.contiguous().view(batch_size, -1), target_variable[:, step + 1])

            # adversarial loss
            target = torch.zeros(input_variable.size(0), dtype=torch.long)
            if self.use_cuda:
                target = target.cuda()
            adversarial_loss = self.adv_loss(feature, target)

            # different loss
            share_features = share_feature - torch.mean(share_feature, dim=0)
            private_features = private_feature - torch.mean(private_feature, dim=0)
            
            correlation_matrix = torch.mm(torch.transpose(share_features, 0, 1), private_features)
            diff_loss = torch.mean(torch.pow(correlation_matrix, 2)) * 0.005
            
            loss = norm_loss - adversarial_loss * 0.001 + diff_loss
            # loss = norm_loss
            # Backward propagation

            loss.backward()
            self.optimizer.step()
            return loss.cpu().item()
        else:
            class_result, share_feature, private_feature, feature = model(input_variable, c_inputs,
                                                                          target_variable,
                                                                          teacher_forcing_ratio=teacher_forcing_ratio)
            class_loss = self.class_loss(class_result, target_variable.squeeze())
            # adversarial loss
            target = torch.ones(input_variable.size(0), dtype=torch.long)
            if self.use_cuda:
                target = target.cuda()
            adversarial_loss = self.adv_loss(feature, target)

            # different loss
            share_features = share_feature - torch.mean(share_feature, dim=0)
            private_features = private_feature - torch.mean(private_feature, dim=0)
            
            correlation_matrix = torch.mm(torch.transpose(share_features, 0, 1), private_features)
            diff_loss = torch.mean(torch.pow(correlation_matrix, 2)) * 0.005
            
            loss = class_loss - adversarial_loss * 0.001 + diff_loss
            # loss = class_loss

            loss.backward()
            self.optimizer.step()

            return loss.cpu().item()

    def _pre_train_epochs(self, data, model, n_epochs, start_epoch, start_step,
                          dev_data=None, teacher_forcing_ratio=0.6):
        log = self.logger

        print_loss_total = 0  # Reset every print_every
        epoch_loss_total = 0  # Reset every epoch

        epoch_loss = []
        batch_loss = []

        steps_per_epoch = len(data)
        total_steps = steps_per_epoch * n_epochs

        step = start_step
        step_elapsed = 0
        for epoch in range(start_epoch, n_epochs + 1):
            log.info("Epoch: %d, Step: %d" % (epoch, step))

            model.train(True)
            if self.config.model_type == 'word':
                for input_variables, target_variables in data:
                    step += 1
                    step_elapsed += 1

                    if self.use_cuda:
                        input_variables = input_variables.cuda()
                        target_variables = target_variables.cuda()

                    loss = self._train_batch(input_variables,
                                             None, target_variables,
                                             model, teacher_forcing_ratio, 'pre_train')
                    # Record average loss
                    print_loss_total += loss
                    epoch_loss_total += loss
                    batch_loss.append(loss)

                    if step % self.print_every == 0 and step_elapsed > self.print_every:
                        print_loss_avg = print_loss_total / self.print_every
                        print_loss_total = 0
                        log_msg = 'Progress: %d%%, Train %s: %.4f' % (
                            step / total_steps * 100,
                            self.loss.name,
                            print_loss_avg)
                        log.info(log_msg)
            elif self.config.model_type == 'elmo':
                for input_variables, c_inputs, target_variables in data:
                    step += 1
                    step_elapsed += 1

                    if self.use_cuda:
                        input_variables = input_variables.cuda()
                        c_inputs = c_inputs.cuda()
                        target_variables = target_variables.cuda()

                    loss = self._train_batch(input_variables,
                                             c_inputs, target_variables,
                                             model, teacher_forcing_ratio, 'pre_train')
                    # Record average loss
                    print_loss_total += loss
                    epoch_loss_total += loss
                    batch_loss.append(loss)

                    if step % self.print_every == 0 and step_elapsed > self.print_every:
                        print_loss_avg = print_loss_total / self.print_every
                        print_loss_total = 0
                        log_msg = 'Progress: %d%%, Train %s: %.4f' % (
                            step / total_steps * 100,
                            self.loss.name,
                            print_loss_avg)
                        log.info(log_msg)

            if step_elapsed == 0:
                continue

            epoch_loss_avg = epoch_loss_total / min(steps_per_epoch, step - start_step)
            epoch_loss.append(epoch_loss_avg)
            epoch_loss_total = 0
            log_msg = "Finished epoch %d: Normalization Train %s: %.4f" % (epoch, self.loss.name, epoch_loss_avg)
            if dev_data is not None:
                dev_loss, accuracy = self.evaluator.pre_train_evaluator(model, dev_data)
                # self.optimizer.update(dev_loss, epoch)
                #
                # if dev_loss < min_loss:
                #     Checkpoint(model=model,
                #                optimizer=self.optimizer,
                #                epoch=epoch, step=step,
                #                input_vocab=data.fields[seq2seq.norm_src_field_name].vocab,
                #                output_vocab=data.fields[seq2seq.norm_tgt_field_name].vocab).save(self.expt_dir)
                #     log.info('Model saved.')
                log_msg += ", Dev %s: %.4f, Normlization Accuracy: %.4f" % (self.loss.name, dev_loss, accuracy)

                model.train(mode=True)
            # else:
            #     self.optimizer.update(epoch_loss_avg, epoch)

            log.info(log_msg)
        visual_loss(batch_loss, os.path.join(self.expt_dir, 'visual/norm/n_batch_loss_{}.jpg'.format(time.strftime("%m-%d_%H-%M-%S"))))
        # visual_loss(epoch_loss, os.path.join(self.expt_dir, 'visual/norm/n_epoch_loss_{}.jpg'.format(time.strftime("%m-%d_%H-%M-%S"))))

    def _train_epoches(self, data, model, n_epochs, start_epoch, start_step,
                       dev_data=None, test_data=None, teacher_forcing_ratio=0):
        log = self.logger

        print_loss_total = 0  # Reset every print_every
        epoch_loss_total = 0  # Reset every epoch

        batch_loss = []
        epoch_loss = []

        steps_per_epoch = len(data)
        total_steps = steps_per_epoch * n_epochs

        step = start_step
        step_elapsed = 0
        min_f1 = 0
        for epoch in range(start_epoch, n_epochs + 1):
            log.info("Epoch: %d, Step: %d" % (epoch, step))

            model.train(True)
            if self.config.model_type == 'word':
                for input_variables, target_variables in data:
                    step += 1
                    step_elapsed += 1

                    if self.use_cuda:
                        input_variables = input_variables.cuda()
                        target_variables = target_variables.cuda()

                    loss = self._train_batch(input_variables,
                                             None, target_variables,
                                             model, teacher_forcing_ratio, 'train')
                    # Record average loss
                    print_loss_total += loss
                    epoch_loss_total += loss
                    batch_loss.append(loss)

                    if step % self.print_every == 0 and step_elapsed > self.print_every:
                        print_loss_avg = print_loss_total / self.print_every
                        print_loss_total = 0
                        log_msg = 'Progress: %d%%, Classification Train average loss/%d: %.4f' % (
                            step / total_steps * 100,
                            self.print_every,
                            print_loss_avg)
                        log.info(log_msg)
            elif self.config.model_type == 'elmo':
                for input_variables, c_inputs, target_variables in data:
                    step += 1
                    step_elapsed += 1

                    if self.use_cuda:
                        input_variables = input_variables.cuda()
                        c_inputs = c_inputs.cuda()
                        target_variables = target_variables.cuda()

                    loss = self._train_batch(input_variables,
                                             c_inputs, target_variables,
                                             model, teacher_forcing_ratio, 'train')
                    # Record average loss
                    print_loss_total += loss
                    epoch_loss_total += loss
                    batch_loss.append(loss)

                    if step % self.print_every == 0 and step_elapsed > self.print_every:
                        print_loss_avg = print_loss_total / self.print_every
                        print_loss_total = 0
                        log_msg = 'Progress: %d%%, Classification Train average loss/%d: %.4f' % (
                            step / total_steps * 100,
                            self.print_every,
                            print_loss_avg)
                        log.info(log_msg)

            if step_elapsed == 0:
                continue

            epoch_loss_avg = epoch_loss_total / min(steps_per_epoch, step - start_step)
            epoch_loss.append(epoch_loss_avg)
            epoch_loss_total = 0
            log_msg = "Finished epoch %d: Train epoch average loss: %.4f" % (epoch, epoch_loss_avg)
            if dev_data is not None:
                pred_result, gold_result = self.evaluator.evaluate(model, dev_data)
                # self.optimizer.update(dev_loss, epoch)

                f1 = metrics.f1_score(gold_result, pred_result, average='weighted')
                p = metrics.precision_score(gold_result, pred_result, average='weighted')
                r = metrics.recall_score(gold_result, pred_result, average='weighted')
                acc = metrics.accuracy_score(gold_result, pred_result)

                if f1 > min_f1:
                    Checkpoint(model=model,
                               optimizer=self.optimizer,
                               epoch=epoch, step=step).save(self.expt_dir)
                # log_msg += ", Dev %s: %.4f, Normlization Accuracy: %.4f" % (self.loss.name, dev_loss, accuracy)
                log_msg += '\nAggressive language detection ,weighted_f1: {:.4f}, p: {:.4f}, r: {:.4f}, ' \
                           'acc: {:.4f}'.format(f1, p, r, acc)
                model.train(mode=True)
            if test_data is not None:
                fb_pred_result, fb_gold_result = self.evaluator.evaluate(model, test_data[0])
                fb_f1 = metrics.f1_score(fb_gold_result, fb_pred_result, average='weighted')
                fb_p = metrics.precision_score(fb_gold_result, fb_pred_result, average='weighted')
                fb_r = metrics.recall_score(fb_gold_result, fb_pred_result, average='weighted')
                fb_acc = metrics.accuracy_score(fb_gold_result, fb_pred_result)
                log_msg += '\nFacebook test dataset, weighted_f1: {:.4f}, p: {:.4f}, r: {:.4f}, acc: {:.4f}'.format(fb_f1, fb_p, fb_r, fb_acc)

                tw_pred_result, tw_gold_result = self.evaluator.evaluate(model, test_data[1])
                tw_f1 = metrics.f1_score(tw_gold_result, tw_pred_result, average='weighted')
                tw_p = metrics.precision_score(tw_gold_result, tw_pred_result, average='weighted')
                tw_r = metrics.recall_score(tw_gold_result, tw_pred_result, average='weighted')
                tw_acc = metrics.accuracy_score(tw_gold_result, tw_pred_result)
                log_msg += '\nTwitter test dataset, weighted_f1: {:.4f}, p: {:.4f}, r: {:.4f}, acc: {:.4f}'.format(tw_f1, tw_p, tw_r, tw_acc)

                if fb_f1 > self.best_fb_f1:
                    pickle.dump([fb_pred_result, fb_gold_result],
                                open(os.path.join(self.expt_dir, 'best_fb_f1.pickle'), 'wb'))
                    self.best_fb_f1 = fb_f1
                    print('best_fb_f1: {} is stored'.format(self.best_fb_f1))
                if tw_f1 > self.best_tw_f1:
                    pickle.dump([tw_pred_result, tw_gold_result],
                                open(os.path.join(self.expt_dir, 'best_tw_f1.pickle'), 'wb'))
                    self.best_tw_f1 = tw_f1
                    print('best_tw_f1: {} is stored'.format(self.best_tw_f1))

                model.train(mode=True)
            else:
                pass
                # self.optimizer.update(epoch_loss_avg, epoch)

            log.info(log_msg)
        visual_loss(batch_loss, os.path.join(self.expt_dir,
                                             'visual/class/c_batch_loss_{}.jpg'.format(time.strftime("%m-%d_%H-%M-%S"))))
        # visual_loss(epoch_loss, os.path.join(self.expt_dir,
        #                                      'visual/class/c_epoch_loss_{}.jpg'.format(time.strftime("%m-%d_%H-%M-%S"))))

    def train(self, model, data, round1=2, round2=10, norm_epochs=3, class_epochs=3,
              resume=False, dev_data=None, test_data=None,
              optimizer=None, teacher_forcing_ratio=0, lr=0.003):
        """ Run training for a given model.
        Args:
            model (seq2seq.models): model to run training on, if `resume=True`, it would be
               overwritten by the model loaded from the latest checkpoint.
            data (seq2seq.dataset.dataset.Dataset): dataset object to train on
            norm_epochs (int, optional): number of epochs to run (default 5)
            resume(bool, optional): resume training with the latest checkpoint, (default False)
            dev_data (seq2seq.dataset.dataset.Dataset, optional): dev Dataset (default None)
            optimizer (seq2seq.optim.Optimizer, optional): optimizer for training
               (default: Optimizer(pytorch.optim.Adam, max_grad_norm=5))
            teacher_forcing_ratio (float, optional): teaching forcing ratio (default 0)
        Returns:
            model (seq2seq.models): trained model.
        """
        # If training is set to resume
        if resume:
            latest_checkpoint_path = Checkpoint.get_latest_checkpoint(self.expt_dir)
            resume_checkpoint = Checkpoint.load(latest_checkpoint_path)
            model = resume_checkpoint.model
            self.optimizer = resume_checkpoint.optimizer

            # A walk around to set optimizing parameters properly
            resume_optim = self.optimizer.optimizer
            defaults = resume_optim.param_groups[0]
            defaults.pop('params', None)
            defaults.pop('initial_lr', None)
            self.optimizer.optimizer = resume_optim.__class__(model.parameters(), **defaults)

            start_epoch = resume_checkpoint.epoch
            step = resume_checkpoint.step
        else:
            start_epoch = 1
            step = 0
            # norm_parameters = list(map(id, model.decoder.parameters()))
            # class_parameters = list(map(id, model.classification.parameters()))
            # base_params = filter(lambda p: id(p) not in class_parameters, model.parameters())
            # self.logger.info(norm_parameters)
            # self.logger.info(class_parameters)
            # self.logger.info(base_params)
            # ignored_params = list(map(id, model.encoder.elmo._scalar_mixes[0].parameters()))
            # base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
            if optimizer is None:
                # optimizer = Optimizer(optim.Adam([{'params': base_params},
                #                                   {'params': model.encoder.elmo._scalar_mixes[0].parameters(), 'lr':1e-2}],
                #                                  lr=lr, weight_decay=1e-4), max_grad_norm=5)
                optimizer = Optimizer(optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4), max_grad_norm=5)
            self.optimizer = optimizer

        self.logger.info("Optimizer: %s, Scheduler: %s" % (self.optimizer.optimizer, self.optimizer.scheduler))

        self.logger.info("{} rounds of per training data, {} epochs of per round, Starting.....".format(round1, norm_epochs))
        for i in range(round1):
            self.logger.info("Round: {}".format(i))
            self._pre_train_epochs(data[0], model, norm_epochs, 1, 0, dev_data=dev_data[0],
                                   teacher_forcing_ratio=teacher_forcing_ratio)

            self._train_epoches(data[1], model, class_epochs,
                                1, 0, dev_data=dev_data[1], test_data=test_data,
                                teacher_forcing_ratio=teacher_forcing_ratio)
        self.logger.info("{} rounds of per training data, 1 epoch of per round, Starting......".format(round2))
        for i in range(round2):
            self.logger.info("Round : {}".format(i))
            self._pre_train_epochs(data[0], model, 1, 1, 0, dev_data=dev_data[0],
                                   teacher_forcing_ratio=teacher_forcing_ratio)

            self._train_epoches(data[1], model, 1, 1, 0, dev_data=dev_data[1], test_data=test_data,
                                teacher_forcing_ratio=teacher_forcing_ratio)

        self.logger.info('best_fb_f1: {}, best_tw_f1: {}'.format(self.best_fb_f1, self.best_tw_f1))

        return model
