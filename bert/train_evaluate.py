# coding=utf-8

import numpy as np
from tqdm import tqdm
import torch
from tensorboardX import SummaryWriter
import time
import math

from Utils.utils import classifiction_metric


def train(epoch_num, n_gpu, model, train_dataloader,
          optimizer, criterion, gradient_accumulation_steps, device, label_list,
          data_type='norm', adv_weight=0.05, diff_weight=0.01):
    """ 模型训练过程
    Args:
        epoch_num: epoch 数量
        n_gpu: 使用的 gpu 数量
        train_dataloader: 训练数据的Dataloader
        optimizer: 优化器
        criterion： 损失函数定义
        gradient_accumulation_steps: 梯度积累
        device: 设备，cuda， cpu
        label_list: 分类的标签数组
        data_type:
        adv_weight:
        diff_weight:
    """

    epoch_loss = 0.0
    global_step = 0

    for epoch in range(int(epoch_num)):

        print(f'---------------- Epoch: {epoch + 1:02} ----------')

        try:
            with tqdm(train_dataloader, desc='Iteration') as t:
                for step, batch in enumerate(t):
                    model.train()

                    batch = tuple(t.to(device) for t in batch)
                    _, input_ids, input_mask, segment_ids, label_ids = batch

                    if data_type == 'norm':
                        logits, share_feature, pri_feature, feature = model(input_ids, segment_ids, input_mask,
                                                                            labels=label_ids[:, :-1])
                        output_dim = logits.shape[-1]

                        logits = logits.contiguous().view(-1, output_dim)
                        label_ids = label_ids[:, 1:].contiguous().view(-1)

                        loss = criterion(logits, label_ids)

                        target = torch.ones(input_ids.size(0), dtype=torch.long).to(device)

                        adv_loss = criterion(feature, target)
                        # normalization different loss
                        batch_size = share_feature.size(0)
                        share_feature = share_feature.contiguous().view(batch_size, -1)
                        pri_feature = pri_feature.contiguous().view(batch_size, -1)
                        share_features = share_feature - torch.mean(share_feature, dim=0)
                        pri_feature = pri_feature - torch.mean(pri_feature, dim=0)

                        correlation_matrix = torch.mm(share_features, torch.transpose(pri_feature, 0, 1))
                        diff_loss = torch.mean(torch.pow(correlation_matrix, 2))

                        loss = loss + adv_loss * adv_weight + diff_loss * diff_weight
                    else:
                        logits, share_feature, pri_feature, feature = model(input_ids, segment_ids, input_mask,
                                                                            data_type=data_type)
                        loss = criterion(logits.view(-1, len(label_list)), label_ids.view(-1))

                        target = torch.zeros(input_ids.size(0), dtype=torch.long).to(device)
                        adv_loss = criterion(feature, target)

                        # classification different loss
                        batch_size = share_feature.size(0)
                        share_feature = share_feature.contiguous().view(batch_size, -1)
                        pri_feature = pri_feature.contiguous().view(batch_size, -1)
                        share_features = share_feature - torch.mean(share_feature, dim=0)
                        pri_feature = pri_feature - torch.mean(pri_feature, dim=0)

                        correlation_matrix = torch.mm(share_features, torch.transpose(pri_feature, 0, 1))
                        diff_loss = torch.mean(torch.pow(correlation_matrix, 2))

                        loss = loss + adv_loss * adv_weight + diff_loss * diff_weight

                    """ 修正 loss """
                    if n_gpu > 1:
                        loss = loss.mean()  # mean() to average on multi-gpu.
                    if gradient_accumulation_steps > 1:
                        loss = loss / gradient_accumulation_steps

                    loss.backward()
                    if (step + 1) % gradient_accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                        global_step += 1
                    # optimizer.step()
                    epoch_loss += loss.item()
        except KeyboardInterrupt:
            t.close()
            raise
        t.close()
        # if step % 200 == 0:
        #     print('train loss {} : {}'.format(step, loss.item()))
    avg_loss = epoch_loss / len(train_dataloader)
    print(f'{data_type} Train Loss: {avg_loss:.3f} | Train PPL: {math.exp(avg_loss):7.3f}')


def evaluate(model, dataloader, criterion, device, label_list, data_type='norm', adv_weight=0.05, diff_weight=0.01):
    model.eval()

    src_label = []
    pred_label = []
    gold_label = []

    epoch_loss = 0
    step = 0
    try:
        with tqdm(dataloader, desc='Eval') as t:
            for _, input_ids, input_mask, segment_ids, label_ids in t:
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                label_ids = label_ids.to(device)

                with torch.no_grad():
                    if data_type == 'norm':
                        logits, share_feature, pri_feature, feature = model(input_ids, segment_ids, input_mask,
                                                                            labels=label_ids[:, :-1])
                        output_dim = logits.shape[-1]
                        batch_size = logits.shape[0]

                        logits = logits.contiguous().view(-1, output_dim)
                        label_ids = label_ids[:, 1:].contiguous().view(-1)

                        pred_label.extend(torch.max(logits, dim=1)[1].view(batch_size, -1).cpu().numpy().tolist())
                        gold_label.extend(label_ids.squeeze().view(batch_size, -1).cpu().numpy().tolist())
                        src_label.extend(input_ids[:, 1:].squeeze().view(batch_size, -1).cpu().numpy().tolist())

                        loss = criterion(logits, label_ids)

                        target = torch.ones(input_ids.size(0), dtype=torch.long).to(device)

                        adv_loss = criterion(feature, target)
                        # normalization different loss
                        batch_size = share_feature.size(0)
                        share_feature = share_feature.contiguous().view(batch_size, -1)
                        pri_feature = pri_feature.contiguous().view(batch_size, -1)
                        share_features = share_feature - torch.mean(share_feature, dim=0)
                        pri_feature = pri_feature - torch.mean(pri_feature, dim=0)

                        correlation_matrix = torch.mm(share_features, torch.transpose(pri_feature, 0, 1))
                        diff_loss = torch.mean(torch.pow(correlation_matrix, 2))

                        loss = loss + adv_loss * adv_weight + diff_loss * diff_weight
                    else:
                        logits, share_feature, pri_feature, feature = model(input_ids, segment_ids, input_mask,
                                                                            data_type=data_type)

                        pred_label.extend(torch.max(logits, dim=1)[1].cpu().numpy().tolist())
                        if type(label_ids.squeeze().cpu().numpy().tolist()) == int:
                            gold_label.append(label_ids.squeeze().cpu().numpy().tolist())
                        else:
                            gold_label.extend(label_ids.squeeze().cpu().numpy().tolist())

                        loss = criterion(logits.view(-1, len(label_list)), label_ids.view(-1))

                        target = torch.zeros(input_ids.size(0), dtype=torch.long).to(device)
                        adv_loss = criterion(feature, target)

                        # classification different loss
                        batch_size = share_feature.size(0)
                        share_feature = share_feature.contiguous().view(batch_size, -1)
                        pri_feature = pri_feature.contiguous().view(batch_size, -1)
                        share_features = share_feature - torch.mean(share_feature, dim=0)
                        pri_feature = pri_feature - torch.mean(pri_feature, dim=0)

                        correlation_matrix = torch.mm(share_features, torch.transpose(pri_feature, 0, 1))
                        diff_loss = torch.mean(torch.pow(correlation_matrix, 2))

                        loss = loss + adv_loss * adv_weight + diff_loss * diff_weight

                epoch_loss += loss.mean().item()
    except KeyboardInterrupt:
        t.close()
        raise
    # except RuntimeError:
    #     t.close()
    #     raise
    # finally:
    #     t.close()
    t.close()
        # if step % 200 == 0:
        #     print('valid loss {} : {}'.format(step, loss.item()))
        # step += 1

    return epoch_loss / len(dataloader), pred_label, gold_label, src_label


def evaluate_save(model, dataloader, criterion, device, label_list):
    model.eval()

    all_preds = np.array([], dtype=int)
    all_labels = np.array([], dtype=int)
    all_idxs = np.array([], dtype=int)

    epoch_loss = 0

    for idxs, input_ids, input_mask, segment_ids, label_ids in tqdm(dataloader, desc="Eval"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask, labels=None)
        loss = criterion(logits.view(-1, len(label_list)), label_ids.view(-1))

        preds = logits.detach().cpu().numpy()
        outputs = np.argmax(preds, axis=1)
        all_preds = np.append(all_preds, outputs)

        label_ids = label_ids.to('cpu').numpy()
        all_labels = np.append(all_labels, label_ids)

        idxs = idxs.detach().cpu().numpy()
        all_idxs = np.append(all_idxs, idxs)

        epoch_loss += loss.mean().item()

    acc, report, auc = classifiction_metric(all_preds, all_labels, label_list)
    return epoch_loss / len(dataloader), acc, report, auc, all_idxs, all_labels, all_preds