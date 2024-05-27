import time
import os
import logging

import torch
import torch.nn.functional as F
from .eval import eval, eval_str

def dgl_train(train_loader, device, model, optimizer):

    train_total_loss = 0.0
    train_total_correct = 0

    num_graphs = 0

    for batch_graphs, batch_labels in train_loader:

        batch_graphs = batch_graphs.to(device)
        batch_x = batch_graphs.ndata['feat'].to(device)
        batch_e = batch_graphs.edata['feat'].to(device)
        batch_labels = batch_labels.to(device)
        optimizer.zero_grad()

        if model.name in ["PRGNN", "LiNet"]:
            batch_scores, score1, score2 = model.forward(batch_graphs, batch_x, batch_e)
            loss = model.loss(batch_scores, batch_labels, score1, score2)
        else:
            batch_scores = model.forward(batch_graphs, batch_x, batch_e)
            loss = model.loss(batch_scores, batch_labels)

        loss.backward()
        optimizer.step()

        y_pred = batch_scores.data.argmax(dim=1)

        train_total_loss += loss.item() * batch_labels.size(0)
        train_total_correct += torch.sum(y_pred == batch_labels).item()

        num_graphs += batch_labels.size(0)

    train_loss = train_total_loss / num_graphs
    train_accu = train_total_correct / num_graphs

    return model, train_loss, train_accu

def pyg_train(train_loader, device, model, optimizer):
    
    train_total_loss = 0.0
    train_total_correct = 0

    num_graphs = 0

    for data in train_loader:
        data.to(device)
        optimizer.zero_grad()
        labels = data.y

        y_true = labels.view(-1)

        output = model(data)
        if model.name in ['BrainGNN']:
            loss = model.loss(output, y_true)
        else:
            loss = F.cross_entropy(output, y_true)

        loss.backward()
        optimizer.step()

        y_pred = output.data.argmax(dim=1)

        train_total_loss += loss.item() * labels.size(0)
        train_total_correct += torch.sum(y_pred == labels).item()

        num_graphs += labels.size(0)

    train_loss = train_total_loss / num_graphs
    train_accu = train_total_correct / num_graphs

    return model, train_loss, train_accu

def train_model(model, optimizer, scheduler, device, train_loader, val_loader, test_loader, fold, save_path, params, writer=None):
    
    best_val_accu = 0.0

    best_train_eval = None
    best_val_eval = None
    best_test_eval = None
    best_epoch = None

    for epoch in range(params['epochs']):
        model.train()

        t = time.time()

        if model.name in ['BrainGNN', 'BNTF']:
            model, train_loss, train_accu = pyg_train(train_loader, device, model, optimizer)
        else:
            model, train_loss, train_accu = dgl_train(train_loader, device, model, optimizer)

        elapsed = time.time() - t

        model.eval()

        # (loss, accu, precision, recall, f1)
        train_eval = eval(model, train_loader, device)
        val_eval = eval(model, val_loader, device)
        test_eval = eval(model, test_loader, device)

        scheduler.step(val_eval[0])

        eval_results = ((train_loss, train_accu), val_eval, test_eval)
        write_epoch(writer, train_eval, val_eval, test_eval, optimizer, epoch)

        selected = False
        if val_eval[1] >= best_val_accu and epoch > 10: # optimizer.param_groups[0]['lr'] <= params['init_lr'] / 2:
            best_val_accu = val_eval[1]
            best_train_eval = (train_loss, train_accu)
            best_val_eval = val_eval
            best_test_eval = test_eval
            best_epoch = epoch
            selected = True

            if save_path is not None:
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                state = {
                    'model': model.state_dict(),
                    'eval': [best_train_eval, best_val_eval, best_test_eval],
                    'epoch': best_epoch,
                    'elapsed': elapsed,
                    'timestamp': t
                }

                torch.save(state, f'{save_path}/fold_{fold}_best_model.pth')

        logging.info(eval_str(eval_results, epoch, elapsed, selected, optimizer.param_groups[0]['lr']))

        if optimizer.param_groups[0]['lr'] < params['min_lr']:
            logging.info(f"Early stopped at lr = {optimizer.param_groups[0]['lr']}")
            return best_epoch, (best_train_eval, best_val_eval, best_test_eval)

    return best_epoch, (best_train_eval, best_val_eval, best_test_eval)

def write_epoch(writer, train_eval, val_eval, test_eval, optimizer, epoch):

    for eval, eval_str in zip([train_eval, val_eval, test_eval], ['train', 'val', 'test']):
        for measure, m_str in zip(eval, ['loss', 'accuracy', 'precision', 'recall', 'f1']):
            writer.add_scalar(f'{m_str.capitalize()}/{eval_str}', measure, epoch)        

    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
