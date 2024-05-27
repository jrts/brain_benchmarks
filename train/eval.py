import torch
import torch.nn.functional as F
import numpy as np

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def eval(model, loader, device):

    y_true = torch.tensor([], dtype=torch.long, device=device)
    all_outputs = torch.tensor([], device=device)

    train_total_loss = 0.0
    num_graphs = 0
    
    with torch.no_grad():
        for data in loader:

            if model.name in ['BrainGNN', 'BNTF']:
                data = data.to(device)
                batch_scores = model.forward(data)
                batch_labels = data.y.view(-1)
                if model.name in ['BrainGNN']:
                    loss = model.loss(batch_scores, batch_labels)
                else:
                    loss = F.cross_entropy(batch_scores, batch_labels)
            else:
                batch_graphs, batch_labels = data
                batch_graphs = batch_graphs.to(device)
                batch_x = batch_graphs.ndata['feat'].to(device)
                batch_e = batch_graphs.edata['feat'].to(device)
                batch_labels = batch_labels.to(device)

                if model.name in ["PRGNN", "LiNet"]:
                    batch_scores, score1, score2 = model.forward(batch_graphs, batch_x, batch_e)
                    loss = model.loss(batch_scores, batch_labels, score1, score2)
                else:
                    batch_scores = model.forward(batch_graphs, batch_x, batch_e)
                    loss = model.loss(batch_scores, batch_labels)

            y_true = torch.cat((y_true, batch_labels), 0)
            all_outputs = torch.cat((all_outputs, batch_scores), 0)

            train_total_loss += loss.item() * batch_labels.size(0)
            num_graphs += batch_labels.size(0)

    y_true = y_true.cpu().numpy()
    y_pred = all_outputs.data.argmax(dim=1).cpu().numpy()

    accu = accuracy_score(y_true, y_pred)
    num_classes = len(np.unique(y_true))

    if num_classes > 2:
        average = 'micro'
    else:
        average = 'binary'
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=average, zero_division=0.0)
    
    return train_total_loss / num_graphs, accu, precision, recall, f1


def eval_str(results, epoch=None, elapsed=None, selected=False, lr=None):
    train_eval, val_eval, test_eval = results

    epoch_str = '' if epoch is None else f'epoch {epoch:05d} - '

    lr_str = '' if lr is None else f'lr: {lr:.4e}, '

    results_str = f'train_loss: {train_eval[0]:.3f}, train_accu: {train_eval[1]:.3f}, ' \
                  f'val_loss: {val_eval[0]:.3f}, val_accu: {val_eval[1]:.3f}, ' \
                  f'test_loss: {test_eval[0]:.3f}, test_accu: {test_eval[1]:.3f}, test_prec: {test_eval[2]:.3f}, test_recall: {test_eval[3]:.3f}'
    
    elapsed_str = '' if elapsed is None else f', time: {elapsed:.3f}s'
    selected_symbol = '' if not selected else ' <'

    str = ''.join([epoch_str, lr_str, results_str, elapsed_str, selected_symbol]).capitalize()

    return str


def eval_results_dict(results):
    results_dict = {}
    # 'default' is for `test_accu``
    keys = ['train_loss', 'train_accu', 
            'val_loss', 'val_accu', 'val_precision', 'val_recall', 'val_f1', 
            'test_loss', 'default', 'test_precision', 'test_recall', 'test_f1']
    
    results_list = results
    if isinstance(results, tuple):
        results_list = list(sum(results_list, ()))

    for k, v in zip(keys, results_list):
        results_dict[k] = v

    return results_dict

def append_epoch_results(results_list, epoch_results):
    epoch_results_list = epoch_results
    if isinstance(epoch_results_list, tuple):
        epoch_results_list = list(sum(epoch_results_list, ()))

    if results_list is None:
        results_list = []
        for v in epoch_results_list:
            results_list.append([v])
    else:
        for l, v in zip(results_list, epoch_results_list):
            l.append(v)

    return results_list