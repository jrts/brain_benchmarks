import numpy as np
import os
import random
import argparse, json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as PYGDataLoader
from torch.utils.tensorboard import SummaryWriter
import logging
from datetime import datetime

from train.eval import append_epoch_results, eval_str
from train.train_model import train_model

from nets.load_net import gnn_model # import GNNs
from data.data import LoadData # import dataset


def view_model_param(MODEL_NAME, net_params):
    model = gnn_model(MODEL_NAME, net_params)
    total_param = 0
    logging.info("MODEL DETAILS:")
    for param in model.parameters():
        total_param += np.prod(list(param.data.size()))
    logging.info(f'MODEL/Total parameters: {MODEL_NAME}, {total_param}')
    return total_param

def train_val_pipeline(dataset, model_name, params, net_params, save_path, timestamp):

    logging.info(f'Model saved at: {save_path}')

    if torch.cuda.is_available():
        if net_params.get('gpu_id') is not None:
            device = torch.device(f"cuda:{net_params['gpu_id']}")
        else:
            device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    logging.info(f'device: {device}')
    logging.info(params)
    logging.info(net_params)

    net_params['device'] = device
    print(net_params['device'])

    net_params['total_param'] = view_model_param(model_name, net_params)

    best_results = None
    convergence_epochs = []

    folds = list(range(10))

    for fold in folds:

        # setting seeds
        random.seed(params['seed'])
        np.random.seed(params['seed'])
        torch.manual_seed(params['seed'])
        if device.type == 'cuda':
            torch.cuda.manual_seed(params['seed'])

        logging.info(f'\nFold {fold}:')

        if model_name in ['BrainGNN', 'PSCR', 'BNTF']:
            train_indices, val_indices, test_indices = dataset.cv_splits[fold]
            train_set, val_set, test_set = dataset[train_indices], dataset[val_indices], dataset[test_indices]
        else:
            train_set, val_set, test_set = dataset.train[fold], dataset.val[fold], dataset.test[fold]

        logging.info(f"Training Graphs: {len(train_set)}")
        logging.info(f"Validation Graphs: {len(val_set)}")
        logging.info(f"Test Graphs: {len(test_set)}")
        logging.info(f"Number of Classes: {net_params['n_classes']}")

        drop_last = True if model_name in ['DiffPool', 'ContrastPool'] else False

        if model_name in ['BrainGNN', 'PSCR', 'BNTF']:
            train_loader = PYGDataLoader(train_set, batch_size=params['batch_size'], shuffle=True, drop_last=drop_last, num_workers=1)
            val_loader = PYGDataLoader(val_set, batch_size=params['batch_size'], shuffle=False, drop_last=drop_last, num_workers=1)
            test_loader = PYGDataLoader(test_set, batch_size=params['batch_size'], shuffle=False, drop_last=drop_last, num_workers=1)
        else:
            train_loader = DataLoader(train_set, batch_size=params['batch_size'], shuffle=True, collate_fn=dataset.collate, drop_last=drop_last, num_workers=1)
            val_loader = DataLoader(val_set, batch_size=params['batch_size'], shuffle=False, collate_fn=dataset.collate, drop_last=drop_last, num_workers=1)
            test_loader = DataLoader(test_set, batch_size=params['batch_size'], shuffle=False, collate_fn=dataset.collate, drop_last=drop_last, num_workers=1)

        model = gnn_model(model_name, net_params)
        model = model.to(device)

        if model_name in ['DiffPool', 'ContrastPool']:
            model.cal_contrast(train_set, device)

        optimizer = optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                            factor=params['lr_reduce_factor'],
                                                            patience=params['lr_schedule_patience'],
                                                            verbose=True)
        
        if fold == 0:
            logging.info(model)
            logging.info(optimizer)
            logging.info(scheduler)

        log_dir = f'{save_path}/run_{fold}/'
        writer = SummaryWriter(log_dir=log_dir)

        epoch, epoch_best_results = train_model(model, optimizer, scheduler, device, 
                                                train_loader, val_loader, test_loader,
                                                fold, save_path, params, writer)

        logging.info(f'\nBest results on Fold {fold}:\n{eval_str(epoch_best_results, epoch)}')
    
        best_results = append_epoch_results(best_results, epoch_best_results)
        convergence_epochs.append(epoch)
        print(epoch_best_results)

        exp_str = f"{dataset.name}_{model_name}_{timestamp}_{fold}"

        results_dict = {
            'hparam/best_test_accuracy': epoch_best_results[2][1],
            'hparam/best_test_precision': epoch_best_results[2][2],
            'hparam/best_test_recall': epoch_best_results[2][3],
            'hparam/best_test_f1': epoch_best_results[2][4],
            'hparam/best_val_accuracy': epoch_best_results[1][1],
            'hparam/best_val_precision': epoch_best_results[1][2],
            'hparam/best_val_recall': epoch_best_results[1][3],
            'hparam/best_val_f1': epoch_best_results[1][4],
        }
        writer.add_hparams({'L': net_params['L']}, results_dict, run_name=exp_str)
        writer.close()

    logging.info(f"\nTest set accuracy:\n[{', '.join([f'{v:.4f}' for v in best_results[8]])}]")
    logging.info(f'Test set avg accuracy: {np.mean(best_results[8]):.4f} ± {np.std(best_results[8]):.4f}')
    logging.info(f'Test set avg precision: {np.mean(best_results[9]):.4f} ± {np.std(best_results[9]):.4f}')
    logging.info(f'Test set avg recall: {np.mean(best_results[10]):.4f} ± {np.std(best_results[10]):.4f}')
    logging.info(f'Test set avg f1-score: {np.mean(best_results[11]):.4f} ± {np.std(best_results[11]):.4f}')
    logging.info(f'epochs of best models: {convergence_epochs}')


def main():
    """
        USER CONTROLS
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_base_url', help="Please give a config.json file with training/model/data/param details", default='./configs')
    parser.add_argument('--save_base_url', help="Please give a value for out_dir", default='../baseline_results')
    parser.add_argument('--threshold', help="Please give a threshold to drop edge", default='20_80')
    parser.add_argument('--dataset_base_url', default='../datasets')
    parser.add_argument('--splits_base_url', default='../cv_splits')
    parser.add_argument('--seed', help="Please give a value for seed", default=123)
    parser.add_argument('--gpu_id', help="Please give a value for gpu id")
    parser.add_argument('--model', help="Please give a value for model name")
    parser.add_argument('--dataset', help="Please give a value for dataset name")
    parser.add_argument('--epochs', help="Please give a value for epochs")
    parser.add_argument('--batch_size', help="Please give a value for batch_size")
    parser.add_argument('--init_lr', help="Please give a value for init_lr")
    parser.add_argument('--lr_reduce_factor', help="Please give a value for lr_reduce_factor")
    parser.add_argument('--lr_schedule_patience', help="Please give a value for lr_schedule_patience")
    parser.add_argument('--min_lr', help="Please give a value for min_lr")
    parser.add_argument('--weight_decay', help="Please give a value for weight_decay")
    parser.add_argument('--L', help="Please give a value for L")
    parser.add_argument('--hidden_dim', help="Please give a value for hidden_dim", type=lambda x: (None if str(x).lower() == 'none' else int(x)))
    parser.add_argument('--out_dim', help="Please give a value for out_dim")
    parser.add_argument('--residual', help="Please give a value for residual")
    parser.add_argument('--edge_feat', help="Please give a value for edge_feat")
    parser.add_argument('--readout', help="Please give a value for readout")
    parser.add_argument('--kernel', help="Please give a value for kernel")
    parser.add_argument('--n_heads', help="Please give a value for n_heads")
    parser.add_argument('--gated', help="Please give a value for gated")
    parser.add_argument('--in_feat_dropout', help="Please give a value for in_feat_dropout")
    parser.add_argument('--dropout', help="Please give a value for dropout")
    parser.add_argument('--layer_norm', help="Please give a value for layer_norm")
    parser.add_argument('--batch_norm', help="Please give a value for batch_norm")
    parser.add_argument('--sage_aggregator', help="Please give a value for sage_aggregator")
    parser.add_argument('--data_mode', help="Please give a value for data_mode")
    parser.add_argument('--num_pool', help="Please give a value for num_pool")
    parser.add_argument('--gnn_per_block', help="Please give a value for gnn_per_block")
    parser.add_argument('--embedding_dim', help="Please give a value for embedding_dim")
    parser.add_argument('--pool_ratio', help="Please give a value for pool_ratio")
    parser.add_argument('--linkpred', help="Please give a value for linkpred")
    parser.add_argument('--cat', help="Please give a value for cat")
    parser.add_argument('--self_loop', help="Please give a value for self_loop")
    parser.add_argument('--config_dir', help="config dir")
    args = parser.parse_args()

    with open(f"{args.config_base_url}/{args.config_dir}/{args.model}.json") as f:
        config = json.load(f)
        f.close()

    MODEL_NAME = args.model
    DATASET_NAME = args.dataset

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    L = args.L if args.L is not None else config['net_params']['L']
    hidden_dim = config['net_params']['hidden_dim'] if args.hidden_dim is None else args.hidden_dim
    save_path = f"{args.save_base_url}/{MODEL_NAME}/{DATASET_NAME}_{MODEL_NAME}_{L}_{hidden_dim}_{timestamp}"

    filename = f'{save_path}/{DATASET_NAME}_{MODEL_NAME}.log'
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[
                            logging.FileHandler(filename),
                            logging.StreamHandler()
                        ])

    # parameters
    params = config['params']
    if args.seed is not None:
        params['seed'] = int(args.seed)
    if args.epochs is not None:
        params['epochs'] = int(args.epochs)

    params['config_base_url'] = args.config_base_url
    params['save_base_url'] = args.save_base_url

    if args.batch_size is not None:
        params['batch_size'] = int(args.batch_size)
    if args.init_lr is not None:
        params['init_lr'] = float(args.init_lr)
    if args.lr_reduce_factor is not None:
        params['lr_reduce_factor'] = float(args.lr_reduce_factor)
    if args.lr_schedule_patience is not None:
        params['lr_schedule_patience'] = int(args.lr_schedule_patience)
    if args.min_lr is not None:
        params['min_lr'] = float(args.min_lr)
    if args.weight_decay is not None:
        params['weight_decay'] = float(args.weight_decay)

    # network parameters
    net_params = config['net_params']
    net_params['node_num'] = int(DATASET_NAME.split('_')[1][-3:])
    net_params['gpu_id'] = args.gpu_id
    net_params['batch_size'] = params['batch_size']
    if args.L is not None:
        net_params['L'] = int(args.L)
    if args.hidden_dim is not None:
        net_params['hidden_dim'] = int(args.hidden_dim)
    if args.out_dim is not None:
        net_params['out_dim'] = int(args.out_dim)
    if args.residual is not None:
        net_params['residual'] = True if args.residual.lower()=='True' else False
    if args.edge_feat is not None:
        net_params['edge_feat'] = True if args.edge_feat.lower()=='True' else False
    if args.readout is not None:
        net_params['readout'] = args.readout
    if args.kernel is not None:
        net_params['kernel'] = int(args.kernel)
    if args.n_heads is not None:
        net_params['n_heads'] = int(args.n_heads)
    if args.gated is not None:
        net_params['gated'] = True if args.gated.lower()=='True' else False
    if args.in_feat_dropout is not None:
        net_params['in_feat_dropout'] = float(args.in_feat_dropout)
    if args.dropout is not None:
        net_params['dropout'] = float(args.dropout)
    if args.layer_norm is not None:
        net_params['layer_norm'] = True if args.layer_norm=='True' else False
    if args.batch_norm is not None:
        net_params['batch_norm'] = True if args.batch_norm=='True' else False
    if args.sage_aggregator is not None:
        net_params['sage_aggregator'] = args.sage_aggregator
    if args.data_mode is not None:
        net_params['data_mode'] = args.data_mode
    if args.num_pool is not None:
        net_params['num_pool'] = int(args.num_pool)
    if args.gnn_per_block is not None:
        net_params['gnn_per_block'] = int(args.gnn_per_block)
    if args.embedding_dim is not None:
        net_params['embedding_dim'] = int(args.embedding_dim)
    if args.pool_ratio is not None:
        net_params['pool_ratio'] = float(args.pool_ratio)
    if args.linkpred is not None:
        net_params['linkpred'] = True if args.linkpred.lower()=='True' else False
    if args.cat is not None:
        net_params['cat'] = True if args.cat=='True' else False
    if args.self_loop is not None:
        net_params['self_loop'] = True if args.self_loop.lower()=='True' else False

    dataset = LoadData(args.dataset_base_url, args.splits_base_url, 
                       DATASET_NAME, MODEL_NAME, 
                       threshold=args.threshold)
    
    if MODEL_NAME in ['BrainGNN', 'BNTF']:
        net_params['in_dim'] = dataset.num_nodes()
        net_params['edge_dim'] = dataset.edge_dim()
        net_params['n_classes'] = dataset.n_classes()
    else:
        net_params['in_dim'] = dataset.all.graph_lists[0].ndata['feat'][0].shape[0]
        net_params['edge_dim'] = dataset.train[0].graph_lists[0].edata['feat'][0].shape[0] \
            if 'feat' in dataset.train[0].graph_lists[0].edata else None
        num_classes = len(np.unique(dataset.all.graph_labels))
        net_params['n_classes'] = num_classes

    train_val_pipeline(dataset, MODEL_NAME, params, net_params, save_path, timestamp)


if __name__ == "__main__":
    main()

