# 来自GCN,改为GAT
from load_data import Data
import numpy as np
import torch
import time
from collections import defaultdict
from model import *
from torch.optim.lr_scheduler import ExponentialLR
import argparse
import setproctitle
import mlflow
from mlflow.tracking import MlflowClient
import shutil
import os
from tqdm import tqdm
import json
import copy
import random

import dgl
import dgl.function as fn

import os
# os.environ['CUDA_VISIBLE_DEVICES']='7'
import setproctitle
setproctitle.setproctitle('GAT@zzl')

device = torch.device('cuda')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Experiment:
    def __init__(self, lr, edim, batch_size, dr):
        self.lr = lr
        self.edim = edim
        self.batch_size = batch_size
        self.dr = dr
        self.num_iterations = args.num_iterations
        self.kwargs = params
        self.kwargs['device'] = device
        self.nreg=len(d.region2id)
        self.g, self.etype, self.enorm = self.build_graph()

    def build_graph(self):
        edge_data = [[x[0] for x in d.kg_data], [x[2] for x in d.kg_data]]
        g = dgl.graph((edge_data[0], edge_data[1])).to(device)

        # g = dgl.add_self_loop(g)

        etype = torch.tensor([x[1] for x in d.kg_data], dtype=torch.long, device=device)
        indegs = g.in_degrees().float().clamp(min=1)
        innorm = 1.0 / indegs
        g.ndata['xxx'] = innorm
        g.apply_edges(lambda edges: {'xxx': edges.dst['xxx']})
        enorm = g.edata.pop('xxx').to(device)
        return g, etype, enorm.view(-1, 1)

    def get_er_vocab(self, data):
        er_vocab = defaultdict(list)
        for triple in data:
            er_vocab[(triple[0], triple[1])].append(triple[2])
        return er_vocab

    def get_batch(self, er_vocab, er_vocab_pairs, idx):
        batch = er_vocab_pairs[idx:idx + self.batch_size]
        targets = torch.zeros((len(batch), len(d.ent2id)), device=device)
        for idx, pair in enumerate(batch):
            targets[idx, er_vocab[pair]] = 1.
        return torch.tensor(batch, dtype=torch.long, device=device), targets

    def train_and_eval(self):
        print('building model....')
        model = TuckER(d, self.edim, **self.kwargs)
        model = model.to(device)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        if self.dr:
            scheduler = ExponentialLR(opt, self.dr)

        er_vocab = self.get_er_vocab(d.kg_data)
        er_vocab_pairs = list(er_vocab.keys())

        best_loss = 1e10
        E_epoch, R_epoch, loss_epoch = [], [], []
        print("Starting training...")
        for it in range(1, self.num_iterations + 1):
            print('\n=============== Epoch %d Starts...===============' % it)
            start_train = time.time()
            model.train()
            losses = []
            np.random.shuffle(er_vocab_pairs)
            for j in tqdm(range(0, len(er_vocab_pairs), self.batch_size)):
                data_batch, targets = self.get_batch(er_vocab, er_vocab_pairs, j)
                h_idx = data_batch[:, 0]
                predictions = model.forward(self.g, h_idx)
                opt.zero_grad()
                loss = model.loss(predictions, targets)
                loss.backward()
                opt.step()
                losses.append(loss.item())
            if self.dr:
                scheduler.step()
            print('\nEpoch=%d, train time cost %.4fs, loss:%.8f' % (it, time.time() - start_train, np.mean(losses)))
            loss_epoch.append(np.mean(losses))
            mlflow.log_metrics({'train_time': time.time()-start_train, 'loss': loss_epoch[-1], 'current_it': it}, step=it)

            # E_out, R_out = model.E.weight, model.R.weight
            # E_epoch.append(E_out.detach().cpu().numpy())
            # R_epoch.append(R_out.detach().cpu().numpy())

            # np.savez(archive_path + 'ER_{}_{}.npz'.format(current_run.info.run_id,it),
            #         E_pretrain=model.E.weight.detach().cpu().numpy(), R_pretrain=model.R.weight.detach().cpu().numpy())

            # if loss_epoch[-1] < best_loss:
            #     print('loss decreases!')

            #     best_loss = loss_epoch[-1]
            #     best_valid_iter = it
            #     # best_model_param = copy.deepcopy(model.state_dict())
            #     E_best, R_best = model.E.weight, model.R.weight
            #     # torch.save(model, archive_path + '/model.pkl')
            #     # torch.save(model.state_dict(), archive_path + '/model_params_'+args.model_name+'.pth')
            #     if it == args.num_iterations-1:
            #         np.savez(archive_path + 'ER_{}.npz'.format(current_run.info.run_id),
            #                  E_pretrain=E_best.detach().cpu().numpy(), R_pretrain=R_best.detach().cpu().numpy())
            #         # torch.save(best_model_param, archive_path + 'model_params.pth')
            # else:
            #     if it - best_valid_iter > args.patience:
            #         np.savez(archive_path + 'ER_{}.npz'.format(current_run.info.run_id),
            #                  E_pretrain=E_best.detach().cpu().numpy(), R_pretrain=R_best.detach().cpu().numpy())
            #         # torch.save(best_model_param, archive_path + 'model_params.pth')
            #         print('\n\n=========== Final Results ===========')
            #         print('Best Epoch: %d\nLoss: %.8f\n' % (best_valid_iter, best_loss))
            #         break
            #     elif it == args.num_iterations-1:
            #         np.savez(archive_path + 'ER_{}.npz'.format(current_run.info.run_id),
            #                  E_pretrain=E_best.detach().cpu().numpy(), R_pretrain=R_best.detach().cpu().numpy())
            #         # torch.save(best_model_param, archive_path + 'model_params.pth')
            #     else:
            #         print('loss didn\'t decrease for %d epochs, Best Iter=%d, Best Loss=%.8f' % (it - best_valid_iter, best_valid_iter, best_loss))
            # mlflow.log_metric(key='best_it', value=best_valid_iter, step=it)

        E_1=model.E.weight[:self.nreg]
        E_2=model.encode(self.g)[:self.nreg]
        np.savez(archive_path + 'ER_{}.npz'.format(current_run.info.run_id),
                             E_1=E_1.detach().cpu().numpy(),E_2=E_2.detach().cpu().numpy())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("--dataset", type=str, default="subkg_health", nargs="?", help="Dataset")###############
    parser.add_argument("--num_iterations", type=int, default=200, nargs="?", help="Number of iterations.")
    parser.add_argument("--batch_size", type=int, default=2048, nargs="?", help="Batch size.")
    parser.add_argument("--lr", type=float, default=0.003, nargs="?", help="Learning rate.")
    parser.add_argument("--dr", type=float, default=0.995, nargs="?", help="Decay rate.")
    parser.add_argument("--edim", type=int, default=64, nargs="?", help="Entity embedding dimensionality.")
    # parser.add_argument("-rmrel", "--remove_rel", type=str, action='append')
    # parser.add_argument("-adrel", "--add_rel", type=str, action='append')
    parser.add_argument("--dropout", type=float, default=0.2, nargs="?", help="Dropout rate.")
    parser.add_argument('--n_layer', default=1, type=int, help='Number of RGCN Layers to use')

    # parser.add_argument("--exp_name", type=str, default="pretrain")
    # parser.add_argument("--prefix", type=str, default="pretrain")
    parser.add_argument("--patience", type=int, default=10, nargs="?", help="valid patience.")
    parser.add_argument("--seed", type=int, default=20, nargs="?", help="random seed.")
    # parser.add_argument("--model_name", type=str, default="TuckER")
    # parser.add_argument("--loss", type=str, default="CE")
    # parser.add_argument("--kg_name", type=str, default="kg_reverse")
    parser.add_argument('--num_heads', default=8, type=int, help='GAT attention head number')


    args = parser.parse_args()
    print(args)

    data_dir = "./data/data_KG_0715_reg/" ###########################
    archive_path = './output/output_KG_0715_5runs/' ###########################

    assert os.path.exists(data_dir)
    if not os.path.exists(archive_path):
        os.mkdir(archive_path)

    # ~~~~~~~~~~~~~~~~~~ mlflow experiment ~~~~~~~~~~~~~~~~~~~~~

    experiment_name = 'GAT_bj_reg_5runs_restnum'
    # experiment_name = 'GAT_ny_5runs_crime' ###########################
    # experiment_name = 'test'
    mlflow.set_tracking_uri('/data/zhouzhilun/Region_Profiling/mlflow_output/')
    client = MlflowClient()
    try:
        EXP_ID = client.create_experiment(experiment_name)
        print('Initial Create!')
    except:
        experiments = client.get_experiment_by_name(experiment_name)
        EXP_ID = experiments.experiment_id
        print('Experiment Exists, Continuing')
    with mlflow.start_run(experiment_id=EXP_ID) as current_run:
        
        # ~~~~~~~~~~~~~~~~~ reproduce setting ~~~~~~~~~~~~~~~~~~~~~
        seed = args.seed
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        print('Loading data....')
        d = Data(data_dir=data_dir, ind_type='pop', reverse=True)
        params = vars(args)
        mlflow.log_params(params)

        experiment = Experiment(batch_size=args.batch_size, lr=args.lr, dr=args.dr, edim=args.edim)
        experiment.train_and_eval()

