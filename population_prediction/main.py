from load_data import Data
# from load_relpaths import Relpaths
# from load_relpaths_jl import Relpathsjl
# from load_relpaths_cl import Relpathscl
from load_subkg import *
from model import *
import numpy as np
import torch
import time
from collections import defaultdict
import argparse
import setproctitle
import mlflow
from mlflow.tracking import MlflowClient
import os
from tqdm import tqdm
import random
from torch_geometric.data import Data as geoData
import torch_geometric.transforms as T
import torch.nn.functional as F
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import setproctitle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
setproctitle.setproctitle('RP@zzl')

device = torch.device('cuda')


def compute_metrics(y_pred, y_test):
    y_pred[y_pred<0] = 0
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mae, np.sqrt(mse), r2

class Experiment:
    def __init__(self, lr, edim, batch_size):
        self.lr = lr
        self.edim = edim
        self.batch_size = batch_size
        self.num_iterations = args.num_iterations

        # if params['pretrain'] == 'true':
        #     print('Loading pretrained weights....')
        #     pretrain_emb = np.load('./pretrain_emb/ER_' + args.dataset + '_TuckER' + '_' + str(edim) + '.npz')
        #     params['E_pretrain'] = torch.from_numpy(pretrain_emb['E_pretrain']).to(device)
        #     params['R_pretrain'] = torch.from_numpy(pretrain_emb['R_pretrain']).to(device)
    

        self.kwargs = params
        self.kwargs['device'] = device
        self.gs, self.edge_index = self.build_graph()

    def build_graph(self):
        gs=[]
        for k,v in d.mp2data.items():
            edge_index=torch.tensor([[x[0] for x in v['kg_data']], [x[2] for x in v['kg_data']]],dtype=torch.long,device=device)
            edge_type= torch.tensor([x[1] for x in v['kg_data']], dtype=torch.int, device=device)
            data=geoData(x=torch.zeros(len(v['ent2kgid']),1),edge_index=edge_index,edge_attr=edge_type)
            trans=T.ToSparseTensor()
            trans(data)
            edge_index=data.adj_t

            eids=torch.tensor(list(v['ent2kgid'].values()),device=device)
            gs.append([edge_index,eids])

        # full kg
        edge_index=torch.tensor([[x[0] for x in d.kg_data], [x[2] for x in d.kg_data]],dtype=torch.long,device=device)
        edge_type= torch.tensor([x[1] for x in d.kg_data], dtype=torch.int, device=device)
        data=geoData(edge_index=edge_index,edge_attr=edge_type)
        trans=T.ToSparseTensor()
        trans(data)
        edge_index=data.adj_t

        return gs,edge_index

    def train_and_eval(self):
        print('building model....')
        model = HAN(d, **self.kwargs)
        model = model.to(device)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)

        print("Starting training...")
        ind = torch.tensor(d.indicator,device=device)

        best_valmetric, best_valiter = float('-inf'), 0

        allreg=list(range(d.nreg))
        for it in range(1, self.num_iterations + 1):
            print('\n=============== Epoch %d Starts...===============' % it)
            start_train = time.time()
            model.train()

            np.random.shuffle(allreg)
            k=0
            losses=[]
            for j in tqdm(range(0, int(d.nreg/self.batch_size)+1)):
                if k+self.batch_size<=len(allreg):
                    uids=allreg[k:k+self.batch_size]
                else:
                    uids=allreg[k:]+allreg[:k+self.batch_size-len(allreg)]
                k=(k+self.batch_size)%len(allreg)
                u_idx = torch.tensor(uids, device=device)

                pred = model.forward(self.gs, self.edge_index, d.metapath_emb, all_tasks_emb=d.all_tasks_emb)
                opt.zero_grad()

                # loss 
                pred_batch = pred[u_idx]
                true = ind[u_idx]
                loss = model.loss(pred_batch, true)

                loss.backward()
                opt.step()

                losses.append(loss.item())
            mlflow.log_metrics({'train_time': time.time()-start_train,
                                'loss':np.mean(losses),
                                'current_it': it}, step=it)
            print('loss:%.3f'%np.mean(losses))     

            model.eval()
            with torch.no_grad():
                print("Validation:")
                valid_pred=np.array([pred[x[0],0].item() for x in d.valid_data])
                valid_true=np.array([x[1] for x in d.valid_data])
                mae_valid, rmse_valid, r2_valid = compute_metrics(valid_pred,valid_true)
                valmetric=r2_valid
                valid_dict = {'valid_mae': mae_valid, 'valid_rmse': rmse_valid, 'valid_r2': r2_valid}
                mlflow.log_metrics(valid_dict, step=it)
                print('MAE: %.6f' % (mae_valid))
                print('RMSE: %.6f' % (rmse_valid))
                print('R2: %.6f' % (r2_valid))

                print("Test:")
                test_pred=np.array([pred[x[0],0].item() for x in d.test_data])
                test_true=np.array([x[1] for x in d.test_data])
                mae_test, rmse_test, r2_test = compute_metrics(test_pred,test_true)
                test_dict = {'test_mae': mae_test, 'test_rmse': rmse_test, 'test_r2': r2_test}
                mlflow.log_metrics(test_dict, step=it)
                print('MAE: %.6f' % (mae_test))
                print('RMSE: %.6f' % (rmse_test))
                print('R2: %.6f' % (r2_test))


                if valmetric > best_valmetric:
                    best_valmetric = valmetric
                    best_valiter = it
                    best_test_mae = mae_test
                    best_test_rmse = rmse_test
                    best_test_r2 = r2_test
                    print('Valid R2 increases, Best Test MAE=%.4f, RMSE=%.4f, R2=%.4f,' % (best_test_mae, best_test_rmse, best_test_r2))

                    # save best embedding
                    E_reg, E_kg = model.get_emb(self.gs, self.edge_index, d.metapath_emb)
                    E_reg=E_reg.cpu().detach().numpy()
                    E_kg=E_kg.cpu().detach().numpy()
                    np.savez(output_dir+'best_emb.npz', E_reg=E_reg, E_kg=E_kg)
                    
                else:
                    if it - best_valiter >= args.patience:
                        print('\n\n=========== Final Results ===========')
                        print('Best Epoch: %d\nTest MAE=%.4f, RMSE=%.4f, R2=%.4f,' % (best_valiter, best_test_mae, best_test_rmse, best_test_r2))
                        break
                    else:
                        print('Valid R2 didn\'t increase for %d epochs, Best Epoch=%d, Best MAE=%.4f, RMSE=%.4f, R2=%.4f, Best_Valid_Metric_R2=%.4f,' %
                              (it - best_valiter, best_valiter, best_test_mae, best_test_rmse, best_test_r2, best_valmetric))
        mlflow.log_metric(key='best_it', value=best_valiter, step=it)
        mlflow.log_metric(key='best_valid_r2', value=best_valmetric, step=it)
        mlflow.log_metric(key='best_test_mae', value=best_test_mae, step=it)
        mlflow.log_metric(key='best_test_rmse', value=best_test_rmse, step=it)
        mlflow.log_metric(key='best_test_r2', value=best_test_r2, step=it)
        return best_test_mae, best_test_rmse, best_test_r2



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="beijing", help="choose the dataset.")
    parser.add_argument("--num_iterations", type=int, default=200, nargs="?", help="Number of iterations.")
    parser.add_argument("--indicator", type=str, default="pop", nargs="?", help="choose the indicator.Beijing:pop, orders, eco, rest;Shanghai:")
    parser.add_argument("--patience", type=int, default=20, help="valid patience.")
    parser.add_argument("--batch_size", type=int, default=128, nargs="?", help="Batch size.")
    parser.add_argument("--lr", type=float, default=0.0001, nargs="?", help="Learning rate.")
    parser.add_argument("--edim", type=int, default=64, nargs="?", help="Entity embedding dimension")
    parser.add_argument("--dropout", type=float, default=0.0, nargs="?", help="Dropout rate.")
    parser.add_argument("--seed", type=int, default=20, nargs="?", help="random seed.")
    parser.add_argument('--hidden_size', default=128, type=int, help='')
    # parser.add_argument("--pretrain", type=str, default="true", help="whether to use pretrain embedding.")
    # parser.add_argument("--freeze", type=str, default="true", help="whether to freeze parameters in training.")
    parser.add_argument('--sum_layer', default=2, type=int, help='')
    parser.add_argument('--sub_layer', default=1, type=int, help='')
    parser.add_argument('--current_task', default='Population_Prediction', type=str, help='')
    args = parser.parse_args()
    print(args)

    all_tasks = ['Population_Prediction', 'Order_Prediction']

    data_dir = './data/{}_data/'.format(args.dataset)
    output_dir = './output/{}_output/'.format(args.dataset)
    assert os.path.exists(data_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    # ~~~~~~~~~~~~~~~~~~ mlflow experiment ~~~~~~~~~~~~~~~~~~~~~
    experiment_name = 'test'

    mlflow.set_tracking_uri('./mlflow_output/')
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

        params = vars(args)
        mlflow.log_params(params)
        
        # r1 = Relpaths(data_dir=data_dir)
        # rr2 = Relpathsjl(data_dir=data_dir)
        # rr3 = Relpathscl(rrr1=r1, rrr2=rr2)
        # get_subkg(args.dataset, rr3.impact_aspects)
        # relpaths = [k for k in rr3.impact_aspects.keys()] # ['Geographical_Proximity', 'Business_and_Service...essibility', 'Population_Flow', 'Brand_Presence', 'Functional_Similarity']

        print(f'''+++++++++++++++++++++++{params['current_task']}+++++++++++++++++++++++++++''')
        print('Loading data....')
        with open(output_dir + 'relpaths.json', 'r') as f:
            relpaths = json.load(f)

        d = Data(data_dir=data_dir, output_dir=output_dir, relpaths=relpaths, indicator=args.indicator, all_tasks=all_tasks, device=device, params=params)

        experiment = Experiment(batch_size=args.batch_size, lr=args.lr, edim=args.edim)
        mae, rmse, r2 = experiment.train_and_eval()

        for rel in relpaths:
            os.remove(data_dir + "kg_{}.txt".format(rel))
        
        with open(output_dir + 'result.json', 'a', encoding='utf-8') as json_file:
            ree = "Test MAE=%.4f, RMSE=%.4f, R2=%.4f" % (mae, rmse, r2)
            json.dump(ree, json_file)
        