import numpy as np
import json
import os
import torch

class Data:
    def __init__(self, data_dir, output_dir, relpaths, indicator, all_tasks, device, params):
        self.relpaths = relpaths
        self.all_tasks = all_tasks
        self.device = device
        self.dataset = params['dataset']

        self.reg2id = self.load_region_data(data_dir)
        self.ent2id, self.rel2id, self.kg_data = self.load_full_kg(data_dir)
        self.mp2data = self.load_subkg_data(data_dir)
        self.nreg = len(self.reg2id)
        self.indicator = np.load(data_dir+'{}.npy'.format(indicator))
        self.train_data, self.valid_data, self.test_data = self.load_dataset(data_dir, indicator)
        
        self.metapath_emb = self.load_metapath_emb(output_dir) # n_meta_paths, 768
        self.all_tasks_emb = self.load_task_emb() # {'Population_Prediction_E_reg': None, 'Population_Prediction_E_kg': None

        print('number of node=%d, number of edge=%d, number of relations=%d' % (len(self.ent2id), len(self.kg_data), len(self.rel2id)))
        print('sub-KGs:',relpaths)
        print('region num={}'.format(len(self.reg2id)))
        print('load finished..')

    def load_region_data(self, data_dir):      
        with open(data_dir + 'region2info.json', 'r') as f:
            region2info=json.load(f)
        regions=sorted(region2info.keys(),key=lambda x:x)
        reg2id=dict([(x,i) for i,x in enumerate(regions)])
        return reg2id
    
    def load_full_kg(self, data_dir):
        ent2id, rel2id = self.reg2id.copy(), {}
        kg_data_str = []
        with open(data_dir + 'kg.txt', 'r') as f:
            for line in f.readlines(): 
                h,r,t=line.strip().split('\t')
                kg_data_str.append((h,r,t))
        ents = sorted(list(set([x[0] for x in kg_data_str] + [x[2] for x in kg_data_str])))
        rels = sorted(list(set([x[1] for x in kg_data_str])))
        for i, x in enumerate(ents):
            try:
                ent2id[x]
            except KeyError:
                ent2id[x] = len(ent2id)
        rel2id = dict([(x, i) for i, x in enumerate(rels)])
        kg_data = [[ent2id[x[0]], rel2id[x[1]], ent2id[x[2]]] for x in kg_data_str]
        
        return ent2id, rel2id, kg_data

    def load_subkg_data(self, data_dir):
        mp2data={}
        for mp in self.relpaths:
            ent2id, rel2id = self.reg2id.copy(), {}
            kg_data_str = []
            with open(data_dir + 'kg_{}.txt'.format(mp), 'r') as f:
                for line in f.readlines():
                    h,r,t=line.strip().split('\t')
                    kg_data_str.append((h,r,t))
            ents = sorted(list(set([x[0] for x in kg_data_str] + [x[2] for x in kg_data_str])))
            rels = sorted(list(set([x[1] for x in kg_data_str])))
            for i, x in enumerate(ents):
                try:
                    ent2id[x]
                except KeyError:
                    ent2id[x] = len(ent2id)
            rel2id = dict([(x, i) for i, x in enumerate(rels)])
            kg_data = [[ent2id[x[0]], rel2id[x[1]], ent2id[x[2]]] for x in kg_data_str]
            ent2kgid={}
            for e in ent2id.keys():
                ent2kgid[e]=self.ent2id[e]
            mp2data[mp]={'ent2id':ent2id,'rel2id':rel2id,'kg_data':kg_data,'ent2kgid':ent2kgid}
        
        return mp2data
    
    def load_metapath_emb(self, data_dir):
        file = data_dir + 'metapath_embeddings.npy'
        metapath_emb = np.load(file)
        return torch.from_numpy(metapath_emb).to(self.device)
    

    def load_task_emb(self):
        # load all tasks emb
        all_tasks_emb = {}
        for task in self.all_tasks:
            # if task == args.current_task:
            #     continue
            task_dir = f'../{task}/output/{self.dataset}_output/'
            task_emb_file = task_dir + 'best_emb.npz'
            if os.path.exists(task_emb_file):
                task_emb = np.load(task_emb_file)
                all_tasks_emb[task+'_E_reg'] = torch.from_numpy(task_emb['E_reg']).to(self.device)
                all_tasks_emb[task+'_E_kg'] = torch.from_numpy(task_emb['E_kg']).to(self.device)
                print(f'{task} emb loaded..')
            else:
                print(f'{task} emb not found!')
                all_tasks_emb[task+'_E_reg'] = None
                all_tasks_emb[task+'_E_kg'] = None
        return all_tasks_emb
    
    def load_dataset(self, data_dir, indicator):
        reg2id = self.reg2id.copy()
        with open(data_dir + 'dataset.json','r') as f:
            datas=json.load(f)
        
        train = datas["train_data"]
        train_region_str = list(train.keys())
        train_region = [reg2id[r] for r in train_region_str]
        train_region_indicator = [np.log(i["{}".format(indicator)]) if i["{}".format(indicator)]>1 else 0  for i in train.values()]
        
        valid = datas["valid_data"]
        valid_region_str = list(valid.keys())
        valid_region = [reg2id[r] for r in valid_region_str]
        valid_region_indicator = [np.log(i["{}".format(indicator)]) if i["{}".format(indicator)]>1 else 0  for i in valid.values()]

        test = datas["test_data"]
        test_region_str = list(test.keys())
        test_region = [reg2id[r] for r in test_region_str]
        test_region_indicator = [np.log(i["{}".format(indicator)]) if i["{}".format(indicator)]>1 else 0  for i in test.values()]

        train_data = [[train_r, train_indicator] for train_r, train_indicator in zip(train_region, train_region_indicator)]
        valid_data = [[valid_r, valid_indicator] for valid_r, valid_indicator in zip(valid_region, valid_region_indicator)]
        test_data = [[test_r, test_indicator] for test_r, test_indicator in zip(test_region, test_region_indicator)]
        return train_data, valid_data, test_data
    
