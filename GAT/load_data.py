import numpy as np
from collections import defaultdict
import json
from tqdm import tqdm
class Data: 
    def __init__(self, data_dir, ind_type, reverse):
        self.ind_type = ind_type
        self.train_data, self.valid_data, self.test_data, self.region2id, self.ind2id = self.load_ind_data(data_dir)
        self.ent2id, self.rel2id, self.kg_data = self.load_kg_data(data_dir, reverse)

        self.nind=len(self.ind2id)
        print('number of node=%d, number of edge=%d, number of relations=%d' % (len(self.ent2id), len(self.kg_data), len(self.rel2id)))
        print('total region num={}, ind num={}'.format(len(self.region2id),self.nind))
        print('load finished..')

    def load_ind_data(self, data_dir):       
        data = []
        region_str, ind_str = [], []
        with open(data_dir + 'region2allinfo_ring6.json', 'r') as f:
            region2info=json.load(f)
        for k,v in region2info.items():
            ind=v[self.ind_type+'_cate']
            ind_str.append(ind)
            region_str.append(k)
            data.append([k,ind])
        
        data.sort(key=lambda x:int(x[0].split('_')[1]))
        # data.sort(key=lambda x:x[0])

        np.random.seed(seed=20)
        np.random.shuffle(data)
        region_str = sorted(list(set(region_str)), key=lambda x: int(x.split('_')[1]))
        # region_str = sorted(list(set(region_str)), key=lambda x: x)
        ind_str = sorted(list(set(ind_str)))
        region2id, ind2id = dict([(x, i) for i, x in enumerate(region_str)]), dict([(x, i) for i, x in enumerate(ind_str)])
        data_id = [[region2id[x[0]], ind2id[x[1]]] for x in data]


        L = len(data_id)
        train_data, valid_data, test_data = \
            data_id[0:int(L * 0.6)], data_id[int(L * 0.6):int(L * 0.8)], data_id[int(L * 0.8)::]

        return train_data, valid_data, test_data, region2id, ind2id

    def load_kg_data(self, data_dir, reverse):
        ent2id, rel2id = self.region2id.copy(), {}
        
        kg_data_str = []  # [(h,r,t),]
        # 读kg文件
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
