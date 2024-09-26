import json
import re
import sys
import numpy as np
from tqdm import tqdm
from py2neo import Graph, Node, Relationship
import time


def get_subkg(dataset, impact_aspects):
    kg = []
    with open('./data/{}_data/kg.txt'.format(dataset), 'r', encoding='utf-8') as txt_file:
        for line in txt_file:
            kg.append(line.strip().split('\t'))

    graph = Graph("bolt://localhost:7688")

    # 需要先导入kg，见construct_neo4j_kg.py

    for k, v in impact_aspects.items():
        print("============Aspect:", k)
        sub_kg_name = k
        sub_kg_paths = v
        triplets = set()
        for metapath in sub_kg_paths:
            print('Searching Metapath:', metapath)
            start_time = time.time()
            # 动态构建 Cypher 查询
            if len(metapath) == 1:
                query = f"""
                MATCH p=(n0)-[:{metapath[0]}]->(n1)
                RETURN n0.name AS n0, n1.name AS n1
                """
            else:
                query = 'MATCH p='
                for i, rel in enumerate(metapath):
                    query += f"(n{i})-[:{rel}]->"
                query += f"(n{len(metapath)})"
                query += ' RETURN ' + ', '.join([f'n{i}.name AS n{i}' for i in range(len(metapath) + 1)])
            # 执行查询并打印结果
            result = graph.run(query)
            all_paths = []
            for record in result:
                path = tuple(record[f"n{i}"] for i in range(len(metapath) + 1))
                all_paths.append(path)

            print(len(all_paths))
            for path in all_paths:  # ['node1', 'node2', 'node3']
                assert len(path) == len(metapath) + 1
                for i in range(len(metapath)):
                    triplets.add(tuple([path[i], metapath[i], path[i+1]]))
            print("Time:", time.time() - start_time)
        unique_triplets = list(triplets)#去除重复元素
        print(len(unique_triplets))
        with open('./data/{}_data/kg_{}.txt'.format(dataset, sub_kg_name), 'w', encoding='utf-8') as txt_file:
            txt_file.write('\n'.join(["{}\t{}\t{}".format(t[0], t[1], t[2]) for t in unique_triplets]))


