from load_relpaths import Relpaths
from load_subkg import *
import argparse
import os
import json


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="beijing", help="choose the dataset.")
    parser.add_argument('--current_task', default='Population_Prediction', type=str, help='')
    args = parser.parse_args()
    print(args)

    all_tasks = ['Population_Prediction', 'Order_Prediction']

    data_dir = './data/{}_data/'.format(args.dataset)
    output_dir = './output/{}_output/'.format(args.dataset)
    assert os.path.exists(data_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    r1 = Relpaths(data_dir=data_dir, all_tasks=all_tasks, current_task=args.current_task, dataset=args.dataset)

    get_subkg(args.dataset, r1.impact_aspects_cl)

    with open(output_dir + 'IndAgent.txt', 'a', encoding='utf-8') as txt_file:
        txt_file.write(r1.llm_prompt + '\n\n\n')

    with open(output_dir + 'jlIndAgent.txt', 'a', encoding='utf-8') as txt_file:
        txt_file.write(r1.llm_prompt_jl + '\n\n\n')
    
    relpaths = [k for k in r1.impact_aspects_cl.keys()] # ['Geographical_Proximity', 'Business_and_Service...essibility', 'Population_Flow', 'Brand_Presence', 'Functional_Similarity']
    with open(output_dir + 'relpaths.json', 'w') as f:
        json.dump(relpaths, f)

    with open(output_dir + 'best_LLM_output.json', 'w', encoding='utf-8') as json_file:
        json.dump(r1.llm_output+r1.llm_output_jl, json_file)
    with open(output_dir + 'r3_output.json', 'a', encoding='utf-8') as json_file:
        json.dump(r1.llm_output_cl, json_file)

    with open(output_dir + 'path2info.json', 'w') as f:
        json.dump(r1.path2info, f)