import json
import os
import random
import re
import time
# import openai
from prompt import *
from utils import *

class Relpaths:
    def __init__(self, data_dir, all_tasks, current_task, dataset):
        self.all_tasks = all_tasks
        self.current_task = current_task
        self.dataset = dataset

        self.llm_prompt, self.impact_aspects, self.llm_output = self.get_relpaths(data_dir)
        self.llm_prompt_jl, self.impact_aspects_jl, self.llm_output_jl = self.get_relpaths_jl(data_dir)
        self.impact_aspects_cl, self.llm_output_cl = self.get_relpaths_cl()
        self.path2info = self.get_relpaths_description()

    def clean_text(self, text):
        results = []
        pattern1 = r'(\d+\.)\s*([^:]+):'
        matches1 = re.findall(pattern1, text)
        keys = [match[1].strip().replace(' ', '_') for match in matches1]
        pattern2 = r'Relevant relation paths include:\s*(\{[^}]*\})'
        matches2 = re.findall(pattern2, text)
        string_arrays = [match.strip() for match in matches2]
        for array in string_arrays:
            # 使用正则表达式提取关系链
            pattern = re.compile(r'\((.*?)\)')
            matches = pattern.findall(array)
            # 将每个匹配项分割为关系列表
            result = [match.split(', ') for match in matches]
            results.append(result)
        impact_aspects = dict(zip(keys, results))

        pattern3 = r'\d+\.s*[^:]*:.*Relevant relation paths include:\s*\{[^}]*\}'
        llm_output= re.findall(pattern3, text, re.DOTALL)
        return impact_aspects, llm_output

    def get_relpaths(self, data_dir):
        with open(data_dir + 'top_40_paths.json',encoding='utf-8') as f:
            top_40_paths = json.load(f)
        if os.path.exists(data_dir + 'best_LLM_output.json'):
            with open(data_dir + 'best_LLM_output.json',encoding='utf-8') as f:
                previous_output = json.load(f)       
                reference = "For your reference, here are the most optimal paths we have discovered to date / paths that have been recently identified.\n"+"Reference results 1:\n"+previous_output[0]+"\n\nreference results 2:\n"+previous_output[1]+"\n\n"+"""Please note that you must organize the output from 5 aspects and only 1 relation path per aspect. The content of the output can't be exactly the same as the reference results, it has to be combined with your thinking to output a better answer. The format (not the content) of the output must be the same as the reference results, in particular, "Relevant relation paths include: {.....}" must be outputted."""
        else:
            reference = "Please note that you must organize the output from 5 aspects and only 1 relation path per aspect. And you must output in below format(not content), in particular,"+""""Relevant relation paths include: {.....}" must be outputted"""+"in every aspects:\n 1. Geographical Proximity: Relation paths that involve relations like 'NearBy(rel_34), 'BorderBy(rel_33) indicate geographical proximity. Region entities with similar characteristics may be geographically close to each other. Relevant relation paths include: {(rel_33)}.\n2.Business and Service Accessibility: Relation paths that involve relations like 'Region_ServedBy_BusinessArea(rel_29), 'BusinessArea_Serve_Region(rel_30) indicate the commonality of business centers. Region entities with similar characteristics may have the same business centers with similar services and business activities. Relevant relation paths include: {(rel_34, rel_29, rel_30)}.\n......\n"
        
        llm_prompt = impact_aspects_prompt.format(top_paths=', '.join(rel_path for rel_path in top_40_paths), reference=reference)
        result = run_llm(llm_prompt)
        impact_aspects, llm_output= self.clean_text(result)
        print(impact_aspects) # {'Geographical_Proximity': [['rel_33']],
        print(llm_output)
        
        return llm_prompt, impact_aspects, llm_output
    
    def get_relpaths_jl(self, data_dir):
        with open(data_dir + 'top_40_paths.json',encoding='utf-8') as f:
            top_40_paths = json.load(f)

        other_task_output = ''
        for task in self.all_tasks:
            if task == self.current_task:
                continue
            task_dir = f'../{task}/output/{self.dataset}_output/'
            task_LLM_output_file = task_dir + 'best_LLM_output.json'
            if os.path.exists(task_LLM_output_file):
                with open(task_LLM_output_file,encoding='utf-8') as f:
                    previous_output = json.load(f)
                    other_task_output += f'{task} LLM output:\n{previous_output[1]}\n\n'
                print(f'{task} LLM output loaded..')
            else:
                print(f'{task} LLM output not found!')
        
        if other_task_output == '':
            reference = "Please note that you must organize the output from 5 aspects and only 1 relation path per aspect. And you must output in below format(not content), in particular,"+""""Relevant relation paths include: {.....}" must be outputted"""+"in every aspects:\n 1. Geographical Proximity: Relation paths that involve relations like 'NearBy(rel_34), 'BorderBy(rel_33) indicate geographical proximity. Region entities with similar characteristics may be geographically close to each other. Relevant relation paths include: {(rel_34, rel_34)}.\n2.Business and Service Accessibility: Relation paths that involve relations like 'Region_ServedBy_BusinessArea(rel_29), 'BusinessArea_Serve_Region(rel_30) indicate the commonality of business centers. Region entities with similar characteristics may have the same business centers with similar services and business activities. Relevant relation paths include: {(rel_29, rel_30)}.\n......\n"
        else:
            reference = "As some other tasks about urban regions are highly relevant with the current task, we provide the optimal paths we have discovered for some other tasks for your reference.\n"+ other_task_output +"""Please note that you must organize the content of the output from 5 aspects and only 1 relation path per aspect. The content of the output can't be exactly the same as the reference results, it has to be combined with your thinking to output a better answer. The format (not the content) of the output must be the same as the reference above, in particular, "Relevant relation paths include: {.....}" must be outputted in every aspects:"""
                
        
        llm_prompt = impact_aspects_prompt_jl.format(top_paths=', '.join(rel_path for rel_path in top_40_paths), reference=reference)
        result = run_llm(llm_prompt)
        impact_aspects, llm_output= self.clean_text(result)
        print(impact_aspects)
        print(llm_output)
        
        return llm_prompt, impact_aspects, llm_output
    
    def get_relpaths_cl(self):
        str1 = ["{}. {}: Relevant relation paths include: {{({})}}".format(i+1,k,", ".join(v[0])) for i, (k, v) in enumerate(self.impact_aspects.items())]
        str2 = ["{}. {}: Relevant relation paths include: {{({})}}".format(i+6,k,", ".join(v[0])) for i, (k, v) in enumerate(self.impact_aspects_jl.items())]
        llm_prompt = impact_aspects_prompt_cl.format(reference="\n".join(str1+str2))
        result = run_llm(llm_prompt)
        impact_aspects, llm_output= self.clean_text(result)
        print(llm_prompt)
        print(llm_output)
        return impact_aspects, llm_output
    
    def get_relpaths_description(self):
        path2info = {}
        for k,v in self.impact_aspects_cl.items():
            desc = ''
            for i, rel in enumerate(v[0]):
                if i == 0:
                    desc += f'{rel2name[rel][0]} THAT {rel2name[rel][1]} {rel2name[rel][2]}'
                else:
                    desc += f' THAT {rel2name[rel][1]} {rel2name[rel][2]}'

            path2info[k] = {'Relations':v[0], 'Description':desc}
        
        return path2info