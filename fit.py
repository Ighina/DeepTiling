# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 15:37:35 2021

@author: Iacopo
"""
import numpy as np
from models import DeepTilingModels
from choiloader_sentences import *
from wiki_loader_sentences import *
from sklearn.metrics import f1_score, precision_score, recall_score
import argparse
import sys
import os
import json

class MyParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)
        

parser = MyParser(
        description = 'Run training with parameters defined in the relative json file')

parser.add_argument('--training_folder', '-folder', default='data/wiki-50', type=str,
                    help='Folder containing data for fitting deep tiling parameters.')

parser.add_argument('--config_file', '-hyper', default='parameters.json', type=str, 
                    help='Configuration file defining the hyperparameters and options to be used in training.')

args = parser.parse_args()

config_file = os.path.join(args.config_file)

print(config_file)

assert os.path.exists(config_file), "Configuration file wasn't detected in the directory from which you are\
 running the current script: please move the configuration file to this directory --> {}".format(os.getcwd())


with open(config_file, encoding='utf-8') as f:
    temp = f.read()

config_file = json.loads(temp)

print(config_file)

if config_file['corpus']=='choi':
    data = ChoiDataset(args.training_folder)
elif config_file['corpus']=='wiki':
    data = WikipediaDataSet(args.training_folder, folder = True)
else:
    import nltk
    nltk.download('punkt')
    
    data = []

    file_paths = []
    
    for root, directory, files in os.walk(args.training_folder):
        for file in files:
            file_paths.append(os.path.join(root, file))
    
    for file_path in file_paths:
        if os.stat(file_path).st_size:
            doc = read_wiki_file(file_path, 
                                 remove_preface_segment=False, 
                                 high_granularity=False, 
                                 return_as_sentences=True)
            
            sents = []
            labs = []
            for subs in doc[0]:
                if subs.startswith('===='):
                    labs.append(index)
                else:
                    sents.extend(nltk.sent_tokenize(subs))
                    index = len(sents)-1
            labs.append(len(sents)-1)
            path = file_path
            data.append([sents, labs, path])

print('Loading the encoder...')
deeptiling = DeepTilingModels.DeepTiling(config_file['encoder'])
print('Encoder Loaded!')

Pks = {k:{v:0 for v in config_file['threshold_multipliers']} 
       for k in config_file['window_values']}

WDs = {k:{v:0 for v in config_file['threshold_multipliers']} 
       for k in config_file['window_values']}

F1s = {k:{v:0 for v in config_file['threshold_multipliers']} 
       for k in config_file['window_values']}

precisions = {k:{v:0 for v in config_file['threshold_multipliers']} 
       for k in config_file['window_values']}

recalls = {k:{v:0 for v in config_file['threshold_multipliers']} 
       for k in config_file['window_values']}
       
       
print("Example of empty PKs: {}".format(Pks))

if config_file['CONCATENATE'] == 'TRUE':
    
    print("concatenating training data...")
    
    def join_segments(dataset):
        joined_dataset = [[],[],[]]
        last_segment = 0
        for sample in dataset:
          
          sentences, labels, path = sample
          if labels:
            joined_dataset[0].extend(sentences)
            joined_dataset[1].extend([lab+last_segment for lab in labels])
            joined_dataset[2].append(path)
            last_segment = joined_dataset[1][-1] + 1
      
          else:
            joined_dataset[0].extend(sentences)
        
        return [joined_dataset]
    
    data = join_segments(data)
    
    print("Done!")



for window in Pks:
  for threshold in Pks[window]:
    results = []
    Pk = []
    WD = []
    F1 = []
    precision = []
    recall = []
    for doc in data:
        sentences, lab, path = doc
        if sentences:
           results.append(deeptiling.predict(sentences, parameters={'window':window, 'threshold':threshold}))
           long_true_lab = np.array([1 if i in lab else 0 for i in range(len(sentences))])
           Pk.append(deeptiling.compute_Pk(boundaries = results[-1]['boundaries'], ground_truth = long_true_lab[:-1], window_size=None))
           WD.append(deeptiling.compute_window_diff(boundaries = results[-1]['boundaries'], ground_truth = long_true_lab[:-1], window_size=None))
           F1.append(f1_score(results[-1]['boundaries'], long_true_lab[:-1]))
           precision.append(precision_score(results[-1]['boundaries'], long_true_lab[:-1]))
           recall.append(recall_score(results[-1]['boundaries'], long_true_lab[:-1], zero_division=1))
      
    Pks[window][threshold] = np.mean(Pk)
    WDs[window][threshold] = np.mean(WD)
    F1s[window][threshold] = np.mean(F1)
    precisions[window][threshold] = np.mean(precision)
    recalls[window][threshold] = np.mean(recall)
    
if config_file['metric']=='Pk':
    best_config = ()
    best_result = 1
    for window in Pks:
        for threshold in Pks[window]:
            if Pks[window][threshold]<best_result:
                best_result = Pks[window][threshold]
                best_config = (window, threshold)
elif config_file['metric'] == 'F1':
    best_config = ()
    best_result = 0
    for window in F1s:
        for threshold in F1s[window]:
            if F1s[window][threshold]<best_result:
                best_result = F1s[window][threshold]
                best_config = (window, threshold)
elif config_file['metric']=='WD':
    best_config = ()
    best_result = 1
    for window in WDs:
        for threshold in WDs[window]:
            if WDs[window][threshold]<best_result:
                best_result = WDs[window][threshold]
                best_config = (window, threshold)
elif config_file['metric'] == 'precision':
    best_config = ()
    best_result = 0
    for window in precisions:
        for threshold in precisions[window]:
            if precisions[window][threshold]<best_result:
                best_result = precisions[window][threshold]
                best_config = (window, threshold)
else:
    best_config = ()
    best_result = 0
    for window in recalls:
        for threshold in recalls[window]:
            if recalls[window][threshold]<best_result:
                best_result = recalls[window][threshold]
                best_config = (window, threshold)
                
print('best results are: {}.\n\
      with best configuration = {}'.format(best_result, best_config))

wv, th = best_config

best_results = {'Best Pk': float(Pks[wv][th]),
                'Best F1': float(F1s[wv][th]),
                'Best WindowDiff': float(WDs[wv][th]),
                'Best precision': float(precisions[wv][th]),
                'Best recall': float(recalls[wv][th]),
                'Optimization metric': config_file['metric'],
                'Best Window Value': int(wv),
                'Best Threshold Muliplier': float(th)}

with open('best_parameters.json', 'w+') as f:
    json.dump(best_results, f)