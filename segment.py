# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 15:20:20 2021

@author: Iacopo
"""

import numpy as np
from models import DeepTilingModels
from wiki_loader_sentences import *
from sklearn.metrics import f1_score, precision_score, recall_score
import argparse
import sys
import os
import json
import re
from tqdm import tqdm


def main(args):
    verbose = args.verbose
    
    if not os.path.exists(args.out_directory):
        os.makedirs(os.path.join(args.out_directory, 'segments'))
        os.makedirs(os.path.join(args.out_directory, 'embeddings'))
        
    elif not os.path.exists(os.path.join(args.out_directory, 'segments')):
        os.makedirs(os.path.join(args.out_directory, 'segments'))
        os.makedirs(os.path.join(args.out_directory, 'embeddings'))
        
    elif not os.path.exists(os.path.join(args.out_directory, 'embeddings')):
        os.makedirs(os.path.join(args.out_directory, 'embeddings'))
    
    config_file = os.path.join(args.config_file)
    
    assert os.path.exists(config_file), "Configuration file wasn't detected in the directory from which you are\
     running the current script: please move the configuration file to this directory --> {}".format(os.getcwd())
    
    
    with open(config_file, encoding='utf-8') as f:
        temp = f.read()
    
    config_file = json.loads(temp)
    
    import nltk
    nltk.download('punkt')
    
    data = []
    
    file_paths = []
    
    for root, directory, files in os.walk(args.data_directory):
        for file in files:
            file_paths.append(os.path.join(root, file))
    
    for file_path in file_paths:
        if os.stat(file_path).st_size:
            doc = read_wiki_file(file_path, 
                                 remove_preface_segment=False, 
                                 high_granularity=False, 
                                 return_as_sentences=True)
            
            sents = []
            for subs in doc[0]:
                if re.findall('[A-Za-z]+', subs):
                    sents.extend(nltk.sent_tokenize(subs))
            if sents:
                data.append([sents])
    
    encoder = args.encoder if args.encoder is not None else config_file['encoder']
    
    deeptiling = DeepTilingModels.DeepTiling(encoder)
    
    if args.window_value is None:
        
        with open('best_parameters.json', 'r') as f:
            parameters = json.loads(f.read())
            
        wv = parameters['Best Window Value']
        
        if args.threshold_multiplier is None and args.number_of_segments is None:
        
            th = parameters['Best Threshold Muliplier']
            
        else:
            
            th = 0
    
    elif args.threshold_multiplier is None:
        
        with open('best_parameters.json', 'r') as f:
            parameters = json.loads(f.read())
            
        wv = args.window_value
        if args.number_of_segments is None:
            
            th = parameters['Best Threshold Muliplier']
            
        else:
            
            th = 0
            
    else:
        wv = args.window_value
        th = args.threshold_multiplier
    
    if args.number_of_segments is None:
        number_of_segments = config_file['number_of_segments']
    else:
        number_of_segments = args.number_of_segments
    
    
    if args.Concatenate is None:
        cat = True if config_file['CONCATENATE']=='TRUE' else False
    else:
        cat = True if args.Concatenate=='TRUE' else False
    
    if cat:
        
        if verbose:
            print('Concatenating the files before predicting segments...')
        
        def join_segments(dataset):
            
            joined_dataset = []
            
            for sample in dataset:
              
              joined_dataset.extend(sample[0])
            
            return joined_dataset
        
        data = join_segments(data)
        
        if verbose:
            print('Files have been concatenated, starting segmentation...')
        
        if number_of_segments is not None:
            ns = number_of_segments[0]
            
        else:
            ns = None
        
        results = deeptiling.predict(data,
                                     parameters = {'window': wv,
                                                   'threshold': th},
                                                   number_of_segments = ns)
        
        if verbose:
            print('Segmentation done!\n {} segments were extracted...'.format(len(results['segments'])))
        
        for i, segment in enumerate(results['segments']):
            if verbose:
                print('Writing segment {} to {}'.format(str(i), args.out_directory+'/segments'))
            with open(os.path.join(args.out_directory,'segments', 'segment_'+str(i)), 'w') as f:
                f.writelines('%s\n' % sentence for sentence in segment)
            np.save(os.path.join(args.out_directory,'embeddings', 'segment_'+str(i)), results['embeddings'][i])
        
    
    else:
        pbar = tqdm(data)
        if verbose:
            print('Starting segmentation of provided files...')
            
        for index, doc in enumerate(pbar):
            
            pbar.set_description('Segmenting input text file number {}'.format(str(index+1)))
            
            if number_of_segments is not None:
                try:
                    ns = number_of_segments[index]
                    print(ns)
                except IndexError:
                    raise('If provided, number of segments must be passed as a list containing the number of segments per each document to be segmented!')
            else:
                ns = None
            
            results = deeptiling.predict(doc[0],
                                     parameters = {'window': wv,
                                                   'threshold': th},
                                                   number_of_segments = ns)
            
            if verbose:
                print('Segmentation for document {} done!\n {} segments were extracted...'.format(str(index+1),len(results['segments'])))
            
            for i, segment in enumerate(results['segments']):
                empty_segs = 0
                if segment:
                    filename = re.findall('[/]?([\w]+)[\.[\w]+]?', file_paths[index])[-1] + '_segment_'+ str(i-empty_segs)
                    if verbose:
                        print('Writing results of segmentation for document {} to {}, with filename {}'.format(str(index+1), args.out_directory+'/segments', filename))
                    with open(os.path.join(args.out_directory,'segments', filename), 'w') as f:
                        f.writelines('%s\n' %sentence for sentence in segment)
                    np.save(os.path.join(args.out_directory,'embeddings', filename), results['embeddings'][i])
                else:
                    empty_segs+=1
        
    if verbose:
            print('All segments have been written in separate files: find them in the output directory {}'.format(args.out_directory+'/segments'))


if __name__=='__main__':
    class MyParser(argparse.ArgumentParser):
        def error(self, message):
            sys.stderr.write('error: %s\n' % message)
            self.print_help()
            sys.exit(2)
            
    
    parser = MyParser(
            description = 'Run segmentation with parameters defined in the relative json file')
    
    parser.add_argument('--data_directory', '-data', type=str,
                        help='directory containing the data to be segmented')
    
    parser.add_argument('--config_file', '-cfg', default='parameters.json', type=str, 
                        help='Configuration file defining the hyperparameters and options to be used in training.')
    
    parser.add_argument('--out_directory', '-od', default='results', type=str,
                        help='the directory where to store the segmented texts')
    
    parser.add_argument('--window_value', '-wd', 
                        type=int,
                        default=None, 
                        help='Window value for the TextTiling algorithm, if not specified the programme will assume that the optimal value is stored in best_parameters.json file, previously obtained by running fit.py')
    
    parser.add_argument('--threshold_multiplier', '-th',
                        type=float,
                        default=None,
                        help='Threshold multiplier for the TextTiling algorithm without known number of segments, if not specified the programme will assume that the optimal value is stored in best_parameters.json file, previously obtained by running fit.py')
    
    parser.add_argument('--number_of_segments', '-ns',
                        type=int,
                        nargs = '+',
                        default=None,
                        help='List of number of segments (per document) to be returned (if known). Default is when number of segments are not known, otherwise the algorithm returns the n number of segments with higher depth score, as specified by the number at the index of the list relative to the current document.')
    
    parser.add_argument('--encoder', '-enc', type=str,
                        default=None, help='sentence encoder to be used (all sentence encoders from sentence_transformers library are supported)')
    
    parser.add_argument('--Concatenate', '-cat', type=str,
                        default=None, help='whether to concatenate the input files or to segment them individually')
                 
    parser.add_argument('--verbose', '-vb', type=bool, default=True, help='Whether to print messages during running.')
    
    
    args = parser.parse_args()
    
    main(args)