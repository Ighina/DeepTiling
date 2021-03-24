# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 16:09:56 2021

@author: Iacopo
"""

from LexRank import degree_centrality_scores
from sentence_transformers import util
import numpy as np
import argparse
import shutil
import sys
import os
import segment as seg

def main(args):

    out_directory = args.out_directory
    
    
    if not os.path.exists(os.path.join(out_directory, 'segments')):
        seg.main(args)
    
    if os.path.exists(os.path.join(out_directory, 'summaries')):
        overwrite = input(bool('The summaries directory inside {} folder already conatains material:\ndo you want to overwrite the content in {}/summaries?\nEnter anything for yes or nothing for no...'.format(out_directory)))
        
        if overwrite:
            shutil.rmtree(os.path.join(out_directory, 'summaries'))
            os.makedirs(os.path.join(out_directory, 'summaries'))
        else:
            raise ValueError('Then either remove the content in summaries folder manually or change output directory with the --out_directory argument')
    
    else:
        os.makedirs(os.path.join(out_directory, 'summaries'))
    
    for root, directory, files in os.walk(os.path.join(out_directory, 'segments')):
        for file in files:
            if file.startswith('paths_cach'):
                pass
            else:
                segment = []
                
                with open(os.path.join(out_directory, 'segments', file), 'r') as f:
                    for line in f:
                        segment.append(line)
                
                if args.number_top_sentences>=len(segment):
                    shutil.copyfile(os.path.join(out_directory, 'segments', file),
                                    os.path.join(out_directory, 'summaries', file))
                    
                else:
                
                    embeddings = np.load(os.path.join(out_directory, 'embeddings', file+'.npy'))
                    
                    #Compute the pair-wise cosine similarities
                    cos_scores = util.pytorch_cos_sim(embeddings, embeddings).numpy()
                    
                    #Compute the centrality for each sentence
                    centrality_scores = degree_centrality_scores(cos_scores, threshold=None)
                    
                    #We argsort so that the first element is the sentence with the highest score
                    most_central_sentence_indices = np.argsort(-centrality_scores)
                    
                    top_index = 1
                    summary = []
                    for idx in most_central_sentence_indices[:args.number_top_sentences]:
                        
                        top_sentence = segment[idx]
                        
                        if args.verbose:
                            print('Sentence number {} of segment {} is: {}'.format(
                                top_index, file, top_sentence))
                            
                            top_index += 1
                        
                        summary.append(top_sentence)
                        
                    with open(os.path.join(out_directory, 'summaries', file), 'w') as f:
                        f.writelines('%s\n' %sentence for sentence in summary)

if __name__ == '__main__':
    
    class MyParser(argparse.ArgumentParser):
        def error(self, message):
            sys.stderr.write('error: %s\n' % message)
            self.print_help()
            sys.exit(2)
    
    parser = MyParser(
            description = 'Run segmentation and summarization with parameters defined in the relative json file\
                or as passed in the command line')
    
    parser.add_argument('--data_directory', '-data', type=str,
                        help='directory containing the data to be segmented')
    
    parser.add_argument('--config_file', '-cfg', default='parameters.json', type=str, 
                        help='Configuration file defining the hyperparameters and options to be used in training.')
    
    parser.add_argument('--out_directory', '-od', default='results', type=str,
                        help='the directory where to store the segmented texts')
    
    
    parser.add_argument('--number_top_sentences', '-nt', type=int,
                        default=1, help='number of sentences to extract per segment as summary')
    
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
                        help='Number of segments to be returned (if known). Default is when number of segments are not None, otherwise the algorithm returns the n number of segments with higher depth score, as specified by this parameter')
    
    parser.add_argument('--encoder', '-enc', type=str,
                        default=None, help='sentence encoder to be used (all sentence encoders from sentence_transformers library are supported)')
    
    parser.add_argument('--Concatenate', '-cat', type=bool,
                        default=None, help='whether to concatenate the input files or to segment them individually')
    
    parser.add_argument('--verbose', '-vb', type=bool, default=True, help='Whether to print messages during running.')
    
    args = parser.parse_args()
    
    main(args)
