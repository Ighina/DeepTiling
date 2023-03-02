# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 15:08:08 2021

@author: Iacopo

All code related to the C99 algorithm is taken from https://github.com/intfloat/uts
"""
from transformers import BertTokenizer, BertForNextSentencePrediction
import torch
from sentence_transformers import SentenceTransformer
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.metrics.segmentation import pk
from nltk.metrics.segmentation import windowdiff
import segeval
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.decomposition import PCA
from umap import UMAP
import time
import os

def create_sentence_pair(sentences):
  return ([s for s in sentences[:-1]]+["none"], ["none"]+[s for s in sentences[1:]])

class Region:
    """
    Used to denote a rectangular region of similarity matrix,
    never instantiate this class outside the package.
    """
    def __init__(self, l, r, sm_matrix):
        assert(r >= l)
        self.tot = sm_matrix[l][r]
        self.l = l
        self.r = r
        self.area = (r - l + 1)**2
        self.lch, self.rch, self.best_pos = None, None, -1

    def split(self, sm_matrix):
        if self.best_pos >= 0:
            return
        if self.l == self.r:
            self.best_pos = self.l
            return
        assert(self.r > self.l)
        mx, pos = -1e9, -1
        for i in range(self.l, self.r):
            carea = (i - self.l + 1)**2 + (self.r - i)**2
            cur = (sm_matrix[self.l][i] + sm_matrix[i + 1][self.r]) / carea
            if cur > mx:
                mx, pos = cur, i
        assert(pos >= self.l and pos < self.r)
        self.lch = Region(self.l, pos, sm_matrix)
        self.rch = Region(pos + 1, self.r, sm_matrix)
        self.best_pos = pos

def get_model(model_name, nxt_sentence_prediction = False):
    
    if nxt_sentence_prediction:
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertForNextSentencePrediction.from_pretrained(model_name)
        return model, tokenizer
        
    if model_name.startswith('https') or model_name.lower().startswith('universal') or model_name.lower()=='use':
        module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
        use = hub.load(module_url)
        print ("module %s loaded" % module_url)
        class SBERT_model():
          def __init__(self, model):
            self.model = model
          def encode(self, sentences):
            return self.model(sentences).numpy()
        return (SBERT_model(use), None)
    
    else:
        return (SentenceTransformer(model_name), None)

class DeepTiling:
    
    def __init__(self, encoding_model = 'paraphrase-xlm-r-multilingual-v1',
                 nxt_sentence_prediction = False):
        
        self.next_sentence_prediction = nxt_sentence_prediction
        
        self.encoder, self.tokenizer = get_model(encoding_model,
                                                 nxt_sentence_prediction)
        
        self.encoder_name = encoding_model
                                                 
                                                 
    def precompute_embeddings(self, docs, dataset_name = 'Choi', parent_directory = 'embeddings'):
        out_dir = os.path.join(parent_directory, dataset_name, self.encoder_name)
            
        if os.path.exists(out_dir):
            assert len(os.listdir(out_dir))==0, 'The folder where to save the embeddings is not empty: if you want to save your embeddings first make sure that the folder {} is empty'.format(out_dir)
        else:
            os.makedirs(out_dir)
        
        for index, sentences in enumerate(docs):
            embedding = self.encoder.encode(sentences)
            np.save(os.path.join(out_dir, 'embeddings_'+str(index)), embedding)
                
    
    def compute_depth_score(self, sentences, window, clip = 2, 
                        single = False,
                        next_sentence_pred = False, tokenizer = None,
                        next_sentence_model = None, use_softmax = False,
                        smooth = False):
        
        scores = []
        if single:
          for index in range(len(sentences)-1):
            scores.append(cosine_similarity(sentences.iloc[index,:].values.reshape(1, -1),
                                            sentences.iloc[index+1,:].values.reshape(1,-1))[0][0])
      
        
        elif next_sentence_pred:
          paired_sentences = create_sentence_pair(sentences.values.tolist())
      
          assert tokenizer is not None and next_sentence_model is not None
          for pair in paired_sentences:
            encoding = tokenizer(pair[0], pair[1], return_tensors= 'pt')
            if use_softmax:
              scores.append(torch.nn.functional.softmax(next_sentence_model(
                  **encoding,
                  labels = torch.LongTensor([1])).logits)[0].detach().numpy()[0])
            else:
              scores.append(next_sentence_model(
                  **encoding,
                  labels = torch.LongTensor([1])).logits[0][0].detach().numpy())
      
        else:
          for index in range(len(sentences)-1):
            if index <= window:
              scores.append(cosine_similarity(sentences.iloc[:index+1,:].mean().values.reshape(1,-1), 
                            sentences.iloc[index+1:index+window+1,:].mean().values.reshape(1,-1))[0][0])
            
            else:
              scores.append(cosine_similarity(sentences.iloc[index-window+1:index+1,:].mean().values.reshape(1,-1), 
                              sentences.iloc[index+1:index+window+1,:].mean().values.reshape(1,-1))[0][0])
        
        
        
        """
        Below code is re-adapted from the nltk implementation of TextTiling.
        Calculates the depth of each gap, i.e. the average difference
        between the left and right peaks and the gap's score"""
      
        depth_scores = [0 for x in scores]
        # clip boundaries: this holds on the rule of thumb(my thumb)
        # that a section shouldn't be smaller than at least 2
        # pseudosentences for small texts and around 5 for larger ones.
      
        index = clip
      
        for gapscore in scores[clip:-clip]:
            lpeak = gapscore
            for score in scores[index::-1]:
                if score >= lpeak:
                    lpeak = score
                else:
                    break
            rpeak = gapscore
            for score in scores[index:]:
                if score >= rpeak:
                    rpeak = score
                else:
                    break
            depth_scores[index] = 1/2*(lpeak + rpeak - 2 * gapscore)
            index += 1
            
        if smooth:
            n = len(depth_scores)
            smooth_dep_score = [0 for _ in range(n)]
            for i in range(n):
                if i - 1 < 0 or i + 1 >= n:
                    smooth_dep_score[i] = depth_scores[i]
                else:
                    smooth_dep_score[i] = np.average(depth_scores[(i - 1):(i + 2)])
            depth_scores = smooth_dep_score
      
        return scores, depth_scores
        
    def C99(self, sentences, std_coeff = 1.2, window = 4, transform_rank = False):
        assert len(sentences) > 0
        if len(sentences) < 3:
            return [1] + [0 for _ in range(len(sentences) - 1)]
            
        n = len(sentences)
        self.window = min(window, n)
        
        self.sim = cosine_similarity(sentences)
        if transform_rank:
            self.rank = np.zeros((n, n))
            for i in range(n):
                for j in range(i, n):
                    r1 = max(0, i - self.window + 1)
                    r2 = min(n - 1, i + self.window - 1)
                    c1 = max(0, j - self.window + 1)
                    c2 = min(n - 1, j + self.window - 1)
                    sublist = self.sim[r1:(r2 + 1), c1:(c2+1)].flatten()
                    lowlist = [x for x in sublist if x < self.sim[i][j]]
                    self.rank[i][j] = 1.0 * len(lowlist) / ((r2 - r1 + 1) * (c2 - c1 + 1))
                    self.rank[j][i] = self.rank[i][j]
        else:
            self.rank = self.sim
            
        self.sm = np.zeros((n, n))
        # O(n^4) solution
        # for i in xrange(n):
        #     for j in xrange(i, n):
        #         self.sm[i][j] = sum(self.rank[i:(j + 1), i:(j + 1)].flatten())
        #         self.sm[j][i] = self.sm[i][j]
        # O(n^2) solution
        prefix_sm = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                prefix_sm[i][j] = self.rank[i][j]
                if i - 1 >= 0: prefix_sm[i][j] += prefix_sm[i - 1][j]
                if j - 1 >= 0: prefix_sm[i][j] += prefix_sm[i][j - 1]
                if i - 1 >= 0 and j - 1 >= 0: prefix_sm[i][j] -= prefix_sm[i - 1][j - 1]
        for i in range(n):
            for j in range(i, n):
                if i == 0:
                    self.sm[i][j] = prefix_sm[j][j]
                else:
                    self.sm[i][j] = prefix_sm[j][j] - prefix_sm[i - 1][j] \
                                    - prefix_sm[j][i - 1] + prefix_sm[i - 1][i - 1]
                self.sm[j][i] = self.sm[i][j]

        # step 4, determine boundaries
        D = 1.0 * self.sm[0][n - 1] / (n * n)
        darr, region_arr, idx = [D], [Region(0, n - 1, self.sm)], []
        sum_region, sum_area = float(self.sm[0][n - 1]), float(n * n)
        for i in range(n - 1):
            mx, pos = -1e9, -1
            for j, region in enumerate(region_arr):
                if region.l == region.r:
                    continue
                region.split(self.sm)
                den = sum_area - region.area + region.lch.area + region.rch.area
                cur = (sum_region - region.tot + region.lch.tot + region.rch.tot) / den
                if cur > mx:
                    mx, pos = cur, j
            assert(pos >= 0)
            tmp = region_arr[pos]
            region_arr[pos] = tmp.rch
            region_arr.insert(pos, tmp.lch)
            sum_region += tmp.lch.tot + tmp.rch.tot - tmp.tot
            sum_area += tmp.lch.area + tmp.rch.area - tmp.area
            darr.append(sum_region / sum_area)
            idx.append(tmp.best_pos)

        dgrad = [(darr[i + 1] - darr[i]) for i in range(len(darr) - 1)]

        # optional step, smooth gradient
        smooth_dgrad = [dgrad[i] for i in range(len(dgrad))]
        if len(dgrad) > 1:
            smooth_dgrad[0] = (dgrad[0] * 2 + dgrad[1]) / 3.0
            smooth_dgrad[-1] = (dgrad[-1] * 2 + dgrad[-2]) / 3.0
        for i in range(1, len(dgrad) - 1):
            smooth_dgrad[i] = (dgrad[i - 1] + 2 * dgrad[i] + dgrad[i + 1]) / 4.0
        dgrad = smooth_dgrad

        avg, stdev = np.average(dgrad), np.std(dgrad)
        cutoff = avg + std_coeff * stdev
        assert(len(idx) == len(dgrad))
        above_cutoff_idx = [i for i in range(len(dgrad)) if dgrad[i] >= cutoff]
        if len(above_cutoff_idx) == 0: boundary = []
        else: boundary = idx[:max(above_cutoff_idx) + 1]
        ret = [0 for _ in range(n)]
        for i in boundary:
            ret[i-1] = 1
            # boundary should not be too close
            for j in range(i - 2, i + 2):
                if j >= 0 and j < n and j != (i-1) and ret[j] == 1:
                    ret[i-1] = 0
                    break
        return [1] + ret[:-1]
        
            
    def compute_boundaries(self, depth_scores, threshold = 'default', 
                           postprocess = True, clip = 2, number_of_segments = None):
        if number_of_segments is not None:
          boundaries = np.array([False for i in range(len(depth_scores))])
          segments = np.argpartition(depth_scores, -number_of_segments)[-number_of_segments:]
          boundaries[segments] = True
          return boundaries
        if threshold=='default':
          threshold = max(0.01, np.mean(depth_scores) + np.std(depth_scores)*2)
        if threshold:
          boundaries = np.array(depth_scores)>threshold
        else:
          boundaries = np.array(depth_scores)>0
        if postprocess:
          new_boundaries = np.copy(boundaries)
          for i in range(clip, len(boundaries)):
            slices = boundaries[i-clip:i+1]
            
            if np.sum(slices)>1:
              depth_scores_slices = depth_scores[i-clip:i+1]
              new_slice = [0 for i in range(len(slices))]
              new_slice[np.argmax(depth_scores_slices)] = 1
              new_boundaries[i-clip:i+1] = new_slice
          return new_boundaries  
        else:
          return boundaries
    
    def compute_Pk(self, boundaries, ground_truth, window_size = None, 
                   segval = True, boundary_symb = '1'):
        hypothesis = ''.join(boundaries.astype(int).astype(str).tolist())
        reference = ''.join(ground_truth.astype(int).astype(str).tolist())
        if isinstance(window_size, int):
          if segval:
            result = segeval.pk(segeval.convert_nltk_to_masses(hypothesis, boundary_symbol=boundary_symb),
                      segeval.convert_nltk_to_masses(reference, boundary_symbol=boundary_symb),
                      window_size = window_size)
          else:
            result = pk(reference, hypothesis, window_size)
        else:
          if segval:
            result = segeval.pk(segeval.convert_nltk_to_masses(hypothesis, boundary_symbol=boundary_symb),
                      segeval.convert_nltk_to_masses(reference, boundary_symbol=boundary_symb))
          else:
            result = pk(reference, hypothesis)
        return result
    
    def compute_window_diff(self, boundaries, ground_truth, window_size = None, 
                            segval = True, boundary_symb = '1'):
        hypothesis = ''.join(boundaries.astype(int).astype(str).tolist())
        reference = ''.join(ground_truth.astype(int).astype(str).tolist())
        if isinstance(window_size, int):
          if segval:
            result = segeval.window_diff(segeval.convert_nltk_to_masses(hypothesis, boundary_symbol=boundary_symb),
                      segeval.convert_nltk_to_masses(reference, boundary_symbol=boundary_symb),
                      window_size = window_size)
          else:
            result = windowdiff(reference, hypothesis, window_size)
        else:
          if segval:
            result = segeval.window_diff(segeval.convert_nltk_to_masses(hypothesis, boundary_symbol=boundary_symb),
                     segeval.convert_nltk_to_masses(reference, boundary_symbol=boundary_symb))
          else:
            result = windowdiff(reference, hypothesis)
        return result
    
    def fit(self, dataset, clip = 2,
            window_range = (1, 11, 1), 
            threshold_range = (0.5, 2.5, 0.5),
            verbose = True, timer = False,
            multi_encode = False, 
            pca = False, n_components = 30,
            umap = False, n_neighbors = 15,
            use_softmax = False,
            tune_on = 'Pk', from_precomputed = False):
        
        model = self.encoder
        tokenizer = self.tokenizer
        next_sentence_pred = self.next_sentence_prediction
        
        best_results = {'window_valuePk': 0, 'window_valueWD': 0, 
                        'Pk': 1, 
                        'WindowDiff': 1,
                        'f1':0, 'precision':0, 'recall':0, 
                        'threshold_valuePk': 0,
                        'threshold_valueWD': 0, 'best_windowF1': 0, 
                        'best_thresholdF1': 0}
        
        if pca:
          pca_obj = PCA(n_components = n_components)
          best_results['PCA_transformer'] = pca_obj
        if umap:
          pca_obj = UMAP(n_neighbors=n_neighbors, 
                              n_components=n_components, 
                              metric='cosine')
          best_results['PCA_transformer'] = pca_obj
        
        window_start, window_end, window_step = window_range
        
        thr_start, thr_end, thr_step = threshold_range
        
        for window in range(window_start, window_end, window_step):
            for threshold in range(int(thr_start*10), int(thr_end*10), int(thr_step*10)):
                threshold /= 10
                Pk = []
                WD = []
                F1 = []
                precision = []
                recall = []
                times = []
                index = 0
                for index, doc in enumerate(dataset):
                  sentences, true_lab, path = doc
                  path = os.normpath(path) # normalize path to the current file
                  dataset_name = path.split(os.sep)[1] # assume that the second directory in the path is the name of the dataset being used
                  
          
                  if sentences:
                    if timer:
                      start = time.time()
                    index += 1
                    if next_sentence_pred:
                      scores, depth_scores = self.compute_depth_score(sentences=pd.DataFrame(sentences),
                                                                 window = 0, clip=clip,
                                                                 next_sentence_pred = True,
                                                                 next_sentence_model = model,
                                                                 tokenizer = tokenizer,
                                                                 use_softmax = use_softmax)
                    else:
                      if multi_encode:
                        embs = []
                        for index in range(len(sentences)):
                          if index<window:
                            embs.append(model.encode(' '.join(sentences[:index+1])))
                          
                          else:
                            embs.append(model.encode(' '.join(sentences[index-window+1:index+1])))
                        embs = np.array(embs)
                      elif from_precomputed:
                          filename = os.path.join('embeddings', dataset_name, self.encoder_name, 'embeddings_'+str(index)+'.npy')
                          assert os.path.exists(filename), 'path to file {} does not exist: have you precomputed the embeddings? If not, set the "from_precomputed" option to False'.format(filename)
                          embs = np.load(filename)
                      else:
                        embs = model.encode(sentences)
                      if pca or umap:
                        embs = pca_obj.fit_transform(embs)
                        
                      embs = pd.DataFrame(embs)
          
                      scores, depth_scores = self.compute_depth_score(embs, 
                                                                      window = window, 
                                                                      clip=clip,
                                                                      single = multi_encode)
          
                    
                    boundaries = self.compute_boundaries(depth_scores, threshold=(np.mean(depth_scores) +
                                                    np.std(depth_scores)*threshold))
                    
                    if timer:
                      end = time.time()
                      times.append(end-start)
          
                    long_true_lab = np.array([1 if i in true_lab else 0 for i in range(len(sentences))])
          
          
          
                    Pk.append(self.compute_Pk(boundaries=boundaries, 
                                              ground_truth = long_true_lab[:-1]))
                    WD.append(self.compute_window_diff(boundaries=boundaries, 
                                                       ground_truth = long_true_lab[:-1]))
          
                    if long_true_lab.sum()>0:
                      F1.append(f1_score(boundaries, long_true_lab[:-1], zero_division = 0))
                      precision.append(precision_score(long_true_lab[:-1], boundaries))
                      recall.append(recall_score(long_true_lab[:-1], boundaries, zero_division = 1))
                    else:
                      print('no true labs!')
                      F1.append(f1_score(boundaries, long_true_lab[:-1], zero_division = 1))
                      precision.append(precision_score(long_true_lab[:-1], boundaries, zero_division = 1))
                      recall.append(recall_score(long_true_lab[:-1], boundaries, zero_division = 1))
            
                Pk = np.mean(Pk)
                WD = np.mean(WD)
                F1 = np.mean(F1)
                precision = np.mean(precision)
                recall = np.mean(recall)
                if timer:
                  avg_time = np.mean(times)
                  print(avg_time)
                
                if best_results['Pk']>Pk:
                  if verbose:
                    print('New best Pk: ', Pk, ' with window size: ', window)
                  best_results['Pk'] = Pk
                  best_results['window_valuePk'] = window
                  best_results['threshold_valuePk'] = threshold
                  
                if best_results['WindowDiff'] > WD:
                  best_results['WindowDiff'] = WD
                  if verbose:
                    print('New best WindowDiff: ', WD, ' with window size: ',window)
                  best_results['window_valueWD'] = window
                  best_results['threshold_valueWD'] = threshold
                if best_results['f1']<F1:
                  best_results['f1'] = F1
                  best_results['precision'] = precision
                  best_results['recall'] = recall
                  best_results['window_valueF1'] = window
                  best_results['threshold_valueF1'] = threshold
      
        if timer:
          best_results['avg_time'] = avg_time
        
        self.best_results = best_results
        
        if 'window_value'+tune_on in best_results:
            self.parameters = {'window' : 
                               best_results['window_value'+tune_on],
                               'threshold': 
                               best_results['threshold_value'+tune_on]}
        
        else:
            print('Provided metric for which to store the best parameters\
                  is invalid: backing-off to Window Difference by default')
            self.parameters = {'window' :
                               best_results['window_valueWD'],
                               'threshold':
                               best_results['threshold_valueWD']}
        
        return best_results
    
    def predict(self, sentences,
                clip = 2,
                parameters = {'window': 0, 'threshold': 0},
                timer = False,
                multi_encode = False, 
                pca = False, pca_components = None,
                use_softmax = False,
                number_of_segments = None,
                smooth = False,
                C99 = False,
                C99_std_coeff = 1.2,
                C99_rank_transform = False,
                precomputed_filename = None):
        
        model = self.encoder
        tokenizer = self.tokenizer
        next_sentence_pred = self.next_sentence_prediction
        
        if parameters['window']<1:
            try:
                parameters = self.parameters
            except AttributeError:
                raise Exception("If window and threshold values are not\
                                provided in the right form, the function\
                                assumes that the fit method was already\
                                called to automatically find the optimal\
                                parameters. Please, call the fit method\
                                on your training dataset or provide a valid\
                                dictionary of the form {'window': window value, 'threshold': threshold value}")
                
        window = parameters['window']
        
        threshold = parameters['threshold']
        
        index = 0
        
        if sentences:
            if timer:
              start = time.time()
            index += 1
            if next_sentence_pred:
              scores, depth_scores = self.compute_depth_score(sentences=pd.DataFrame(sentences),
                                                         window = 0, clip=clip,
                                                         next_sentence_pred = True,
                                                         next_sentence_model = model,
                                                         tokenizer = tokenizer,
                                                         use_softmax = use_softmax,
                                                         number_of_segments = None)
                                                        
            else:
              if multi_encode:
                embs = []
                for index in range(len(sentences)):
                  if index<window:
                    embs.append(model.encode(' '.join(sentences[:index+1])))
                  
                  else:
                    embs.append(model.encode(' '.join(sentences[index-window+1:index+1])))
                embs = np.array(embs)
              elif precomputed_filename is not None:
                embs = np.load(precomputed_filename)
              else:
                embs = model.encode(sentences)
              if pca:
                try: 
                    pca_obj = self.best_results['PCA_transformer']
                    embs = pca_obj.fit_transform(embs)
                except AttributeError:
                    print('Warning: PCA transformation was not previously computed\
                          it will be computed directly on the current sentences')
                          
                    if pca_components is None:
                        print("Warning: provide number of principal components\
                              in order to perform PCA. No such number has been\
                              provided: PCA won't be performed.")
                    
                    else:
                        pca_obj = PCA(n_components = pca_components)
                    
                        embs = pca_obj.fit_transform(embs)
              
              embs = pd.DataFrame(embs)
              
              if C99:
                  boundaries = np.array(self.C99(embs, std_coeff = C99_std_coeff, window = window, 
                  transform_rank = C99_rank_transform))
  
            if not C99:
                scores, depth_scores = self.compute_depth_score(embs, 
                                                              window = window, 
                                                              clip=clip,
                                                              single = multi_encode,
                                                              smooth = smooth)
                
                boundaries = self.compute_boundaries(depth_scores, threshold=(np.mean(depth_scores) +
                                            np.std(depth_scores)*threshold), number_of_segments = number_of_segments)
            
            if timer:
              end = time.time()
              times = end-start
              print("Segmentation completed in {} minutes".format(times))
        
        else:
            raise Exception('Provide a non-empty list of sentences!')
            
        segments = []
        segmented_embs = []
        last_index = 0
        for index, boundary in enumerate(boundaries):
            if boundary:
                segments.append(sentences[last_index:index + 1])
                segmented_embs.append(embs.iloc[last_index:index + 1].values)
                last_index = index + 1
        
        segments.append(sentences[last_index:])
        segmented_embs.append(embs.iloc[last_index:].values)
        if C99:
            return {'segments': segments, 
                    'boundaries': boundaries,
                    'similarity_matrix': self.rank,
                    'embeddings': segmented_embs}
        else:
            return {'segments': segments, 
                    'boundaries': boundaries,
                    'depth_scores': depth_scores,
                    'embeddings': segmented_embs}
