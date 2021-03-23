# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 12:53:32 2021

@author: Iacopo

Part of the code (i.e. data preparation and model fitting) is based on 
https://towardsdatascience.com/topic-modelling-in-python-with-nltk-and-gensim-4ef03213cd21
by Susan Li
"""

import numpy as np
import pandas as pd
import spacy
spacy.load('en')
from spacy.lang.en import English
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
import random
import gensim
from gensim import corpora
import pickle
from collections import Counter
import time
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score, precision_score, recall_score

parser = English()
def tokenize(text):
    lda_tokens = []
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.like_url:
            lda_tokens.append('URL')
        elif token.orth_.startswith('@'):
            lda_tokens.append('SCREEN_NAME')
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens


def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma
    

def get_lemma2(word):
    return WordNetLemmatizer().lemmatize(word)

nltk.download('stopwords')
en_stop = set(nltk.corpus.stopwords.words('english'))

def prepare_text_for_lda(text):
    tokens = tokenize(text)
    tokens = [token for token in tokens if len(token) > 4]
    tokens = [token for token in tokens if token not in en_stop]
    tokens = [get_lemma(token) for token in tokens]
    return tokens


class TopicTiling:
    def __init__(self, dictionary = None, corpus = None,
                 tokenizer = prepare_text_for_lda):
        if isinstance(dictionary, str):
            self.load_dictionary(dictionary)
        else:
            self.dictionary = dictionary
        if isinstance(corpus, str):
            self.load_corpus(corpus)
        else:
            self.corpus = corpus
        
        self.tokenizer = tokenizer
        
    def preprocess(self, documents, verbose = False, 
                   sentences_as_docs = False):
        """
        Input:
            documents (list/array): list of sentences
        """
        
        text_data = []
        if sentences_as_docs:
          for doc in documents:
            for sentence in doc:
                tokens = prepare_text_for_lda(sentence)
                if random.random() > .99 and verbose:
                    print(tokens)
                text_data.append(tokens)
        else:
            for doc in documents:
              if doc:
                tokens = prepare_text_for_lda(' '.join(doc))
                if random.random() > .99 and verbose:
                    print(tokens)
                text_data.append(tokens)
        
        return text_data
    
    def create_corpus(self, documents, preprocess = True,
                      sentences_as_docs = False, verbose = False,
                      save = True, outfile_corpus = 'corpus.pkl',
                      outfile_dictionary = 'dictionary.gensim'):
        
        if preprocess:
            documents = self.preprocess(documents, verbose = verbose,
                                        sentences_as_docs = sentences_as_docs)
        
        if self.dictionary is None:
            dictionary = corpora.Dictionary(documents)
            self.dictionary = dictionary

        
        corpus = [dictionary.doc2bow(text) for text in documents]
        
        if save:
            pickle.dump(corpus, open(outfile_corpus, 'wb'))
            dictionary.save(outfile_dictionary)
        
        
        self.corpus = corpus
    
    def load_dictionary(self, dictionary_file):
        self.dictionary = corpora.Dictionary.load(dictionary_file)
        
    def load_corpus(self, corpus_file):
        f = open(corpus_file, 'r') 
        self.corpus = pickle.load(f)
        f.close()
    
    def fit_lda(self, documents, n_topics, n_iterations = 100,
            preprocess = True, verbose = False,
            sentences_as_docs = False, save = True, outfile_corpus = 'corpus.pkl',
            outfile_dictionary = 'dictionary.gensim', 
            outfile_model = 'model.gensim'):
        
        if self.corpus is None or self.dictionary is None:
            self.create_corpus(documents, preprocess = preprocess,
                               verbose = verbose, sentences_as_docs = sentences_as_docs,
                               save = save, outfile_corpus = outfile_corpus,
                               outfile_dictionary = outfile_dictionary)
            
        self.ldamodel = gensim.models.ldamodel.LdaModel(self.corpus, 
                                                   num_topics = n_topics,
                                                   id2word=self.dictionary, 
                                                   passes=n_iterations)
        
        if save:
            self.ldamodel.save(outfile_model)
        if verbose:
            topics = self.ldamodel.print_topics(num_words=4)
            for topic in topics:
                print(topic)
                
    def load_model(self, model_file):
        self.ldamodel = gensim.models.ldamodel.LdaModel.load(model_file)
        
    def load_dictionary_from_model(self):
        self.dictionary = self.ldamodel.id2word
    
    def get_doc_word_assignment(self, doc, model, dictionary, 
                                iterations = 5):
        word_topic_assignment = {}
        for iter in range(iterations):
          for wordID, topics in model.get_document_topics(
              dictionary.doc2bow(prepare_text_for_lda(doc)), per_word_topics=True)[1]:
            if wordID in word_topic_assignment:
              word_topic_assignment[wordID].append(topics[0])
            else:
              try:
                word_topic_assignment[wordID] = [topics[0]]
              except IndexError:
                pass
        word_topic_assignment = {dictionary.id2token[k]:Counter(v).most_common(1)[0][0] for k, v in word_topic_assignment.items()}
        return word_topic_assignment
    
    def topic_vectorizer_words(self, sents, n_topics, word_topic_dict, 
                               tokenizer):
  
        window_words = []
        for sent in sents:
          words = tokenizer(sent)
          
          window_words.extend([word_topic_dict[word] for word in words if word in word_topic_dict])
        
        one_hot = [0 for i in range(n_topics)]
        
        for el in window_words:
            one_hot[el] += 1
        
        return np.array(one_hot)
    
    def text_vectorizer(self, sents, word_id, tokenizer):
        window_words = []
        for sent in sents:
          words = tokenizer(sent)
          
          window_words.extend([word_id[word] for word in words if word in word_id])
        
        one_hot = [0 for i in range(len(word_id))]
        
        for el in window_words:
            one_hot[el] += 1
        
        return np.array(one_hot)
    
    def compute_depth_score(self, sentences, window, n_topics, 
                                  word_topic_dict, 
                                  tokenizer, clip = 2, combined = False, 
                                  embs = None, TextTiling = False,
                                  word_id = None):
  
        if combined:
      
          scores = []
          for index in range(len(sentences)-1):
            if index <= window:
              scores.append(cosine_similarity(np.concatenate((self.topic_vectorizer_words(sentences[:index+1], n_topics,
                                              word_topic_dict, tokenizer),
                                              embs[:index+1].mean(axis = 0))).reshape(1,-1), 
                              np.concatenate((self.topic_vectorizer_words(sentences[index+1:index+window+1],
                                              n_topics, word_topic_dict, 
                                              tokenizer), 
                                             embs[index+1:index+window+1].mean(axis=0))).reshape(1,-1))[0][0])
              
            else:
              scores.append(cosine_similarity(np.concatenate((self.topic_vectorizer_words(sentences[index-window+1:index+1],
                                                                                    n_topics,
                                              word_topic_dict, tokenizer),
                                              embs[index-window+1:index+1].mean(axis = 0))).reshape(1,-1), 
                              np.concatenate((self.topic_vectorizer_words(sentences[index+1:index+window+1],
                                              n_topics, word_topic_dict, 
                                              tokenizer), 
                                             embs[index+1:index+window+1].mean(axis = 0))).reshape(1,-1))[0][0])
      
        elif TextTiling:
          assert word_id is not None
          scores = []
          for index in range(len(sentences)-1):
            if index <= window:
              scores.append(cosine_similarity(self.text_vectorizer(sentences[:index+1],
                                              word_id, tokenizer).reshape(1,-1), 
                              self.text_vectorizer(sentences[index+1:index+window+1],
                                              word_id, 
                                              tokenizer).reshape(1,-1))[0][0])
              
            else:
              scores.append(cosine_similarity(self.text_vectorizer(sentences[index-window+1:index+1],
                                              word_id, tokenizer).reshape(1,-1), 
                              self.text_vectorizer(sentences[index+1:index+window+1],
                                              word_id, 
                                              tokenizer).reshape(1,-1))[0][0])
            
        
        else:
          scores = []
          for index in range(len(sentences)-1):
            if index <= window:
              scores.append(cosine_similarity(self.topic_vectorizer_words(sentences[:index+1], n_topics,
                                              word_topic_dict, tokenizer).reshape(1,-1), 
                              self.topic_vectorizer_words(sentences[index+1:index+window+1],
                                              n_topics, word_topic_dict, 
                                              tokenizer).reshape(1,-1))[0][0])
              
            else:
              scores.append(cosine_similarity(self.topic_vectorizer_words(sentences[index-window+1:index+1],
                                                                    n_topics,
                                              word_topic_dict, tokenizer).reshape(1,-1), 
                              self.topic_vectorizer_words(sentences[index+1:index+window+1],
                                              n_topics, word_topic_dict, 
                                              tokenizer).reshape(1,-1))[0][0])
        
        """Calculates the depth of each gap, i.e. the average difference
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
      
        return scores, depth_scores
    
    def compute_boundaries(self, depth_scores, 
                           threshold = 'default', 
                           postprocess = True, 
                           clip = 2):
          if threshold=='default':
            threshold = max(0.01, np.mean(depth_scores) + np.std(depth_scores)*1)
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
        
    def fit_parameters(self, dataset, n_topics,
                         ldamodel = None, 
                         dictionary = None,
                         tokenizer = None,
                         inference_iterations = 5, 
                         clip = 2,
                         window_range = (1, 11, 1), 
                         threshold_range = (0.5, 2.5, 0.5),
                         verbose = True, timer = False,
                         combined = False, sentence_encoder = None,
                         pca = False, n_components = 25,
                         tune_on = 'Pk'):
        
        if ldamodel is None:
            ldamodel = self.ldamodel
        
        if dictionary is None:
            dictionary = self.dictionary
        
        if tokenizer is None:
            tokenizer = self.tokenizer
        
        
        best_results = {'window_valuePk': 0, 'window_valueWD': 0, 
                        'Pk': 1, 
                        'WindowDiff': 1,
                        'f1':0, 'precision':0, 'recall':0, 
                        'threshold_valuePk': 0,
                        'threshold_valueWD': 0, 'window_valueF1': 0, 
                        'threshold_valueF1': 0}
        
        
        window_start, window_end, window_step = window_range
        
        thr_start, thr_end, thr_step = threshold_range
        
        for window in range(window_start, window_end, window_step):
            for threshold in range(thr_start, thr_end, thr_step):
                Pk = []
                WD = []
                F1 = []
                precision = []
                recall = []
                times = []
                index = 0
                for doc in dataset:
                  sentences, true_lab, path = doc
        
                  word2topic = self.get_doc_word_assignment(' '.join(sentences), 
                                                            self.ldamodel, 
                                                            self.dictionary,
                                                            inference_iterations)
        
                  if sentences:
                    if timer:
                      start = time.time()
                    index += 1
                    
                    if combined:
                        assert sentence_encoder is not None
                        
                        embs = sentence_encoder.encode(sentences)
                        
                        if pca:
                            pca_obj = PCA(n_components = n_components).fit(embs)
                            
                            best_results['PCA_transformer'] = pca_obj
                            
                            embs = pca_obj.transform(embs)
                            
                    else:
                        embs = None
                            
                    scores, depth_scores = self.compute_depth_score(sentences, window,
                                                               n_topics, word2topic,
                                                               tokenizer, clip = clip,
                                                               combined = combined,
                                                               embs = embs)
          
                    
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
                n_topics,
                ldamodel = None, 
                dictionary = None,
                tokenizer = None,
                inference_iterations = 5,
                clip = 2,
                parameters = {'window': 0, 'threshold': 0},
                timer = False,
                combined = False,
                sentence_encoder = None,
                pca = False,
                n_components = 25,
                TextTiling = False,
                word_id = None):
        
        if ldamodel is None:
            ldamodel = self.ldamodel
        
        if dictionary is None:
            dictionary = self.dictionary
        
        if tokenizer is None:
            tokenizer = self.tokenizer
        
        if parameters['window']<1:
            try:
                parameters = self.parameters
            except ValueError:
                raise Exception("If window and threshold values are not\
                                provided in the right form, the function\
                                assumes that the fit method was already\
                                called to automatically find the optimal\
                                parameters. Please, call the fit method\
                                on your training dataset or provide a valid\
                                dictionary of the form {'window': window value, 'threshold': threshold value}")
                
        window = parameters['window']
        
        threshold = parameters['threshold']
        
        if sentences:
            if timer:
              start = time.time()
            
            word2topic = self.get_doc_word_assignment(' '.join(sentences), 
                                                            self.ldamodel, 
                                                            self.dictionary,
                                                            inference_iterations)
                    
            if combined:
                assert sentence_encoder is not None
                
                embs = sentence_encoder.encode(sentences)
                
                if pca:
                    pca_obj = PCA(n_components = n_components).fit(embs)
                    
                    embs = pca_obj.transform(embs)
                    
            else:
                embs = None
                    
            scores, depth_scores = self.compute_depth_score(sentences, window,
                                                       n_topics, word2topic,
                                                       tokenizer, clip = clip,
                                                       combined = combined,
                                                       embs = embs, TextTiling = TextTiling,
                                                       word_id = word_id)
            
            boundaries = self.compute_boundaries(depth_scores, threshold=(np.mean(depth_scores) +
                                            np.std(depth_scores)*threshold))
            
            if timer:
              end = time.time()
              times = end-start
              print("Segmentation completed in {} minutes".format(times))
        
        else:
            raise Exception('Provide a non-empty list of sentences!')
            
        segments = []
        last_index = 0
        for index, boundary in enumerate(boundaries):
            if boundary:
                segments.append(sentences[last_index:index + 1])
        
        
        return {'segments': segments, 
                'boundaries': boundaries,
                'depth_scores': depth_scores}