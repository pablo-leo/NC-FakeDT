import logging
import os
import numpy as np

from gensim.models import Word2Vec
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import nltk


class W2V():
    def __init__(self, out_size = 100, fname='trained_w2v'):
        self.out_size = out_size
        self.modelname = fname
        self.model = None
        
    def create_w2v(self, vocab, use_corpora=True):
        
        self.model = None
        
        if use_corpora:
            nltk.download('brown')
            nltk.download('reuters')
            nltk.download('movie_reviews')
            data = nltk.corpus.brown.sents() + nltk.corpus.reuters.sents() + nltk.corpus.movie_reviews.sents() + vocab
        else: 
            data = vocab

        # Train Word2Vec on nltk datasets
        model = Word2Vec(data, size=self.out_size, window=5, min_count=1, workers=4, batch_words = 1000)
        self.model = model
        self.retrain_w2v(data)
        
        print(model)
        
    def save_model(self, model):
        model.save(os.path.join('./Utils/w2v_models', self.modelname))
        self.model = model
    
    def load_w2v(self):
        self.model = Word2Vec.load(os.path.join('./Utils/w2v_models', self.modelname))
        self.save_model(self.model)
        self.get_dict()
        
        
    def retrain_w2v(self, data):
        if self.model != None:
            l = len(data)
            data = data
            self.model.train(data, total_examples=l, epochs=15)
            self.save_model(self.model)
        else:
            raise Exception('No model to retrain. Create or load a model first.')
        

    def vectorize_words(self, tweets, pad_size = 70):
        tweet2vec = []
        unknown = []
        for t in tweets:
            vecs = []
            for w in t:
                try:
                    vecs.append(self.model.wv[w])
                except KeyError:
                    # When model does not know the word
                    vecs.append(w)
                    unknown.append(w)
            
            pad_vecs = vecs + [np.zeros(self.out_size).tolist()] * (pad_size - len(vecs))
            tweet2vec.append(pad_vecs)

        print("Length input = "+str(len(tweets))+"\nLength output = " +str(len(tweet2vec)))
        print("Unknown words = " + str(len(unknown)))
        if unknown != []:
            # Train the model on unknown words
            self.retrain_w2v(tweets)
            
            for t in range(len(tweet2vec)):
                for w in range(len(tweet2vec[t])):
                    if type(tweet2vec[w]) == str:
                        tweet2vec[w] = self.model.wv[tweets[t][w]]
        return tweet2vec
    
    def get_dict(self):
        # create dictionary to pair words and vectors
        vectors = []
        words = []
        for k in self.model.wv.index2word[:]:
            vectors.append(self.model.wv[k])
            words.append(k)

        self.dictionary = {z[0]:list(z[1:]) for z in zip(words, vectors)} 
    
    def vec2word(self, vec):
        # returns a list of words matching the input list of vectors 
        words = []
        vec_list = list(vec)
        for vl in vec_list:
            for wor, v in self.dictionary.items():
                if vl == v:
                    words.append(wor)
        return words