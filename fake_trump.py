import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import sys, os, nltk, re, keras
from keras import Model
import time

from argparse import ArgumentParser

# out libraries
import Utils.W2V as wor2vec
from Utils.classifier import create_Trumpifier
from Utils.genetic_algorithm import GeneticAlgorithm

def generate_tweets(args):
    
    # --------------------------------------------------------------------------------
    
    # load raw data
    notTrump_df = pd.read_csv('./processedData/processedDataNotDonaldTrump.csv',delimiter="|", header=None)
    Trump_df = pd.read_csv('./processedData/processedDataDonaldTrump.csv',delimiter="|", header=None)

    # delete special characters, generate labels and training data for word2vec
    X_t = ["".join(re.split('[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>?]', t)).split() for t in Trump_df[1].values.tolist()]
    y_t = np.zeros(len(X_t), dtype=int)
    X_nt = ["".join(re.split('[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>?]', t)).split() for t in notTrump_df[1].values.tolist()]
    y_nt = np.ones(len(X_nt), dtype=int)

    # make random combinations of each tweet
    X_random_t = []
    X_random_nt = []

    for i in range(len(X_t)):
        row = X_t[i].copy()
        np.random.shuffle(row)
        X_random_t.append(row)

    for i in range(len(X_nt)):
        row = X_nt[i].copy()
        np.random.shuffle(row)
        X_random_nt.append(row)

    y_random_t = np.ones(len(X_random_t), dtype=int) * 2
    y_random_nt = np.ones(len(X_random_nt), dtype=int) * 2


    # Convert all data to vectors
    vec_size = 100
    tweet_len = 70

    w2v_ds = np.concatenate((X_t, X_nt, X_random_t, X_random_nt))

    USE_NO_TWEET_W = False
    
    w2v = wor2vec.W2V(out_size = vec_size, fname="Final_w2v_only_tweets")
    w2v.load_w2v()
    w2v_dict = w2v.dictionary

    if USE_NO_TWEET_W:
        # create random data including words that are not inthe tweets
        new_random_X = np.random.choice(list(w2v_dict.keys()), (len(X_t), tweet_len))
        offsets = np.random.randint(1, tweet_len, len(new_random_X))
        new_random_X = [new_random_X[i][:offsets[i]] for i in range(len(new_random_X))]
        new_random_X = [list(c) for c in new_random_X]
        new_random_y = np.ones(len(new_random_X), dtype=int) * 2

        # merge all data
        X = np.concatenate((X_t, X_nt, X_random_t, X_random_nt, new_random_X))
        y = np.concatenate((y_t, y_nt, y_random_t, y_random_nt, new_random_y))
        y = keras.utils.to_categorical(y)

    else:
        # merge all data
        X = np.concatenate((X_t, X_nt, X_random_t, X_random_nt))
        y = np.concatenate((y_t, y_nt, y_random_t, y_random_nt))
        y = keras.utils.to_categorical(y)

    X_vec = w2v.vectorize_words(X, tweet_len)
    
    Trumpifier_name = "Final_only_tweets_trumpifier.h5"

    classifier = keras.models.load_model(os.path.join("./Utils/w2v_models/", Trumpifier_name))
    
    # ------------------------- Genetic Algorithm -----------------------------------

    GA = GeneticAlgorithm(topics = args.top, pop_size = args.pop, n_iter = args.it, fitness_model = classifier, 
                          vocab = w2v_dict, cross_rate = args.cross, mut_rate = args.mut, 
                          ind_shape = (tweet_len, vec_size), 
                          evaluationPercentage = [0.2, 0.2, 0.2, 0.2, 0.2])

    final_tweet, max_fit_hist, avg_fit_hist, max_len_hist, min_len_hist, avg_len_hist = GA.start_algorithm()
    

if __name__ == '__main__':
    p = ArgumentParser('Fake Donnald Trump')

    # variables
    p.add_argument('--pop', type = int, default = 100)
    p.add_argument('--it', type = int, default = 10)
    p.add_argument('--cross', type = float, default = .01)
    p.add_argument('--mut', type = float, default = .2)
    p.add_argument('--top', type = str, nargs='+', default = [])

    generate_tweets(p.parse_args())
    sys.exit(0)
