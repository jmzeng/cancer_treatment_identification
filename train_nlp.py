import pandas as pd
import numpy as np
import re
import time
import argparse
from os import listdir
from os.path import isfile, join

from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
from nltk.tokenize import sent_tokenize, word_tokenize
import gensim
from gensim.models import Doc2Vec, doc2vec, FastText

from sklearn import utils

# Load arguments
parser = argparse.ArgumentParser(description='Process model parameters')
parser.add_argument('-m', '--model', type=str, default="doc2vec", 
                    help='model used for training')
parser.add_argument('-vs', '--size', type=int, default=100,
                    help='vector size of word vectors')
parser.add_argument('-a', '--alpha', type=float, default=0.025,
                    help='learning rate')
parser.add_argument('-t', '--alg', type=int, default=0, 
                    help='doc2vec: distributed memory (1), fasttext: skip-gram (1)')
parser.add_argument('-w', '--window', type=int, default=5, 
                    help='window of maximum distance between words')
parser.add_argument('-s', '--sample', type=float, default=0.1, 
                    help='threshold for downsampling of high-frequency words')
parser.add_argument('-e', '--epochs', type=int, default=10, 
                    help='number of epochs to train the model')
parser.add_argument('-ns', '--ns_exponent', type=float, default=0.75,
                    help='exponent used for negative sampling distribution')
args = parser.parse_args()

# Set the data directory
print("Loading dataset")
data_dir = "/home/jiaming/scirdb/data/"
note_sentences = pd.read_pickle(data_dir + 'note_sentences.pkl')

# Load list of trained models
mypath = "/share/pi/rubin/jiaming/models"
trained_models = [f for f in listdir(mypath) if isfile(join(mypath, f))]

def label_sentences(corpus, label_type):
    labeled = []
    for i, v in enumerate(corpus):
        label = label_type + '_' + str(i)
        labeled.append(doc2vec.TaggedDocument(v.split(), [label]))
    return labeled

if args.model=="doc2vec":
    print("Labeling sentences.")
    all_merged_notes = label_sentences(note_sentences, 'SENT')

# Set hyperparameters
epochs=[5, 10, 30]
alpha=[0.0025, 0.025, 0.25]
#window=[3, 5]
sample=[1e-4, 1e-2, 0]
ns_exponent=[0.75]

for e in epochs:
    for a in alpha:
        #for w in window:
            for s in sample:
                for ns in ns_exponent:
                    args.epochs = e
                    args.alpha = a
                    #args.window = w
                    args.sample = s
                    args.ns_exponent = ns

                    if args.model == 'doc2vec':
                        model_name = "doc2vec_v{}_a{}_e{}_t{}_w{}_s{}_ns{}".format(args.size, args.alpha, 
                                                                                   args.epochs, args.alg, 
                                                                                   args.window, args.sample, 
                                                                                   args.ns_exponent)
                        if model_name + ".model" in trained_models:
                            print (model_name + "already trained. Passing.")
                            continue

                        print("Training model: " + model_name)
                        model_dbow = Doc2Vec(negative=5, min_count=10, 
                                             dm=args.alg, vector_size=args.size, 
                                             alpha=args.alpha, min_alpha=args.alpha,
                                             window=args.window, sample=args.sample,
                                             ns_exponent=args.ns_exponent, workers=10)
                        model_dbow.build_vocab(all_merged_notes)
                        for epoch in range(args.epochs):
                            model_dbow.train(utils.shuffle([x for x in tqdm(all_merged_notes)]),
                                             total_examples=len(all_merged_notes), epochs=1)
                            model_dbow.alpha -= 0.002
                            model_dbow.min_alpha = model_dbow.alpha

                        # Save the particular model
                        model_dbow.save("/share/pi/rubin/jiaming/models/{}.model".format(model_name))
                        model_dbow.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)

                    elif args.model == 'fasttext':
                        model_name = "fasttext_v{}_a{}_e{}_t{}_w{}_s{}_ns{}".format(args.size, args.alpha,
                                                                                    args.epochs, args.alg, 
                                                                                    args.window, args.sample, 
                                                                                    args.ns_exponent)
                        if model_name + ".model" in trained_models:
                            print (model_name + "already trained. Passing.")
                            continue
                        
                        print("Training model: " + model_name)
                        model = FastText(min_count=10, negative=5, 
                                         size=args.size, sg = args.alg,
                                         alpha=args.alpha, min_alpha=args.alpha,
                                         window=args.window, sample=args.sample,
                                         ns_exponent=args.ns_exponent, workers=10)
                        model.build_vocab(sentences=[word_tokenize(line.strip()) for line in note_sentences])
                        for epoch in range(args.epochs):
                            model.train(sentences=utils.shuffle([x for x in tqdm(note_sentences)]), 
                                        total_examples=len(note_sentences), epochs=1)
                            model.alpha -= 0.002
                            model.min_alpha = model.alpha

                        # Save the particular model
                        model.save("/share/pi/rubin/jiaming/models/{}.model".format(model_name))

