# Script for evaluating the models systematically
import pandas as pd
import numpy as np
from gensim.models import Doc2Vec, doc2vec, FastText
from gensim.models.doc2vec import TaggedDocument
from itertools import compress
import importlib
import argparse

from collections import Counter

from sklearn import utils
from sklearn.preprocessing import minmax_scale
from sklearn import metrics
from sklearn.model_selection import train_test_split

from nltk.tokenize import word_tokenize
import re
import utils as ut


parser = argparse.ArgumentParser(description='Process model parameters')
parser.add_argument('-ct', '--cancer-type', type=str, default="prostate", 
                    help='model used for training')
parser.add_argument('-t', '--to-combine', type=int, default=1, 
                    help='whether to combine the treatments. 0: False. 1: True')
parser.add_argument('-c', '--min-count', type=int, default=70, 
                    help='minimum count for treatments')
parser.add_argument('-d', '--data-dir', type=str, default="/home/jiaming/scirdb/", 
                    help='data directory')
parser.add_argument('-md', '--model-dir', type=str, default="/share/pi/rubin/jiaming/models/", 
                    help='model directory')
parser.add_argument('-o', '--output-dir', type=str, default="/share/pi/rubin/jiaming/nlp_results/", 
                    help='model directory')
parser.add_argument('-m', '--model', type=str, default="fasttext", 
                    help='embedding model')
parser.add_argument('-a', '--alg', type=int, default=0, 
                    help='algorithm for embedding model')
parser.add_argument('-vs', '--vector-size', type=int, default=100, 
                    help='vector size of embedding model')
parser.add_argument('-w', '--window', type=int, default=3, 
                    help='window of maximum distance between words')
args = parser.parse_args()


# Create txt file for recording or results
missing_models = open(args.output_dir + "missing_models_{}_t{}_vs{}_w{}.txt".format(
    args.model, args.alg, args.vector_size, args.window), "w+")

# Load the dataset
notes = pd.read_pickle(args.data_dir + 'data/related_notes.pkl')
data_train, data_test, treatments = ut.load_dataset(args.data_dir, args.cancer_type, args.min_count, args.to_combine)

if args.cancer_type == "prostate":
    structured = ['age', 'sex', 'clinical_stage', 'radiation_eb',
                   'radiation_bt', 'prostatectomy', 'hormonal',
                   'race_asian', 'race_black', 'race_other', 'race_white',
                   'ethnicity_hispanic', 'ethnicity_nonhispanic']
elif args.cancer_type == "oropharynx":
    structured = ['age', 'sex', 'clinical_stage', 'radiation_eb',
                   'radiation_bt', 'oropharynx_surgery', 'chemo',
                   'race_asian', 'race_black', 'race_other', 'race_white',
                   'ethnicity_hispanic', 'ethnicity_nonhispanic']
elif args.cancer_type == "esophagus":
    structured = ['age', 'sex', 'clinical_stage', 'radiation_eb',
                   'radiation_bt', 'esophagus_surgery', 'chemo',
                   'race_asian', 'race_other', 'race_white',
                   'ethnicity_hispanic', 'ethnicity_nonhispanic']

# Create df and file for saving results
metrics = ["precision", "recall", "f1"]
ci = ["", "lower", "upper"]
train_metrics = []
test_metrics = []
for treatment in treatments:
    train_metrics += ["train_{}_{}".format(treatment, metric) for metric in metrics]
    test_metrics += ["{}_{}_{}".format(treatment, metric, c) for metric in metrics for c in ci]
        
columns = ["model", "alg", "vs", "epochs", "alpha", "window", "sample", "ns_exponent", 
           "method", "train_accuracy"] + train_metrics + ["accuracy", "accuracy_lower", "accuracy_upper"] + test_metrics
df_results = pd.DataFrame(columns=columns)

# Record the dataset size 
print("Size of training: " + str(data_train.shape) + '\n')
print(*list(zip(range(len(treatments)), treatments, [len(data_train[data_train.Treatment == i]) for i in range(len(treatments))])), sep="\n")
print("Size of testing:" + str(data_test.shape) + '\n')
print(*list(zip(range(len(treatments)), treatments, [len(data_test[data_test.Treatment == i]) for i in range(len(treatments))])), sep="\n")

# Load relevant treatment terms
terms = pd.read_excel(args.data_dir + 'data/cancer_term_dictionary.xlsx')
terms.columns = ['vocab', 'category']
terms = terms[terms.category.isin([1, 2, 3, 6])]
treatment_terms = [word.lower() for word in terms.vocab]

# Load the notes data from the associated note_ids
data_train = ut.load_relevant_notes(data_train, notes, treatment_terms, embedding=True)
data_test = ut.load_relevant_notes(data_test, notes, treatment_terms, embedding=True)

# Turn some variables into one-hot vectors
data_train = ut.data_onehot(data_train, args.cancer_type)
data_test = ut.data_onehot(data_test, args.cancer_type)

# Normalize structured data
X_structured_train = minmax_scale(data_train[structured])
y_train = data_train.Treatment.to_numpy().astype('int')

X_structured_test = minmax_scale(data_test[structured])
y_test = data_test.Treatment.to_numpy().astype('int')

## Perform analysis on structured + notes dataset
# Set hyperparameters
epochs=[5, 10, 30]
alpha=[0.0025, 0.025, 0.25]
sample=[1e-4, 1e-2, 0]
ns_exponent=[0.75]

for e in epochs:
    for a in alpha:
        for s in sample:
            for ns in ns_exponent:
                # Record the parameters
                args.epochs = e
                args.alpha = a
                args.sample = s
                args.ns_exponent = ns

                # Set column prefix
                col_prefix = [args.model, args.alg, args.vector_size, args.epochs, 
                              args.alpha, args.window, args.sample, args.ns_exponent]

                # Read in results file to see what has not been run
                try:
                    df_results = pd.read_csv(args.output_dir + "{}_{}_t{}_vs{}_w{}_results.csv".format(
                        args.cancer_type, args.model, args.alg, args.vector_size, args.window), index_col=0)
                    #print(df_results.head())
                    finished_runs = len([run for run in df_results.iloc[:,0:8].to_numpy() if set(run) == set(col_prefix)])
                    #print(finished_runs)
                    if finished_runs == 10: # 5 runs each, 2 sets
                        print(col_prefix, "already ran.")
                        continue
                except:
                    pass

                if args.model == 'doc2vec':
                    model_name = "doc2vec_v{}_a{}_e{}_t{}_w{}_s{}_ns{}.model".format(args.vector_size, args.alpha, 
                                                                               args.epochs, args.alg, 
                                                                               args.window, args.sample, 
                                                                               args.ns_exponent)
                    try:
                        print ("Loading model " + model_name)
                        model = Doc2Vec.load(args.model_dir + model_name)
                    except:
                        missing_models.write(model_name + "\n")
                        missing_models.flush()
                        continue

                     # Turn notes into vectors
                    print ("Inferring doc2vec vectors...")
                    merged_notes_train = ut.label_sentences(data_train.merged_notes, 'Train')
                    merged_notes_test = ut.label_sentences(data_test.merged_notes, 'Test')

                    notes_train_vectors = ut.vec_for_learning(model, merged_notes_train)
                    notes_test_vectors = ut.vec_for_learning(model, merged_notes_test)

                elif args.model == 'fasttext':
                    model_name = "fasttext_v{}_a{}_e{}_t{}_w{}_s{}_ns{}.model".format(args.vector_size, args.alpha,
                                                                                args.epochs, args.alg, 
                                                                                args.window, args.sample, 
                                                                                args.ns_exponent)
                    try:
                        print ("Loading model " + model_name)
                        model = FastText.load(args.model_dir + model_name)
                    except:
                        missing_models.write(model_name + "\n")
                        missing_models.flush()
                        continue

                    # Turn notes into vectors
                    print ("Inferring fasttext vectors...")
                    merged_notes_train = ut.word_tokenizer(data_train.merged_notes)
                    merged_notes_test = ut.word_tokenizer(data_test.merged_notes)

                    notes_train_vectors = ut.fasttext_vectors(model, merged_notes_train, args.vector_size)
                    notes_test_vectors = ut.fasttext_vectors(model, merged_notes_test, args.vector_size)

                # Evaluating for just doc2vec models
                print ("Evaluating models with just embedding model...")
                results = ut.evaluate_models(notes_train_vectors, notes_test_vectors,
                                              y_train, y_test, args.model, args.cancer_type)
                results2 = pd.DataFrame([col_prefix + r for r in results], columns=columns)
                df_results = df_results.append(results2, ignore_index = True, sort=False)
                df_results.to_csv(args.output_dir + "{}_{}_t{}_vs{}_w{}_results.csv".format(
                    args.cancer_type, args.model, args.alg, args.vector_size, args.window))

                # Evaluating for structured and unstructured
                # Combine the structured and unstructured data together
                print ("Evaluating models for both structured+unstructured data...")
                X_combined_train = ut.concat_vectors(X_structured_train, notes_train_vectors)
                X_combined_test = ut.concat_vectors(X_structured_test, notes_test_vectors)

                results = ut.evaluate_models(X_combined_train, X_combined_test,
                                              y_train, y_test, "structured+" + args.model, args.cancer_type)
                results2 = pd.DataFrame([col_prefix + r for r in results], columns=columns)
                df_results = df_results.append(results2, ignore_index = True, sort=False)
                df_results.to_csv(args.output_dir + "{}_{}_t{}_vs{}_w{}_results.csv".format(
                    args.cancer_type, args.model, args.alg, args.vector_size, args.window))

                        
missing_models.close()
print("Finished.")

