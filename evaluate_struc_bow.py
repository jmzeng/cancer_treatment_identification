# Script for evaluating the models systematically
import pandas as pd
import numpy as np
from itertools import compress
import importlib
import argparse

from collections import Counter

from sklearn import utils
from sklearn.preprocessing import minmax_scale
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.model_selection import train_test_split
#from imblearn.over_sampling import SMOTE

import re
import utils as ut


parser = argparse.ArgumentParser(description='Process model parameters')
parser.add_argument('-ct', '--cancer-type', type=str, default="prostate", 
                    help='model used for training')
parser.add_argument('-t', '--to-combine', type=int, default=1, 
                    help='whether to combine the treatments. 0: False. 1: True')
parser.add_argument('-c', '--min-count', type=int, default=50, 
                    help='minimum count for treatments')
parser.add_argument('-d', '--data-dir', type=str, default="/home/jiaming/scirdb/", 
                    help='data directory for saving the models')
parser.add_argument('-o', '--output-dir', type=str, default="/share/pi/rubin/jiaming/nlp_results/", 
                    help='model directory')
args = parser.parse_args()

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
        
columns = ["model", "train_accuracy"] + train_metrics + ["accuracy", "accuracy_lower", "accuracy_upper"] + test_metrics
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
data_train = ut.load_relevant_notes(data_train, notes, treatment_terms, embedding=False)
data_test = ut.load_relevant_notes(data_test, notes, treatment_terms, embedding=False)

# Turn some variables into one-hot vectors
data_train = ut.data_onehot(data_train, args.cancer_type)
data_test = ut.data_onehot(data_test, args.cancer_type)

# Normalize structured data
X_structured_train = minmax_scale(data_train[structured])
y_train = data_train.Treatment.to_numpy().astype('int')

X_structured_test = minmax_scale(data_test[structured])
y_test = data_test.Treatment.to_numpy().astype('int')


print("Evaluating models on structured data...")
results = ut.evaluate_models(X_structured_train, X_structured_test, 
                          y_train, y_test, "structured", args.cancer_type)
results2 = pd.DataFrame(results, columns=columns)
df_results = df_results.append(results2, ignore_index = True)
df_results.to_csv(args.output_dir + "{}_results.csv".format(args.cancer_type))


## Building BOW models
tfidf_vect = TfidfVectorizer(max_features=1000, ngram_range=(1,1))
X_train_tfidf = tfidf_vect.fit_transform(data_train.merged_notes)
X_test_tfidf = tfidf_vect.transform(data_test.merged_notes)


print("Evaluating models on BOW...")
results = ut.evaluate_models(X_train_tfidf.toarray(), X_test_tfidf.toarray(),
                              y_train, y_test, "bow", args.cancer_type)
results2 = pd.DataFrame(results, columns=columns)
df_results = df_results.append(results2, ignore_index = True)
df_results.to_csv(args.output_dir + "{}_results.csv".format(args.cancer_type))


print("Evaluating models on structured+bow...")
# Combine the two vectors together
X_combined_train = ut.concat_vectors(X_structured_train, X_train_tfidf.toarray())
X_combined_test = ut.concat_vectors(X_structured_test, X_test_tfidf.toarray())

results = ut.evaluate_models(X_combined_train, X_combined_test,
                             y_train, y_test, "structured+bow", args.cancer_type)
results2 = pd.DataFrame(results, columns=columns)
df_results = df_results.append(results2, ignore_index = True)
df_results.to_csv(args.output_dir + "{}_results.csv".format(args.cancer_type))

print("Finished.")


# print("Evaluating models with SMOTE on structured...")
# sm = SMOTE(random_state=42)
# X_structured_res, y_res = sm.fit_resample(X_structured_train, y_train)
# print('Resampled dataset shape %s' % Counter(y_res))

# results2 = ut.evaluate_models(X_structured_res, X_structured_valid, X_structured_test,
#                             y_res, y_valid, y_test, "structured-smote", treatments, args)
# results2 = pd.DataFrame(results2, columns=results.columns)
# results = results.append(results2, ignore_index = True)
# results.to_csv(args.data_dir + "results/{}_{}_results.csv".format(args.cancer_type, args.to_combine))

# print("Evaluating models with SMOTE on bow...")
# X_unstructured_res, y_res = sm.fit_resample(X_train_tfidf.toarray(), y_train)
# print('Resampled dataset shape %s' % Counter(y_res))

# results2 = ut.evaluate_models(X_unstructured_res, X_valid_tfidf.toarray(), X_test_tfidf.toarray(),
#                               y_res, y_valid, y_test, "bow-smote", treatments, args)
# results2 = pd.DataFrame(results2, columns=results.columns)
# results = results.append(results2, ignore_index = True)
# results.to_csv(args.data_dir + "results/{}_{}_results.csv".format(args.cancer_type, args.to_combine))


