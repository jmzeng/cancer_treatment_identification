import pandas as pd
import numpy as np
import re
import random
import matplotlib.pyplot as plt
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import names

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.multiclass import unique_labels
from sklearn import metrics

from gensim.models import Doc2Vec, doc2vec
from gensim.models.doc2vec import TaggedDocument

from xgboost import XGBClassifier


REPLACE_NO_SPACE = re.compile("(\@)|(\#)|(\%)|(\¿)|(\xa0)")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)|(--)")
max_line = 30


def clean_text(df, text_field, new_text_field_name):
    df.loc[:, new_text_field_name] = df.loc[:, text_field].str.lower()
    df.loc[:, new_text_field_name] = df.loc[:, new_text_field_name].apply(lambda elem: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?|(\¿)|(\xa0)", "", elem)) 
    df.loc[:, new_text_field_name] = df.loc[:, new_text_field_name].apply(lambda elem: re.sub(r"(<br\s*/><br\s*/>)|(\-)|(--)", " ", elem)) 
    
    # remove numbers
    df.loc[:, new_text_field_name] = df.loc[:, new_text_field_name].apply(lambda elem: re.sub(r"\d+", "", elem))
    
    # remove clinical terms
    clinical_terms = ["dr.", "dr ", " np", "pa c", " md", "p.a.", "m.d.", "physician", 
                      "admission source", "stanford", "pasteur drive", "primary doctor",
                     "urology", "menlo", "street"]
    return df


def preprocess_meds(data):
	data = data.dropna()
	data = [REPLACE_NO_SPACE.sub("", str(line).lower()) for line in data.name]
	data = [REPLACE_WITH_SPACE.sub(" ", line) for line in data]
	data = [line.strip() for line in data]
	note_sentences = []
	for i in range(len(data)):
		note_sentences += (re.split(r'\s{1}', data[i]))[0:-1]
	note_sentences = list(filter(None, note_sentences))
	note_sentences = [line.strip() for line in note_sentences]
	note_sentences = [line for line in note_sentences if (len(line) > 4)]
	return note_sentences


def count_treatments(df):
    unique_treatments = [set(x) for x in set(tuple(x) for x in df.Treatment)]
    unique_treatments = list(set(tuple(sorted(l)) for l in unique_treatments))
    frequency = [0] * len(unique_treatments)
    for i in range(0, len(unique_treatments)):
        for j in range(len(df)):
            if set(df.Treatment.iloc[j]) == set(unique_treatments[i]):
                frequency[i] += 1

    treatment_types = pd.DataFrame(data = {'treatment_type': unique_treatments, 
                                           'frequency': frequency})
    treatment_types = treatment_types.sort_values(by='frequency')
    print(treatment_types)


def load_dataset(data_dir, cancer_type, min_freq=50, combined=0, drop_death=True):
    data = pd.read_pickle(data_dir + "data/{}_data.pkl".format(cancer_type))
    if cancer_type == "processed":
        types = ["PROSTATE", "UGI TRACT", "HEAD AND NECK"]
        data = data[data.cancer_site.isin(types)]
        print(data.shape)
        # Turning cancer_site into categories
        test = [tuple(x) for x in data.cancer_site]
        for i in range(len(types)):
            category_bool = [y == tuple(types[i]) for y in test]
            data.loc[category_bool, 'cancer_site'] = i
            
        # Use the combined testing data from the other
        test_pat_ids1 = pd.read_pickle(data_dir + 'data/test_pat_ids_prostate.pkl')
        test_pat_ids2 = pd.read_pickle(data_dir + 'data/test_pat_ids_esophagus.pkl')
        test_pat_ids3 = pd.read_pickle(data_dir + 'data/test_pat_ids_oropharynx.pkl')
        test_pat_ids = pd.concat([test_pat_ids1, test_pat_ids2, test_pat_ids3])
        
    else:
        # Load the existing pat_ids
        test_pat_ids = pd.read_pickle(data_dir + 'data/test_pat_ids_{}.pkl'.format(cancer_type))

    data, treatments = preprocess_data(data, cancer_type, min_freq, combined)
    
    data_test = data[data.PATIENT_ID.isin(test_pat_ids)]
    data_train = data[~data.PATIENT_ID.isin(test_pat_ids)]

    # Drop extra columns
    todrop = ['DIAGNOSIS_ID', 'PATIENT_ID', 'pathological_stage', 'cancer_site', 'merged_meds']
    if drop_death:
        todrop += ['death']
        
    # Drop based on cancer type
    if cancer_type == "prostate":
        todrop += ['esophagus_surgery', 'oropharynx_surgery']
    elif cancer_type == "oropharynx":
        todrop += ['prostatectomy', 'esophagus_surgery', 'hormonal']
    elif cancer_type == "esophagus":
        todrop += ['prostatectomy', 'oropharynx_surgery', 'hormonal']
        
    data_train = data_train.drop(todrop, axis=1)
    data_test = data_test.drop(todrop, axis=1)
    
    return data_train, data_test, treatments


def preprocess_data(data, cancer_type, min_freq=50, combined=0):
    # Count the number of unique treatment combination types
    unique_treatments = [list(x) for x in set(tuple(x) for x in data.Treatment)]
    unique_treatments = list(set(tuple(sorted(l)) for l in unique_treatments))
    frequency = [0] * len(unique_treatments)
    note_count = [0] * len(unique_treatments)
    for i in range(0, len(unique_treatments)):
        for j in range(0, len(data)):
            if set(data.Treatment.iloc[j]) == set(unique_treatments[i]):
                frequency[i] += 1
                note_count[i] += data.n_notes.iloc[j]

    treatment_types = pd.DataFrame(data = {'treatment_type': unique_treatments, 
                                           'frequency': frequency,
                                           'total_notes': note_count})
    treatment_types = treatment_types.sort_values(by='frequency')
    print(treatment_types)

    # Filtering out data we don't want to use for treatment and cancer site
    inference_treatments = treatment_types.loc[treatment_types['frequency'] >= min_freq, 'treatment_type']
    inference_treatments = [tuple(x) for x in inference_treatments]
    test = [tuple(x) for x in data.Treatment]
    booleans = [y in inference_treatments for y in test]
    inference_data = data.loc[booleans]
    print("# of Patients:", inference_data.PATIENT_ID.nunique())

    # Turning treatments into categories
    test = [tuple(x) for x in inference_data.Treatment]
    for i in range(len(inference_treatments)):
        category_bool = [y == tuple(inference_treatments[i]) for y in test]
        #print (inference_treatments[i], i)
        inference_data.loc[category_bool, 'Treatment'] = i
    
    # Hack for combining the treatments
    if combined == 1 and cancer_type.startswith("prostate"):
        inference_data.loc[inference_data.Treatment == 1, 'Treatment'] = 0
        inference_data.loc[inference_data.Treatment == 2, 'Treatment'] = 1
        del inference_treatments[1]
        
        inference_treatments[0] = "rad"
        inference_treatments[1] = "surg"
    
    elif combined == 1 and cancer_type == "oropharynx":
        inference_data.loc[inference_data.Treatment == 1, 'Treatment'] = 0
        inference_data.loc[inference_data.Treatment == 2, 'Treatment'] = 0
        inference_data.loc[inference_data.Treatment == 3, 'Treatment'] = 1
        del inference_treatments[1:3]
        
        inference_treatments[0] = "surg"
        inference_treatments[1] = "chemorad"
        
    elif combined == 1 and cancer_type == "esophagus":
        inference_data.loc[inference_data.Treatment == 2, 'Treatment'] = 0
        del inference_treatments[1]
        
        inference_treatments[0] = "surg"
        inference_treatments[1] = "chemorad"
        
    elif combined == 1 and cancer_type == "processed":
        inference_data.loc[inference_data.Treatment == 2, 'Treatment'] = 0
        inference_treatments[0] = "Surgery+others"
        del inference_treatments[1]
    
    # Change the hormonal and chemo to integers
    inference_data.loc[:, 'hormonal'] = [len(x) for x in inference_data['hormonal']]
    inference_data.loc[:, 'chemo'] = [len(x) for x in inference_data['chemo']]
    
    return inference_data, inference_treatments


def load_relevant_notes(data, notes, treatment_terms, embedding=True):
    merged_notes = []
    for i in range(len(data)):
        ids = data.note_ids.iloc[i]
        selected_notes = notes[notes['NOTE_ID'].isin(ids)]
        selected_notes = selected_notes.sort_values('ENCOUNTER_DATE')

        pat_sentences = tokenize_sent(selected_notes.NOTE, treatment_terms)
        if embedding:
            merged_notes.append(pat_sentences)
        else:
            pat_sentences = list(set(pat_sentences))
            seperator = ' '
            merged_notes.append(seperator.join(pat_sentences))
    data = data.drop('note_ids', axis=1)
    data["merged_notes"] = merged_notes
    return data


def tokenize_sent(data, treatment_terms):
    data = [line.strip() for line in data]
    note_sentences = []
    for i in range(len(data)):
        sentences = (re.split(r'\s{3}', data[i]))
        sentences = list(filter(None, sentences))
        sentences = [sent_tokenize(line.strip()) for line in sentences]
        sentences = [item for sublist in sentences for item in sublist][3:-1]
        sentences = [line for line in sentences if (len(line) > max_line)]
        sentences = [sent for sent in sentences if any(treat in sent for treat in treatment_terms)]
        note_sentences.append(sentences)
    
    note_sentences = [item for sublist in note_sentences for item in sublist]
    return note_sentences


def concat_vectors(array1, array2):
    array1 = normalize(array1)
    array2 = normalize(array2)
    if len(array1) != len(array2):
        raise Exception("Arrays should have same number of rows.")
    vectors = np.zeros((len(array1), array1.shape[1] + array2.shape[1]))
    for i in range(0, len(array1)):
        vectors[i] = np.append(array1[i,],array2[i,])
    return vectors


def data_onehot(X, cancer_type):
    # Turn race and ethnicity to one-hot
    X.loc[X.race == 0, "race"] = "asian"
    X.loc[X.race == 1, "race"] = "black"
    X.loc[X.race == 2, "race"] = "white"
    X.loc[X.race == 3, "race"] = "other"
    X.loc[X.race == 4, "race"] = "unknown"
    
    race_onehot = pd.get_dummies(X['race'], prefix='race')
    
    X.loc[X.ethnicity == 0, "ethnicity"] = "hispanic"
    X.loc[X.ethnicity == 1, "ethnicity"] = "nonhispanic"
    X.loc[X.ethnicity == 2, "ethnicity"] = "unknown"

    ethnicity_onehot = pd.get_dummies(X['ethnicity'], prefix='ethnicity')

    X = pd.concat([X, race_onehot, ethnicity_onehot], axis=1)
    
    # Drop extra variables
    if cancer_type == "esophagus":
        to_drop = ['n_notes', "hospitalization", "ED", "HP", "Discharge", "Progress", 
                   "Treatment_Planning", "race", "ethnicity"]
    elif cancer_type == "oropharynx":
        to_drop = ['n_notes', "hospitalization", "ED", "HP", "Discharge", "Progress", 
                   "Treatment_Planning", "race", "ethnicity", "ethnicity_unknown"]
    else:
        to_drop = ['n_notes', "hospitalization", "ED", "HP", "Discharge", "Progress", 
                   "Treatment_Planning", "race", "ethnicity", "race_unknown", "ethnicity_unknown"]
    X = X.drop(columns = to_drop)
    return X

@ignore_warnings(category=ConvergenceWarning)
def train_classifiers(X_train, y_train, nfolds=5):
    # Estabilish baseline of structured dataset
    lr = LogisticRegression(max_iter=5000, solver = 'saga', class_weight='balanced')
    rf = RandomForestClassifier(n_estimators=300)
    xgb = XGBClassifier(n_estimators=300)
    mlp = MLPClassifier(max_iter=5000)
    svc = SVC(max_iter=5000)
    
    # Parameters for logistic regression
    grid_lr = {'C': np.logspace(-4, 4, 10)}
    
    # Parameters for Random forest
    grid_rf = {'max_features': ['sqrt', 'log2'],
                'max_depth' : [4,5,6,7,8],
                'criterion' :['gini', 'entropy']}
    
    # Parameters for xgboost
    grid_xgb = {'max_depth': np.arange(3, 10, 2),
                'subsample': [0.8, 0.9, 1],
                'colsample_bytree': [0.3, 0.5, 0.8]}
    
    # Parameters for SVC
    grid_svc = {'C': [0.1,1, 10, 100], 
                'gamma': [1,0.1,0.01,0.001],
                'kernel': ['rbf', 'poly', 'sigmoid']}
    
    # Parameters for MLP
    grid_mlp = {'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
               'activation': ['tanh', 'relu'],
               'solver': ['sgd', 'adam'],
               'alpha': [0.0001, 0.05],
               'learning_rate': ['constant','adaptive']}
        
    # Create grid search and train model
    print("Training LR...")
    clf_lr = GridSearchCV(lr, param_grid = grid_lr, cv=nfolds, n_jobs=-1).fit(X_train, y_train)
    print("Training RF...")
    clf_rf = GridSearchCV(rf, param_grid = grid_rf, cv=nfolds, n_jobs=-1).fit(X_train, y_train)
    print("Training XGB...")
    clf_xgb = GridSearchCV(xgb, param_grid = grid_xgb, cv=nfolds, n_jobs=-1).fit(X_train, y_train)
    print("Training MLP...")
    clf_mlp = GridSearchCV(mlp, param_grid = grid_mlp, cv=nfolds, n_jobs=-1).fit(X_train, y_train)
    print("Training SVC...")
    clf_svc = GridSearchCV(svc, param_grid = grid_svc, cv=nfolds, n_jobs=-1).fit(X_train, y_train)
    
    return clf_lr, clf_rf, clf_xgb, clf_mlp, clf_svc


def class_accuracy(X, y, model):
    treatment_scores = []
    
    # Add full report of classification
    y_pred = model.predict(X)
    report = classification_report(y, y_pred, output_dict=True)

    # Append the scores
    treatment_scores += list(report['0'].values())[0:3]
    try:
        treatment_scores += list(report['1'].values())[0:3]
    except:
        # Catch exception when sampling does not pick up on any of one class
        #print (report)
        treatment_scores += [1.0, 1.0, 1.0]
        #print (treatment_scores)
        
    return treatment_scores


# Creating bootstrapped CI on test set
def bootstrap_ci(X, y, model, repetitions = 5000, alpha = 0.05, random_state=None):
    print(repetitions)
    stats_ci = []
    bootstrap_sample_size = X.shape[0]

    accuracy = []
    class_stats = []
    for i in range(repetitions):
        bootstrap_sample = np.random.randint(bootstrap_sample_size, size=bootstrap_sample_size)
        X_samp = X[bootstrap_sample]
        y_samp = y[bootstrap_sample].astype('int')
        
        accuracy.append(model.score(X_samp, y_samp))
        class_stats.append(class_accuracy(X_samp, y_samp, model))
    class_stats = np.asarray(class_stats)
    
    # Get stats for mean and CI
    stats_ci += [np.mean(accuracy), np.percentile(accuracy, alpha/2*100), np.percentile(accuracy, 100-alpha/2*100)]
    try:
        for j in range(class_stats.shape[1]):
            stats_ci += [np.mean(class_stats[:,j]), np.percentile(class_stats[:,j], alpha/2*100), np.percentile(class_stats[:,j], 100-alpha/2*100)]
    except:
        print (class_stats.shape)
        #print (class_stats)
        
    return stats_ci

# def bootstrap_ci(X, y, features, model, repetitions = 1000, alpha = 0.05, random_state=None): 
#     stats_ci = []
#     df = pd.DataFrame(np.concatenate((X, y.reshape(y.shape[0],-1)), axis=1), columns=features + ["Treatment"])
#     bootstrap_sample_size = len(df)
#     print(bootstrap_sample_size)
    
#     accuracy = []
#     class_stats = []
#     for i in range(repetitions):
#         bootstrap_sample = df.sample(n = bootstrap_sample_size, replace = True, random_state = random_state)
#         print(bootstrap_sample.shape)
#         X_samp = bootstrap_sample[features]
#         y_samp = bootstrap_sample['Treatment'].astype('int')
        
#         accuracy.append(model.score(X_samp, y_samp))
#         class_stats.append(class_accuracy(X_samp, y_samp, model))
#     class_stats = np.asarray(class_stats)
    
#     # Get stats for mean and CI
#     stats_ci += [np.mean(accuracy), np.percentile(accuracy, alpha/2*100), np.percentile(accuracy, 100-alpha/2*100)]
#     for j in range(class_stats.shape[1]):
#         stats_ci += [np.mean(class_stats[:,j]), np.percentile(class_stats[:,j], alpha/2*100), np.percentile(class_stats[:,j], 100-alpha/2*100)]
        
#     return stats_ci


def evaluate_models(X_train, X_test, y_train, y_test, method, cancer_type):
    if cancer_type == "prostate":
        reps=1000
    else:
        reps=5000
    
    clf_lr, clf_rf, clf_xgb, clf_mlp, clf_svc = train_classifiers(X_train, y_train, nfolds=10)
    model_dict = {"lr": clf_lr,
              "rf": clf_rf,
              "xgb": clf_xgb,
              "mlp": clf_mlp, 
              "svc": clf_svc}
    
    results = []
    for key in model_dict.keys():
        temp_result = []
        model_name = "{}-{}".format(method, key)
        train_score = model_dict[key].score(X_train, y_train)
        train_treatment_scores = class_accuracy(X_train, y_train, model_dict[key])
        temp_result += [model_name, train_score] + train_treatment_scores
        
        test_results = bootstrap_ci(X_test, y_test, model_dict[key], repetitions = reps)
        temp_result += test_results
        results.append(temp_result)

    return results


# Label merged sentences and turn into vectors
def label_sentences(corpus, label_type):
    labeled = []
    for i, v in enumerate(corpus):
        doc = " ".join(v)
        label = label_type + '_' + str(i)
        labeled.append(doc2vec.TaggedDocument(doc.split(), [label]))
    return labeled


# Extract vectors from model for learning
def vec_for_learning(model, tagged_docs):
    vectors = [model.infer_vector(doc.words) for doc in tagged_docs]
    vectors = np.stack(vectors, axis=0)
    return vectors


# Label merged sentences and turn into vectors for fasttext
def word_tokenizer(merged_notes):
    tokens = []
    for note in merged_notes:
        doc_tokens = []
        words = [word_tokenize(line.strip()) for line in note]
        tokens.append(words)
        #tokens += words
    return tokens


# Extract vectors from model for learning
def fasttext_vectors(model, docs, vector_size):
    vectors = []
    for doc in docs:
        doc = list(filter(None, doc)) 
        doc_vec = []
        # Hack: replace empty list with zeros
        if len(doc) == 0:
            vectors.append(np.zeros(vector_size))
        else:
            for sent in doc:
                sent_vec = [model.wv[word] for word in sent if word in model.wv]
                sent_vec = np.mean(np.array(sent_vec), axis=0)
                doc_vec.append(sent_vec)
            doc_vec = np.mean(np.array(doc_vec), axis=0)
            vectors.append(doc_vec)

    vectors = np.stack(vectors, axis=0)
    return vectors