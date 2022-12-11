import collections
import csv
import numpy as np
import pandas as pd
import math
import re
import itertools
from itertools import islice
from scipy.sparse import dok_matrix
import sklearn
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from nltk.stem import PorterStemmer
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel

import scipy.stats as stats

#TASK 1

# This is the section for IR evaluation where different measures like precision, recall, r-precision, Average precision and 
# Normalized Dsitributive Cumulative Gain was calculated on 6 different systems with 10 query each
class EVAL:

    # converts the system results file into an embedded dictionary of system, query number and a list of document number,
    # rank of the particular doc and score respectively
    def retrieved_docs(filename):

        results = collections.defaultdict(lambda: collections.defaultdict(list))
        with open(filename) as f:
            f.readline()
            for row in f:
                system_number, query_number, doc_number, rank_doc, score = row.strip().split(",")
                results[int(system_number)][int(query_number)].append([int(doc_number), int(rank_doc), float(score)])
            
        return results

    # helper function to extract the doc number of from the list of document number, rank of the particular doc and score of a particular system
    def Extract(lst):
        return [int(item[0]) for item in lst]

    # converts the qrels file into an embedded dictionary of query number, doc number and the relevance of each doc for that query
    def relevant_docs(filename):

        results = collections.defaultdict(lambda: collections.defaultdict(list))
        with open(filename, 'r') as f:
            f.readline()
            for row in f:
                query_number, doc_number, relevance = row.strip().split(",")
                results[int(query_number)][int(doc_number)] = int(relevance)
            
        return results

    # calculates what fraction of retrieved docs are relevant
    def precision(retrieved_docs, relevant_docs):
        relevant_retrieved = relevant_docs.keys() & retrieved_docs
        precision = len(relevant_retrieved)/len(retrieved_docs)
        return round(precision, 3)

    # calculates what fraction of relevant docs are retrieved
    def recall(retrieved_docs, relevant_docs):
        relevant_retrieved = relevant_docs.keys() & retrieved_docs
        recall = len(relevant_retrieved)/len(relevant_docs)
        return round(recall, 3)

    # calculates average precision 
    def AP(retrieved_docs, relevant_docs):
        ap = 0 

        for k in retrieved_docs:
            if k[0] in relevant_docs.keys():
                precision_k = EVAL.precision(EVAL.Extract(retrieved_docs[:k[1]]), relevant_docs)
                ap += precision_k * 1
            else:
                ap += 0

        return round(ap/len(relevant_docs), 3)

    # Calculates nDCG by comparing DCG at each rank with the ideal DCG 
    def nDCG(retrieved_docs, relevant_docs):

        rel1 = relevant_docs[retrieved_docs[0][0]] if retrieved_docs[0][0] in relevant_docs.keys() else 0
        DCG = rel1
        for rank, doc_number in enumerate(retrieved_docs[1:]):
            if doc_number[0] in relevant_docs.keys():
                DCG += relevant_docs[doc_number[0]]/np.log2(rank+2)

        irel = list(relevant_docs.values())
        iDCG = irel[0]
        for idx, rel in enumerate(irel[1:]):
            iDCG += rel/np.log2(idx + 2)
            
        nDCG = DCG/iDCG
        # sys.exit(0)

        return round(nDCG, 3)

    # main implementation block for task 1
    def task_1():
        # storing the results of each measure in the format required
        with open('ir_eval.csv', 'w') as result:

            sys_retrieved_doc = EVAL.retrieved_docs("/Users/arnav/Desktop/Y4/ttds/cw2/system_results.csv")
            relevant_doc = EVAL.relevant_docs("/Users/arnav/Desktop/Y4/ttds/cw2/qrels.csv")

            writer = csv.writer(result, delimiter=",")
            writer.writerow(['system_number','query_number','P@10','R@50','r-precision','AP','nDCG@10','nDCG@20'])

            means = []
            for system,retrieved_doc in sys_retrieved_doc.items():
                for_mean = []

                # for 2-tailed t-test
                Ps = []
                rs = []
                rps = []
                aps = []
                d10 = []
                d20 = []
                for query,doc in retrieved_doc.items():

                    precision_10 = EVAL.precision(EVAL.Extract(doc[:10]), relevant_doc[query])

                    recall_50 = EVAL.recall(EVAL.Extract(doc[:50]), relevant_doc[query])

                    # precision at rank r(cut-off rank)
                    cut_off = doc[:len(relevant_doc[query])]
                    r_precision = EVAL.precision(EVAL.Extract(cut_off), relevant_doc[query])

                    ap = EVAL.AP(doc, relevant_doc[query])

                    rel_docs_10 = dict(itertools.islice(relevant_doc[query].items(), 10))
                    nDCG_10 = EVAL.nDCG(doc[:10], rel_docs_10)
                    rel_docs_20 = dict(itertools.islice(relevant_doc[query].items(), 20)) 
                    nDCG_20 = EVAL.nDCG(doc[:20], rel_docs_20)

                    writer.writerow([system,query,precision_10,recall_50,r_precision,ap,nDCG_10,nDCG_20])

                    for_mean.append([precision_10,recall_50,r_precision,ap,nDCG_10,nDCG_20])

                    # for 2-tailed t-test
                    Ps.append(precision_10)
                    rs.append(recall_50)
                    rps.append(r_precision)
                    aps.append(ap)
                    d10.append(nDCG_10)
                    d20.append(nDCG_20)
                
                means.append(np.mean(for_mean, axis=0))
                writer.writerow([system,"mean",round(means[system-1][0],3),round(means[system-1][1],3),round(means[system-1][2],3),
                round(means[system-1][3],3),round(means[system-1][4],3),round(means[system-1][5],3)])
        
        # # algorithms for round robin pvalue calculation. Does not work here though
        # # for 2-tailed t-test
        # # Generated the p-values for test of each system with another for each measure
        # for i in range(6):
        #     for j in range(5-i):
        #         print(stats.ttest_rel(d10[i], d10[i+j+1]))

# TASK 2

# This is the section for Text Analysis where the word-level comparisons are performed using Mutual Information and Chi-squared values
# Topic-level comaprisons are also performed using Lda modelling

class ANALYSIS:

    # preprosses the input text removing all punctuations, case-folding, stopping and stemming to tokenise the provided text
    def preprocessing(text):
        from string import punctuation
        # Eliminate duplicate whitespaces using wildcards
        text = re.sub('\s+', ' ', text)
        # checks for all the alphanumeric characters, spaces and punctuations and keeps them in the corpus
        text = ''.join(t for t in text if (t.isalnum() or t.isspace() or t in punctuation))
        # replaces all the punctiations with spaces
        regex = re.compile('[%s]' % re.escape(punctuation))
        text = regex.sub(' ', text)
        # splits the entire text at blank spaces
        text = text.split()

        # converts the entire vocabulary into lower case
        words = []
        for word in text:
            words.append(word.lower())
        
        # Stop words removal
        stopwords = open("/Users/arnav/Desktop/Y4/ttds/cw1/englishST.txt").read()
        processed_text = [word for word in words if word not in stopwords]

        # Normalization using Porter stemmer
        porter = PorterStemmer()
        stem_words = [porter.stem(word) for word in processed_text]
        
        return stem_words

    # converts the received tsv file into 2 dictionaries: 1. As a dictionary of each class and the respective documents as lists of vocabs
    # 2. an embedded dictionary of class, word and the number of words present in that class. It also disregards 10 words with the least counts
    def tsv_reader(filename):

        classes = collections.defaultdict(lambda: collections.defaultdict(int))
        docs = collections.defaultdict(list)
        classes_final = {}
        with open(filename) as f:
            a = 0
            for row in f:
                class_name, text = row.strip().split("\t")
                a+=1
                vocab = ANALYSIS.preprocessing(text)
                docs[class_name].append(vocab)
                
                for token in set(vocab):
                    classes[class_name][token] += 1
        
        classes_removed_10 = {}
        for i in ["OT","NT","Quran"]:
            classes_removed_10[i] = {token:count for token, count in classes[i].items() if count >= 10}

        return docs, classes_removed_10

    # Section for Token Analysis
    # Helper function to calculate the Mutual Information of each word in each class
    def MI_calc(N_11, N_10, N_00, N_01, N_1, N_0, N1_, N0_, N):
        t1 = (N_11/N * math.log2((N * N_11)/(N1_ * N_1))) if (N * N_11) != 0 and (N1_ * N_1) != 0 else 0
        t2 = (N_10/N * math.log2((N * N_10)/(N1_ * N_0))) if (N * N_10) != 0 and (N1_ * N_0) != 0 else 0
        t3 = (N_01/N * math.log2((N * N_01)/(N0_ * N_1))) if (N * N_01) != 0 and (N0_ * N_1) != 0 else 0
        t4 = (N_00/N * math.log2((N * N_00)/(N0_ * N_0))) if (N * N_00) != 0 and (N0_ * N_0) != 0 else 0

        return t1+t2+t3+t4

    # Helper function to calculate the Chi Squared of each word in each class
    def chi_sq_calc(N_11, N_10, N_00, N_01, N_1, N_0, N1_, N0_, N):
        return (((N * ((N_11 * N_00) - (N_10 * N_01))**2) / (N1_ * N_1 * N0_ * N0_)) if (N1_ * N_1 * N0_ * N0_) != 0 else 0)

    # Main method for token analysis. Calculates MI and Chi Squared values for each word in a particular class and also prints the required
    # top 10 words for each class both according to MI and chi sq values.
    def mi_chi(docs, classes):

        classes_MI = {}
        classes_chi = {}
        for claus, words in classes.items():

            total_in_class = len(docs[claus])

            MI = collections.defaultdict(dict)
            chi_sq = collections.defaultdict(dict)
            
            for word, count in words.items():
                
                N = sum(len(doc) for doc in docs.values())
                N_11 = count

                x = []
                for claus1, words1 in classes.items():
                    if claus1 != claus:
                        if word in words1.keys():
                            x.append(words1[word])
                
                N_10 = sum(x)
                N_01 = total_in_class - N_11

                a = 0
                for claus2, words2, in docs.items():
                    if claus2 != claus:
                        a += len(words2)

                N_00 = a - N_10

                N1_ = N_11 + N_10
                N_1 = N_01 + N_11
                N0_ = N_01 + N_00
                N_0 = N_00 + N_10

                MI[claus][word] = ANALYSIS.MI_calc(N_11, N_10, N_00, N_01, N_1, N_0, N1_, N0_, N)
                chi_sq[claus][word] = ANALYSIS.chi_sq_calc(N_11, N_10, N_00, N_01, N_1, N_0, N1_, N0_, N)
            
            classes_MI.update(dict(MI))
            classes_chi.update(dict(chi_sq))

        for i in ["OT","NT","Quran"]:
            classes_MI_class = dict(sorted(classes_MI[i].items(), key=lambda item: item[1], reverse=True))
            classes_chi_class = dict(sorted(classes_chi[i].items(), key=lambda item: item[1], reverse=True))
            # print(list(islice(classes_MI_class.items(),10)))
            # print(list(islice(classes_chi_class.items(),10)))

        return classes_MI, classes_chi

    # Section for topic Analysis
    # Helper function to give the average score of each topic of the class
    def avg_score_topic(topic_probs):

        avg_score = []
        for topic in range(20):
            topicwise_sum = 0
            for doc in topic_probs:
                topicwise_sum += doc[topic][1]
            avg_score.append(topicwise_sum/len(topic_probs))
        
        return avg_score

    # Main block for Topic Analysis. Here, the gensim LDA model is used with 20 random topics to calculate the topic probability for each document
    # in the class and get the average topic score of the class
    def lda(docs):
        
        common_texts = [t for doc in docs.values() for t in doc]
        common_dictionary = Dictionary(common_texts)
        common_corpus = [common_dictionary.doc2bow(text) for text in common_texts]
        lda = LdaModel(common_corpus, num_topics = 20, id2word = common_dictionary)
        corpus_each = list(docs.values())

        OT_topic_prob = [lda.get_document_topics(bow = text , minimum_probability = 0) for text in common_corpus[:len(corpus_each[0])]]
        NT_topic_prob = [lda.get_document_topics(bow = text , minimum_probability = 0) for text in common_corpus[len(corpus_each[0]):(len(corpus_each[0])+len(corpus_each[1]))]]
        Q_topic_prob = [lda.get_document_topics(bow = text , minimum_probability = 0) for text in common_corpus[(len(corpus_each[0])+len(corpus_each[1])):]]

        OT_avg_topic_score = ANALYSIS.avg_score_topic(OT_topic_prob)
        NT_avg_topic_score = ANALYSIS.avg_score_topic(NT_topic_prob)
        Q_avg_topic_score = ANALYSIS.avg_score_topic(Q_topic_prob)

        maxm = [("OT", OT_avg_topic_score.index(max(OT_avg_topic_score)),max(OT_avg_topic_score)), 
        ("NT", NT_avg_topic_score.index(max(NT_avg_topic_score)),max(NT_avg_topic_score)), ("Quran", Q_avg_topic_score.index(max(Q_avg_topic_score)),max(Q_avg_topic_score))]

        for t in maxm:
            print(f'Topic: {t[2]} for {t[0]} for topic {t[1]}')
            print(lda.print_topic(t[1], topn = 10))

    # main implementation block for task 2
    def task_2():
        docs, classes = ANALYSIS.tsv_reader("/Users/arnav/Desktop/Y4/ttds/cw2/ot_nt_q.tsv")
        MI, chi = ANALYSIS.mi_chi(docs, classes)
        lda_out = ANALYSIS.lda(docs)

# TASK 3

# This section implements a sentiment analyser with an LinearSVC classifier and tries the improvements that can be made on 
# the baseline model to increase accuracy
class CLASSIFICATION:

    # Preprocesses for baseline model by only removing links and tokenising 
    def preprocess(text):

        #removing all the links
        text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
        # tokenisation
        text = text.split()

        return(text)

    # preprocesses for the improvement model and performs all the preprocessing done previously for lda model but does not apply stemming
    # removes all the links
    def preprocess_imp(text):
        from string import punctuation
        #removing all the links
        text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)

        # from string import punctuation
        # Eliminate duplicate whitespaces using wildcards
        text = re.sub('\s+', ' ', text)
        # checks for all the alphanumeric characters, spaces and punctuations and keeps them in the corpus
        text = ''.join(t for t in text if (t.isalnum() or t.isspace() or t in punctuation))
        # replaces all the punctiations with spaces
        punctuation1 = punctuation.replace("\'",'')
        regex = re.compile('[%s]' % re.escape(punctuation1))
        text = regex.sub(' ', text)
        # design choice to replace all punctuations with space but apostophe with no-space('')
        text = re.sub("\'",'',text)
        # splits the entire text at blank spaces
        text = text.split()

        # converts the entire vocabulary into lower case
        words = []
        for word in text:
            words.append(word.lower())

        # Stop words removal
        stopwords = open("/Users/arnav/Desktop/Y4/ttds/cw1/englishST.txt").read()
        processed_text = [word for word in words if word not in stopwords]

        return(processed_text)

    # creates a dataset for the baseline model by creating a dataframe with an added column of preprocessed tweet
    # also shuffles the data everytime
    def baseline(train_dev):
        train_dev_shuffled = train_dev.sample(frac=1)

        preprocessed_tweet = []
        for idx, i in train_dev_shuffled.iterrows():
            preprocessed_tweet.append(CLASSIFICATION.preprocess(i["tweet"]))
        train_dev_shuffled["preprocessed_tweet"] = preprocessed_tweet
        return train_dev_shuffled

    # improved dataset for improvement model with the improved preprocessing
    def baseline_imp(train_dev):
        train_dev_shuffled = train_dev.sample(frac=1)

        preprocessed_tweet = []
        for idx, i in train_dev_shuffled.iterrows():
            preprocessed_tweet.append(CLASSIFICATION.preprocess_imp(i["tweet"]))
        train_dev_shuffled["preprocessed_tweet"] = preprocessed_tweet
        return train_dev_shuffled

    # creates a word2id dictionary for all the unique tokens in the train/dev/test set
    def vocabid(train):
        unique_tokens = []
        for tweet in train:
            unique_tokens += tweet
        unique_tokens = set(unique_tokens)

        vocab_id = {}
        for idx, tokens in enumerate(unique_tokens):
            vocab_id[tokens] = idx

        return vocab_id

    # creates the BOW matrix feature for training the model and predicting too
    def bow_matrix(data, id):
        oov_index = len(id)
        S = dok_matrix((len(data), len(id)+1))
        for doc_id, doc in enumerate(data):
            for word in doc:
                S[doc_id, id.get(word, oov_index)] += 1
        
        return S

    # creates a normalized BOW matrix for improvement model
    def bow_matrix_normalized(data, id):
        oov_index = len(id)
        S = dok_matrix((len(data), len(id)+1))
        for doc_id, doc in enumerate(data):
            for word in doc:
                S[doc_id, id.get(word, oov_index)] += 1/len(doc)
        
        return S

    # creates the list of unique categories in the dataset and provides them ids
    def categoryid(data):
        category_id = {}
        for idx, category in enumerate(set(data)):
            category_id[category] = idx
        E = [category_id[category] for category in data]

        cat_names = []
        for cat,cid in sorted(category_id.items(),key=lambda x:x[1]):
            cat_names.append(cat)

        return E, cat_names

    # prepares the precision, recall and f1 data properly for writing into the file
    def prepared_data(pred, true, cat_names):

        all_dict = classification_report(true, pred, output_dict = True, target_names=cat_names)
        del all_dict['accuracy']
        del all_dict['weighted avg']
        scores = []
        for i in all_dict.keys():
            scores.append(all_dict[i]['precision'])
            scores.append(all_dict[i]['recall'])
            scores.append(all_dict[i]['f1-score'])
        return scores

    # main implementation block for task 3
    def task_3():
        # get the dataset to train the model
        train_dev = pd.read_csv("/Users/arnav/Desktop/Y4/ttds/cw2/train.tsv", sep = "\t")

        # divide the dataset into a 80-20 split for training and development
        train, dev = train_test_split(CLASSIFICATION.baseline(train_dev), test_size=0.2)
        test = pd.read_csv("/Users/arnav/Desktop/Y4/ttds/cw2/test.tsv", sep = "\t")
        test_baseline = CLASSIFICATION.baseline(test)

        # Extract the required data from the dataframe
        Xtrain = train["preprocessed_tweet"].tolist()
        Xdev = dev["preprocessed_tweet"].tolist()
        Ytrain = train["sentiment"].tolist()
        Ydev= dev["sentiment"].tolist()
        Xtest = test_baseline["preprocessed_tweet"].tolist()
        Ytest = test_baseline["sentiment"].tolist()

        # get the word2id
        vocab_id = CLASSIFICATION.vocabid(Xtrain)

        # For training and predicting on the baseline model
        sparse_matrix_train = CLASSIFICATION.bow_matrix(Xtrain, vocab_id)
        sparse_matrix_dev = CLASSIFICATION.bow_matrix(Xdev, vocab_id)
        sparse_matrix_test = CLASSIFICATION.bow_matrix(Xtest, vocab_id)
        category_id_train, train_cat_names = CLASSIFICATION.categoryid(Ytrain)
        category_id_dev, dev_cat_names = CLASSIFICATION.categoryid(Ydev)
        category_id_test, test_cat_names = CLASSIFICATION.categoryid(Ytest)
        model = sklearn.svm.LinearSVC(C=1000, random_state = 42)
        model.fit(sparse_matrix_train, category_id_train)
        y_train_preds = model.predict(sparse_matrix_train)
        y_dev_preds = model.predict(sparse_matrix_dev)
        y_test_preds = model.predict(sparse_matrix_test)

        # gets a list of true label, predicted label, tweet, preprocessed tweet from the development to analyse and improve the model
        difference = []
        counter = 0
        for i, j in zip(category_id_dev, y_dev_preds):
            if i != j:
                difference.append((i,j,dev["tweet"].tolist()[counter],dev["preprocessed_tweet"].tolist()[counter]))
            counter += 1
        # print(difference)

        train_dict = CLASSIFICATION.prepared_data(category_id_train, y_train_preds, train_cat_names)
        dev_dict = CLASSIFICATION.prepared_data(category_id_dev, y_dev_preds, dev_cat_names)
        test_dict = CLASSIFICATION.prepared_data(category_id_test, y_test_preds, test_cat_names)

        # For improving the accuracy of the baseline model
        #IMPROVEMENT model (Normalized bow matric and c=50 for 6-8% improvement -- best model yet)
        train_imp, dev_imp = train_test_split(CLASSIFICATION.baseline_imp(train_dev), test_size=0.1)
        test_baseline_imp = CLASSIFICATION.baseline_imp(test)
        Xtrain_imp = train_imp["preprocessed_tweet"].tolist()
        Xdev_imp = dev_imp["preprocessed_tweet"].tolist()
        Ytrain_imp = train_imp["sentiment"].tolist()
        Ydev_imp = dev_imp["sentiment"].tolist()
        Xtest_imp = test_baseline_imp["preprocessed_tweet"].tolist()
        Ytest_imp = test_baseline_imp["sentiment"].tolist()

        vocab_id_imp = CLASSIFICATION.vocabid(Xtrain_imp)

        sparse_matrix_train_imp = CLASSIFICATION.bow_matrix_normalized(Xtrain_imp, vocab_id_imp)
        sparse_matrix_dev_imp = CLASSIFICATION.bow_matrix_normalized(Xdev_imp, vocab_id_imp)
        sparse_matrix_test_imp = CLASSIFICATION.bow_matrix_normalized(Xtest_imp, vocab_id_imp)
        category_id_train_imp, train_cat_names_imp = CLASSIFICATION.categoryid(Ytrain_imp)
        category_id_dev_imp, dev_cat_names_imp = CLASSIFICATION.categoryid(Ydev_imp)
        category_id_test_imp, test_cat_names_imp = CLASSIFICATION.categoryid(Ytest_imp)

        model_imp = sklearn.svm.SVC(C=500, random_state = 42)
        model_imp.fit(sparse_matrix_train_imp, category_id_train_imp)
        y_train_preds_imp = model_imp.predict(sparse_matrix_train_imp)
        y_dev_preds_imp = model_imp.predict(sparse_matrix_dev_imp)
        y_test_preds_imp = model_imp.predict(sparse_matrix_test_imp)

        train_dict_imp = CLASSIFICATION.prepared_data(category_id_train_imp, y_train_preds_imp, train_cat_names_imp)
        dev_dict_imp = CLASSIFICATION.prepared_data(category_id_dev_imp, y_dev_preds_imp, dev_cat_names_imp)
        test_dict_imp = CLASSIFICATION.prepared_data(category_id_test_imp, y_test_preds_imp, test_cat_names_imp)

        # Writing into the classification.csv file in the required format
        with open('classification.csv', 'w') as cls:
            writer = csv.writer(cls, delimiter=",")
            writer.writerow(['system','split','p-pos','r-pos','f-pos','p-neg','r-neg','f-neg','p-neu','r-neu','f-neu','p-macro','r-macro','f-macro'])
            writer.writerow(['baseline', 'train', train_dict[0], train_dict[1], train_dict[2], train_dict[3], train_dict[4], train_dict[5], train_dict[6], train_dict[7], train_dict[8], train_dict[9], train_dict[10], train_dict[11]])
            writer.writerow(['baseline', 'dev', dev_dict[0], dev_dict[1], dev_dict[2], dev_dict[3], dev_dict[4], dev_dict[5], dev_dict[6], dev_dict[7], dev_dict[8], dev_dict[9], dev_dict[10], dev_dict[11]])
            writer.writerow(['baseline', 'test', test_dict[0], test_dict[1], test_dict[2], test_dict[3], test_dict[4], test_dict[5], test_dict[6], test_dict[7], test_dict[8], test_dict[9], test_dict[10], test_dict[1]])
            writer.writerow(['improved', 'train', train_dict_imp[0], train_dict_imp[1], train_dict_imp[2], train_dict_imp[3], train_dict_imp[4], train_dict_imp[5], train_dict_imp[6], train_dict_imp[7], train_dict_imp[8], train_dict_imp[9], train_dict_imp[10], train_dict_imp[11]])
            writer.writerow(['improved', 'dev', dev_dict_imp[0], dev_dict_imp[1], dev_dict_imp[2], dev_dict_imp[3], dev_dict_imp[4], dev_dict_imp[5], dev_dict_imp[6], dev_dict_imp[7], dev_dict_imp[8], dev_dict_imp[9], dev_dict_imp[10], dev_dict_imp[11]])
            writer.writerow(['improved', 'test', test_dict_imp[0], test_dict_imp[1], test_dict_imp[2], test_dict_imp[3], test_dict_imp[4], test_dict_imp[5], test_dict_imp[6], test_dict_imp[7], test_dict_imp[8], test_dict_imp[9], test_dict_imp[10], test_dict_imp[1]])


if __name__ == "__main__":

    EVAL.task_1()
    ANALYSIS.task_2()
    CLASSIFICATION.task_3()
