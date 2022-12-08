import collections
import csv
import numpy as np
import pandas as pd
import math
import re
from scipy.sparse import dok_matrix
import sklearn
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from nltk.stem import PorterStemmer
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel

def retrieved_docs(filename):

    results = collections.defaultdict(lambda: collections.defaultdict(list))
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader)
        for row in reader:
            results[int(row[0])][int(row[1])].append(row[2:])
        
    return results

def Extract(lst):
    return [int(item[0]) for item in lst]

def relevant_docs(filename):

    results = collections.defaultdict(lambda: collections.defaultdict(list))
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader)
        for row in reader:
            results[int(row[0])][int(row[1])] = int(row[2])
        
    return results

def precision(retrieved_docs, relevant_docs):
    relevant_retrieved = relevant_docs.keys() & retrieved_docs
    precision = len(relevant_retrieved)/len(retrieved_docs)
    return round(precision, 3)

def recall(retrieved_docs, relevant_docs):
    relevant_retrieved = relevant_docs.keys() & retrieved_docs
    recall = len(relevant_retrieved)/len(relevant_docs)
    return round(recall, 3)

def AP(retrieved_docs, relevant_docs):
    ap = 0 

    for k in retrieved_docs:
        if int(k[0]) in relevant_docs.keys():
            precision_k = precision(Extract(retrieved_docs[:int(k[1])]), relevant_docs)
            ap += precision_k * 1
        else:
            ap += 0

    return round(ap/len(relevant_docs), 3)

def nDCG(retrieved_docs, relevant_docs):

    rel1 = relevant_docs[int(retrieved_docs[0][0])] if retrieved_docs[0][0] in relevant_docs.keys() else 0
    DCG = rel1
    for i in retrieved_docs[1:]:
        if int(i[0]) in relevant_docs.keys():
            DCG += relevant_docs[int(i[0])]/np.log2(int(i[1]))
    
    irel = list(relevant_docs.values())

    iDCG = irel[0]
    for idx, rel in enumerate(irel[1:]):
        iDCG += rel/np.log2(idx + 2)
        
    nDCG = DCG/iDCG

    return round(nDCG, 3)

with open('ir_eval.csv', 'w') as result:
    sys_retrieved_doc = retrieved_docs("/Users/arnav/Desktop/Y4/ttds/cw2/system_results.csv")
    relevant_doc = relevant_docs("/Users/arnav/Desktop/Y4/ttds/cw2/qrels.csv")
    writer = csv.writer(result, delimiter=",")
    writer.writerow(['system_number','query_number','P@10','R@50','r-precision','AP','nDCG@10','nDCG@20'])

    means = []
    for system,retrieved_doc in sys_retrieved_doc.items():
        for_mean = []
        for query,doc in retrieved_doc.items():

            precision_10 = precision(Extract(doc[:10]), relevant_doc[query])

            recall_50 = recall(Extract(doc[:50]), relevant_doc[query])

            cut_off = doc[:len(relevant_doc[query])]
            r_precision = precision(Extract(cut_off), relevant_doc[query])

            ap = AP(doc, relevant_doc[query])
            
            nDCG_10 = nDCG(doc[:10], relevant_doc[query])
            nDCG_20 = nDCG(doc[:20], relevant_doc[query])

            writer.writerow([system,query,precision_10,recall_50,r_precision,ap,nDCG_10,nDCG_20])

            for_mean.append([precision_10,recall_50,r_precision,ap,nDCG_10,nDCG_20])
        
        means.append(np.mean(for_mean, axis=0))
        writer.writerow([system,"mean",round(means[system-1][0],3),round(means[system-1][1],3),round(means[system-1][2],3),
        round(means[system-1][3],3),round(means[system-1][4],3),round(means[system-1][5],3)])


# preprosses the input text
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
    # print(processed_text[:100])

    # Normalization using Porter stemmer
    porter = PorterStemmer()
    stem_words = [porter.stem(word) for word in processed_text]
    stem_words_without_stopping = [porter.stem(word) for word in words]
    # print(stem_words[:80])

    # words after stopping and stemming
    # return stem_words
    # comment this return statement out to return without stopping
    # print(stem_words)
    return stem_words


def tsv_reader(filename):

    classes = collections.defaultdict(lambda: collections.defaultdict(int))
    docs = collections.defaultdict(list)
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        vocab = []
        for row in reader:
            vocab = preprocessing(row[1])
            docs[row[0]].append(vocab)
            
            for token in set(vocab):
                classes[row[0]][token] += 1 
            
    return docs, classes

def MI_calc(N_11, N_10, N_00, N_01, N_1, N_0, N1_, N0_, N):
    t1 = (N_11/N * math.log2((N * N_11)/(N1_ * N_1))) if (N * N_11) != 0 and (N1_ * N_1) != 0 else 0
    t2 = (N_10/N * math.log2((N * N_10)/(N1_ * N_0))) if (N * N_10) != 0 and (N1_ * N_0) != 0 else 0
    t3 = (N_01/N * math.log2((N * N_01)/(N0_ * N_1))) if (N * N_01) != 0 and (N0_ * N_1) != 0 else 0
    t4 = (N_00/N * math.log2((N * N_00)/(N0_ * N_0))) if (N * N_00) != 0 and (N0_ * N_0) != 0 else 0

    return t1+t2+t3+t4

def chi_sq_calc(N_11, N_10, N_00, N_01, N_1, N_0, N1_, N0_, N):
    return (((N * ((N_11 * N_00) - (N_10 * N_01))**2) / (N1_ * N_1 * N0_ * N0_)) if (N1_ * N_1 * N0_ * N0_) != 0 else 0)

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

            MI[claus][word] = MI_calc(N_11, N_10, N_00, N_01, N_1, N_0, N1_, N0_, N)
            chi_sq[claus][word] = chi_sq_calc(N_11, N_10, N_00, N_01, N_1, N_0, N1_, N0_, N)
        
        classes_MI.update(dict(MI))
        classes_chi.update(dict(chi_sq))

    return (classes_MI, classes_chi)

def avg_score_topic(topic_probs):

    avg_score = []
    for topic in range(20):
        topicwise_sum = 0
        for doc in topic_probs:
            topicwise_sum += doc[topic][1]
        avg_score.append(topicwise_sum/len(topic_probs))
    return avg_score

def lda(docs):
    
    common_texts = [t for doc in docs.values() for t in doc]
    common_dictionary = Dictionary(common_texts)
    common_corpus = [common_dictionary.doc2bow(text) for text in common_texts]
    lda = LdaModel(common_corpus, num_topics = 20, id2word = common_dictionary, random_state=42)
    corpus_each = list(docs.values())
    
    OT_topic_prob = [lda.get_document_topics(bow = text , minimum_probability = 0) for text in common_corpus[:len(corpus_each[0])]]
    NT_topic_prob = [lda.get_document_topics(bow = text , minimum_probability = 0) for text in common_corpus[len(corpus_each[0]):(len(corpus_each[0])+len(corpus_each[1]))]]
    Q_topic_prob = [lda.get_document_topics(bow = text , minimum_probability = 0) for text in common_corpus[(len(corpus_each[0])+len(corpus_each[1])):]]

    OT_avg_topic_score = avg_score_topic(OT_topic_prob)
    NT_avg_topic_score = avg_score_topic(NT_topic_prob)
    Q_avg_topic_score = avg_score_topic(Q_topic_prob)

    maxm = [("OT", OT_avg_topic_score.index(max(OT_avg_topic_score)),max(OT_avg_topic_score)), 
    ("NT", NT_avg_topic_score.index(max(NT_avg_topic_score)),max(NT_avg_topic_score)), ("Quran", Q_avg_topic_score.index(max(Q_avg_topic_score)),max(Q_avg_topic_score))]

    for t in maxm:
        print(f'Topic: {t[2]} for {t[0]}')
        print(lda.print_topic(t[1], 10))

docs, classes = tsv_reader("/Users/arnav/Desktop/Y4/ttds/cw2/train_and_dev.tsv")

MI, chi = mi_chi(docs, classes)

lda_out = lda(docs)


def preprocess(text):

    #removing all the links
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)

    # from string import punctuation
    # # Eliminate duplicate whitespaces using wildcards
    # text = re.sub('\s+', ' ', text)
    # # checks for all the alphanumeric characters, spaces and punctuations and keeps them in the corpus
    # text = ''.join(t for t in text if (t.isalnum() or t.isspace() or t in punctuation))
    # # replaces all the punctiations with spaces
    # punctuation1 = punctuation.replace("\'",'')
    # regex = re.compile('[%s]' % re.escape(punctuation1))
    # text = regex.sub(' ', text)
    # # design choice to replace all punctuations with space but apostophe with no-space('')
    # text = re.sub("\'",'',text)
    # # splits the entire text at blank spaces
    text = text.split()
    
    # # converts the entire vocabulary into lower case
    # words = []
    # for word in text:
    #     words.append(word.lower())

    
    return(text)

def baseline(train_dev):
    train_dev_shuffled = train_dev.sample(frac=1)

    preprocessed_tweet = []
    for idx, i in train_dev_shuffled.iterrows():
        # preprocess(i["tweet"])
        preprocessed_tweet.append(preprocess(i["tweet"]))
    train_dev_shuffled["preprocessed_tweet"] = preprocessed_tweet
    # print(train_dev_shuffled)
    return train_dev_shuffled

def vocabid(train):
    unique_tokens = []
    for tweet in train:
        unique_tokens += tweet
    unique_tokens = set(unique_tokens)
    # print(len(unique_tokens))

    vocab_id = {}
    for idx, tokens in enumerate(unique_tokens):
        vocab_id[tokens] = idx
    # print(vocab_id)
    return vocab_id
 
def bow_matrix(data, id):
    oov_index = len(id)
    S = dok_matrix((len(data), len(id)+1))
    for doc_id, doc in enumerate(data):
        for word in doc:
            S[doc_id, id.get(word, oov_index)] += 1
    
    return S

def categoryid(data):
    category_id = {}
    for idx, category in enumerate(set(data)):
        category_id[category] = idx

    E = [category_id[category] for category in data]
    # print(category_id)
    # print(E)
    return E

train_dev = pd.read_csv("/Users/arnav/Desktop/Y4/ttds/cw2/train.tsv", sep = "\t")

# 80 20 split
train, dev = train_test_split(baseline(train_dev), test_size=0.2)

Xtrain = train["preprocessed_tweet"].tolist()
Xdev = dev["preprocessed_tweet"].tolist()
Ytrain = train["sentiment"].tolist()
Ydev= dev["sentiment"].tolist()

vocab_id = vocabid(Xtrain)

sparse_matrix_train = bow_matrix(Xtrain, vocab_id)
category_id_train = categoryid(Ytrain)
model = sklearn.svm.SVC(C=1000, random_state=42)
model.fit(sparse_matrix_train, category_id_train)

sparse_matrix_dev = bow_matrix(Xdev, vocab_id)
y_train_preds = model.predict(sparse_matrix_train)
y_dev_preds = model.predict(sparse_matrix_dev)

category_id_dev = categoryid(Ydev)

test = pd.read_csv("/Users/arnav/Desktop/Y4/ttds/cw2/test.tsv", sep = "\t")
test_baseline = baseline(test)
print(test_baseline)
Xtest = test_baseline["preprocessed_tweet"].tolist()
Ytest = test_baseline["sentiment"].tolist()
sparse_matrix_test = bow_matrix(Xtest, vocab_id)
category_id_test = categoryid(Ytest)
y_test_preds = model.predict(sparse_matrix_test)

print(classification_report(category_id_dev, y_dev_preds))
print(classification_report(category_id_train, y_train_preds))
print(classification_report(category_id_test, y_test_preds))

with open('ainvayi.txt','w') as f:
    writer = csv.writer(f, delimiter=",")
    writer.writerow(y_dev_preds)
    writer.writerow(category_id_dev)