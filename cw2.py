import collections
import csv
import numpy as np
import re
from nltk.stem import PorterStemmer

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
    # retrieved_docs.intersection(list(relevant_docs.keys()))
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
    
    # relevant_sort = {k: v for k, v in sorted(relevant_docs.items(), key=lambda item: item[1], reverse = True)}

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
        # print("//////////////////////////////////////////////////////////")
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

    classes = collections.defaultdict(lambda: collections.defaultdict(list))
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        # lines = f.readlines()
        # print(next(reader))
        # print(a)
        text = []
        for row in reader:
            # claus, text = line.split('\t')
            if row[0] in classes:
                text = preprocessing(row[1]) 
                classes[row[0]] += text
            else:
                classes.setdefault(row[0], text)
    # print(classes)
    return classes

def dict_count(classes):

    # print(classes.values())
    total = []
    count_ele_class = {}
    for claus, text in classes.items():
        total += text

        count_ele_class[claus] = dict(collections.Counter(text))
        
    tots = dict(collections.Counter(total))
    total = sum(list(tots.values()))
    
    return count_ele_class, total

def MI_calc(N_11, N_10, N_00, N_01, N_1, N_0, N1_, N0_, N):
    t1 = 0 if N_11 == 0 else (N_11/N * np.log2((N * N_11)/(N1_ * N_1)))
    t2 = 0 if N_10 == 0 else (N_10/N * np.log2((N * N_10)/(N1_ * N_0)))
    t3 = 0 if N_01 == 0 else (N_01/N * np.log2((N * N_01)/(N0_ * N_1)))
    t4 = 0 if N_00 == 0 else (N_00/N * np.log2((N*N_00)/(N0_ * N_0)))

    return t1+t2+t3+t4

def chi_sq_calc(N_11, N_10, N_00, N_01, N):
    return ((N_10 + N_11 + N_01 + N_00) * np.power((N_11 * N_00 - N_10 * N_01),2)) / ((N_01 + N_11) * (N_10 + N_11) * (N_10 + N_00) * (N_01 + N_00))

def mi_chi(class_word_count, total_words):

    x = []
    classes_MI = {}
    classes_chi = {}
    for classes, words in class_word_count.items():

        total_in_class = sum(list(words.values()))

        MI = collections.defaultdict(dict)
        chi_sq = collections.defaultdict(dict)
        print(classes)
        
        for word, count in words.items():

            for claus, words in class_word_count.items():
                if claus != classes:
                    if word in words.keys():
                        x.append(words[word])
            
            N_10 = sum(x)
            N_11 = count
            N_01 = total_in_class - N_11
            N_00 = total_words - N_11 - N_10 - N_01
            N = total_words

            N1_ = N_11 + N_10
            N_1 = N_01 + N_11
            N0_ = N_01 + N_00
            N_0 = N_00 + N_10

            # MI[classes][word] = (N_11/N * np.log2((N * N_11)/(N1_ * N_1))) + (N_01/N * np.log2((N * N_01)/(N0_ * N_1))) + (N_10/N * np.log2((N * N_10)/(N1_ * N_0))) + (N_00/N * np.log2((N*N_00)/(N0_ * N_0)))
            MI[classes][word] = MI_calc(N_11, N_10, N_00, N_01, N_1, N_0, N1_, N0_, N)
            # chi_sq[classes][word] = ((N_10 + N_11 + N_01 + N_00) * (N_11*N_00 - N_10*N_01)^2)/((N_01 + N_11)*(N_10 + N_11)*(N_10 + N_00)*(N_01 + N_00))
            chi_sq[classes][word] = chi_sq_calc(N_11, N_10, N_00, N_01, N)
        
        classes_MI.update(dict(MI))
        classes_chi.update(dict(chi_sq))

    return (classes_MI, classes_chi)


classes = tsv_reader("/Users/arnav/Desktop/Y4/ttds/cw2/test.tsv")

class_word_count, total_words = dict_count(classes)

MI, chi = mi_chi(class_word_count, total_words)
print(chi)