import collections
import csv
import numpy as np

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
        
sys_retrieved_doc = retrieved_docs("/Users/arnav/Desktop/Y4/ttds/cw2/system_results.csv")
relevant_doc = relevant_docs("/Users/arnav/Desktop/Y4/ttds/cw2/qrels.csv")

with open('ir_eval.csv', 'w') as result:
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