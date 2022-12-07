import collections
import csv
import math

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

def retrieved_first_n(retrieved, n):
    return {i: retrieved[i] for i in list(retrieved.keys())[:n]}

def precision(retrieved_docs, relevant_docs):
    # retrieved_docs.intersection(list(relevant_docs.keys()))
    relevant_retrieved = relevant_docs.keys() & retrieved_docs
    precision = len(relevant_retrieved)/len(retrieved_docs)
    return precision

def recall(retrieved_docs, relevant_docs):
    relevant_retrieved = relevant_docs.keys() & retrieved_docs
    recall = len(relevant_retrieved)/len(relevant_docs)
    return recall

def AP(retrieved_docs, relevant_docs):
    ap = 0 

    for k in retrieved_docs:
        if int(k[0]) in relevant_docs.keys():
            precision_k = precision(Extract(retrieved_docs[:int(k[1])]), relevant_docs)
            ap += precision_k * 1
        
        else:
            ap += 0

    return ap/len(relevant_docs)

def nDCG(retrieved_docs, relevant_docs):

    rel1 = relevant_docs[int(retrieved_docs[0][0])] if retrieved_docs[0][0] in relevant_docs.keys() else 0
    DCG = rel1
    for i in retrieved_docs[1:]:
        if int(i[0]) in relevant_docs.keys():
            DCG += relevant_docs[int(i[0])]/math.log2(int(i[1]))
    
    # relevant_sort = {k: v for k, v in sorted(relevant_docs.items(), key=lambda item: item[1], reverse = True)}

    irel = list(relevant_docs.values())

    iDCG = irel[0]
    for idx, rel in enumerate(irel[1:]):
        iDCG += rel/math.log2(idx + 2)
        
    nDCG = DCG/iDCG

    return nDCG
        
sys_retrieved_doc = retrieved_docs("/Users/arnav/Desktop/Y4/ttds/cw2/system_results.csv")
relevant_doc = relevant_docs("/Users/arnav/Desktop/Y4/ttds/cw2/qrels.csv")

for system,retrieved_doc in sys_retrieved_doc.items():
    print("//////////////////////////////////////////////////////////")
    for query,doc in retrieved_doc.items():

        precision_10 = precision(Extract(doc[:10]), relevant_doc[query])

        recall_50 = recall(Extract(doc[:50]), relevant_doc[query])

        cut_off = doc[:len(relevant_doc[query])]
        r_precision = precision(Extract(cut_off), relevant_doc[query])

        ap = AP(doc, relevant_doc[query])
        
        nDCG_10 = nDCG(doc[:10], relevant_doc[query])
        nDCG_20 = nDCG(doc[:20], relevant_doc[query])

        print(nDCG_20)