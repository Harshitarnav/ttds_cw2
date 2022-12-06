import collections
import csv

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
            results[int(row[0])][int(row[1])] = row[2:]
        
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
        
sys_retrieved_doc = retrieved_docs("/Users/arnav/Desktop/Y4/ttds/cw2/system_results.csv")
relevant_doc = relevant_docs("/Users/arnav/Desktop/Y4/ttds/cw2/qrels.csv")

for system,retrieved_doc in sys_retrieved_doc.items():
    # print(system, retrieved_doc.keys())
    print("//////////////////////////////////////////////////////////")

    for query,doc in retrieved_doc.items():
        # print(retrieved_doc.keys())

        retrieved_first_10 = doc[:10] 
        retrieved_first_20 = doc[:20] 
        retrieved_first_50 = doc[:50] 

        precision_10 = precision(Extract(retrieved_first_10), relevant_doc[query])

        recall_50 = recall(Extract(retrieved_first_50), relevant_doc[query])
        # print(recall_50)

        cut_off = doc[:len(relevant_doc[query])]
        r_precision = precision(Extract(cut_off), relevant_doc[query])

        ap = AP(doc, relevant_doc[query])
        print(ap)

        
        