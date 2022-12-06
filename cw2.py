import collections
import csv

def retrieved_docs(filename):
    
    # with open(filename, 'r') as f:

    #     lines = f.readlines()
    #     retrieved_doc = {}
    #     sys_retrieved_doc = {}

    #     print(lines[0])
    #     for line in lines[1:]:
    #         system_number,query_number,doc_number,rank_of_doc,score = line.split(',')
    #         score = score[:-1]
    #         # print(system_number,query_number,doc_number,rank_of_doc,score)
    #         # break
    #         # doc_details = (rank_of_doc, score)
    #         # if query_number in retrieved_doc:
    #         #     retrieved_doc[query_number].update({doc_number:([rank_of_doc, score])})
    #         # else:
    #         #     retrieved_doc.setdefault(query_number, {doc_number:([rank_of_doc, score])})

    #         # if system_number in sys_retrieved_doc:
    #         #     sys_retrieved_doc[system_number].update(retrieved_doc)
    #         # else:
    #         #     sys_retrieved_doc.setdefault(system_number, retrieved_doc)
            
    #         if system_number in sys_retrieved_doc:
    #             if query_number in sys_retrieved_doc[system_number]:
    #                 sys_retrieved_doc[system_number][query_number].update({doc_number:([rank_of_doc, score])})
    #             else:
    #                 sys_retrieved_doc[system_number].setdefault(query_number, {doc_number:([rank_of_doc, score])})
    #         else:
    #             sys_retrieved_doc.setdefault(system_number, {query_number, {doc_number:([rank_of_doc, score])}})
    # results = collections.defaultdict(lambda: collections.defaultdict(list))
    # with open(filename) as f:
    #     reader = csv.reader(f, delimiter=',')
    #     next(reader)
    #     for row in reader:
    #         # print(int(row["system_number"]))
    #         # print(int(row[1]))
    #         # print(int(row[2]))
    #         print([row[3:],row[4]])
    #         results[int(row[0])][int(row[1])][int(row[2])][int(row[3])] = float(row[4])
    #         # results[row["system_number"]][row["query_number"]][row["doc_number"]][row["rank_of_doc"]] = row["score"]
    # print(results)

    # def nesteddict():
    #     return collections.defaultdict(nesteddict)

    # results = nesteddict()

    # f = open(filename)
    # data = csv.DictReader(f, delimiter=",")
    # print(data)
    # for row in data:
    #     results[row["system_number"]][row["query_number"]][row["doc_number"]][row["rank_of_doc"]] = row["score"]

    # print(results)
    # return results

    # data = csv.DictReader(f, delimiter=",")
        # for row in data:
        #     if row["R"] != "R":
        #         item = new_data_dict.get(row["UID"], dict())
        #         item[row["BID"]] = int(row["R"])
            
        #         temp_dict = new_data_dict.get(row["FID"], dict())
        #         if row["UID"] in temp_dict:
        #             temp_dict[row["UID"]].update(item)
        #         else:
        #             temp_dict[row["UID"]] = item
            
        #         new_data_dict[row["FID"]] = temp_dict

    # print (results)

    results = collections.defaultdict(lambda: collections.defaultdict(list))
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader)
        for row in reader:
            results[int(row[0])][int(row[1])].append(row[2:])
        
    return results

def Extract(lst):
    return list(next(zip(*lst)))

def relevant_docs(filename):

    with open(filename, 'r') as f:

        lines = f.readlines()
        relevant_doc = {}

        for line in lines[1:]:
            query_id,doc_id,relevance = line.split(',')
            relevance = relevance[:-1]
            
            if query_id in relevant_doc:
                relevant_doc[query_id].update({doc_id:relevance})
            else:
                relevant_doc.setdefault(query_id, {doc_id:relevance})
    
    # print(relevant_doc)
    return relevant_doc

def retrieved_first_n(retrieved, n):
    return {i: retrieved[i] for i in list(retrieved.keys())[:n]}

def precision(retrieved_docs, relevant_docs):
    relevant_retrieved = list(set(retrieved_docs).intersection(relevant_docs))
    precision = len(relevant_retrieved)/len(retrieved_docs)
    return precision

def recall(retrieved_docs, relevant_docs):
    relevant_retrieved = list(set(retrieved_docs).intersection(relevant_docs))
    recall = len(relevant_retrieved)/len(relevant_docs)
    return recall

def AP(retrieved_docs, relevant_docs):
    ap = 0 

    for i in range(len(retrieved_docs)-1):
        if list(retrieved_docs.keys())[i] in relevant_docs:
            retreived_k = {j: retrieved_docs[j] for j in list(retrieved_docs.keys())[:i+1]}
            precision_k = precision(retreived_k.keys(), relevant_docs)
            ap += precision_k * 1
        
        else:
            ap += 0

    return ap/len(relevant_docs)
        
sys_retrieved_doc = retrieved_docs("/Users/arnav/Desktop/Y4/ttds/cw2/system_results.csv")
relevant_doc = relevant_docs("/Users/arnav/Desktop/Y4/ttds/cw2/qrels.csv")

for system,retrieved_doc in sys_retrieved_doc.items():
    print(system, retrieved_doc.keys())
    print("//////////////////////////////////////////////////////////")

    for query,doc in retrieved_doc.items():
        print(doc[:10])

        retrieved_first_10 = doc[:10] # retrieved_first_n(doc, 10)
        retrieved_first_20 = doc[:20] # retrieved_first_n(doc, 20)
        retrieved_first_50 = doc[:50] # retrieved_first_n(doc, 50)
        # print(relevant_doc[query])
        # print(retrieved_first_10)

        relevant_doc_q = [i for i in relevant_doc[query].keys()]

        precision_10 = precision(Extract(retrieved_first_10), relevant_doc_q)

        recall_50 = recall(Extract(retrieved_first_50), relevant_doc_q)

        cut_off = doc[:len(relevant_doc_q)]
        r_precision = precision(Extract(cut_off), relevant_doc_q)

        ap = AP(doc, relevant_doc_q)
        print(precision_10)
        break
        