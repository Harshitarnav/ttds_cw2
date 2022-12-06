import itertools

def retrieved_docs(filename):
    
    with open(filename, 'r') as f:

        lines = f.readlines()
        retrieved_doc = {}
        sys_retrieved_doc = {}

        print(lines[0])
        for line in lines[1:]:
            system_number,query_number,doc_number,rank_of_doc,score = line.split(',')
            score = score[:-1]
            # print(system_number,query_number,doc_number,rank_of_doc,score)
            # break
            # doc_details = (rank_of_doc, score)
            if query_number in retrieved_doc:
                retrieved_doc[query_number].update({doc_number:([rank_of_doc, score])})
            else:
                retrieved_doc.setdefault(query_number, {doc_number:([rank_of_doc, score])})

            if system_number in sys_retrieved_doc:
                sys_retrieved_doc[system_number].update(retrieved_doc)
            else:
                sys_retrieved_doc.setdefault(system_number, retrieved_doc)

    # print(sys_retrieved_doc)
    return sys_retrieved_doc

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
    for i in range(len(retrieved_docs)-1):
        print(retrieved_docs)
        print(relevant_docs)
        if list(retrieved_docs.keys())[i] in relevant_docs:
            print(retrieved_docs.keys())
        break

            


sys_retrieved_doc = retrieved_docs("/Users/arnav/Desktop/Y4/ttds/cw2/system_results.csv")
relevant_doc = relevant_docs("/Users/arnav/Desktop/Y4/ttds/cw2/qrels.csv")

for _ ,retrieved_doc in sys_retrieved_doc.items():

    for query, _ in retrieved_doc.items():

        retrieved_first_10 = retrieved_first_n(retrieved_doc[query], 10)
        retrieved_first_20 = retrieved_first_n(retrieved_doc[query], 20)
        retrieved_first_50 = retrieved_first_n(retrieved_doc[query], 50)

        relevant_doc_q = [i for i in relevant_doc[query].keys()]

        precision_10 = precision(retrieved_first_10.keys(), relevant_doc_q)

        recall_50 = recall(retrieved_first_50.keys(), relevant_doc_q)

        cut_off = retrieved_first_n(retrieved_doc[query], len(relevant_doc_q))
        r_precision = precision(cut_off.keys(), relevant_doc_q)

        ap = AP(retrieved_doc, relevant_doc_q)

        break
    break