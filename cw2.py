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
            break

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

def precision(retrieved_docs, relevant_docs):
    relevant_retrieved = list(set(retrieved_docs.intersection(relevant_docs)))
    precision = len(relevant_retrieved)/len(retrieved_docs)
    return precision

def recall(retrieved_docs, relevant_docs):
    relevant_retrieved = list(set(retrieved_docs.intersection(relevant_docs)))
    recall = len(relevant_retrieved)/len(relevant_docs)
    return recall

# def AP(retrieved_docs, relevant_docs):
#     for i in retrieved_docs:
        

# def first_n_retrieved(doc, n):

            


sys_retrieved_doc = retrieved_docs("/Users/arnav/Desktop/Y4/ttds/cw2/system_results.csv")
relevant_doc = relevant_docs("/Users/arnav/Desktop/Y4/ttds/cw2/qrels.csv")

for retrieved_doc in sys_retrieved_doc:

    for query in retrieved_doc:

        retrieved_first_10 = dict(itertools.islice(retrieved_doc[query].items(), 10))
        retrieved_first_20 = dict(itertools.islice(retrieved_doc[query].items(), 20))
        retrieved_first_50 = dict(itertools.islice(retrieved_doc[query].items(), 50))

        relevant_doc_q = [i[0] for i in relevant_doc[query]]

        precision_10 = precision(retrieved_first_10.keys(), relevant_doc_q)

        recall_50 = recall(retrieved_first_50.keys(), relevant_doc_q)