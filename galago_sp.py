import subprocess,json,ijson, time,chardet
from collections import defaultdict
data={"questions":
           [
               {
                   "body": 
                   "can phototherapy be given for children suffering from atopic dermatitis?",
                    "id": "15",
                     "documents": []
                     }
               ]
        }

def call_galago():
    dir1=r"interim_resources/sample_out.json"
    
    command=['Index\\home\\document_retrieval\\galago-3.10-bin\\bin\\galago',
             'batch-search',
             '--index=Index\\home\\document_retrieval\\galago-3.10-bin\\bin\\pubmed_only_abstract_galago_index'
             ,'--verbose=False','--requested=25','--scorer=bm25','--defaultTextPart=postings.krovetz',
             '--mode=threaded', dir1]
    
    
    a=time.time()
    rets=subprocess.Popen(command,stdout=subprocess.PIPE,shell=True)
    
    out,err=rets.communicate()
    print(time.time()-a)
    lines       = out.decode("utf-8").split('\n')
    retrieval_results = defaultdict(list)
    for line in lines:
        if(len(line)>0):
            line_splits = line.split()
            q_id = line_splits[0]
            doc_id = line_splits[2]
            bm25_score = float(line_splits[4])
            retrieval_results[q_id].append((doc_id, bm25_score))
            
    return dict(retrieval_results)
    
print(call_galago())

def load_q_rels(data):
    qrels = defaultdict(list)
    n_qrels = defaultdict(int)
    for i in range(len(data['questions'])):
        q_id = data['questions'][i]['id']
        rel_docs = set([doc.split('/')[-1] for doc in data['questions'][i]['documents']])
        qrels[q_id] = rel_docs
        n_qrels[q_id] = len(rel_docs)
    return dict(qrels), dict(n_qrels)
def load_q_text(data):
    q_text = {}
    for i in range(len(data['questions'])):
        q_id = data['questions'][i]['id']
        text = data['questions'][i]['body']
        q_text[q_id] = text
    return dict(q_text)
print(load_q_rels(data))
print(load_q_text(data))