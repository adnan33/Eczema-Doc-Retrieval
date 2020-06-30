import subprocess,json,ijson, time,chardet,sys,re,nltk
from collections import defaultdict
bioclean_mod = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').replace("-", ' ').strip().lower()).split()
try:
    stopwords   = nltk.corpus.stopwords.words("english")
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')
finally:
    stopwords   = nltk.corpus.stopwords.words("english")
data={"questions":
           [
               {
                   "body": 
                   "can phototherapy be given for children suffering from atopic dermatitis?",
                    "id": "121",
                     "documents": []
                     }
               ]
        }

def preprocess(data, path_out):
    
    ############
    
    queries = data['questions']
    ############
    q_array = []
    for query in queries:
        ############
        tokenized_body = bioclean_mod(query['body'])
        tokenized_body = [t for t in tokenized_body if t not in stopwords]
        ############
        body = ' '.join(tokenized_body)
        q_array.append({"text": body, "number": query["id"]})
    with open(path_out, 'w+') as outfile:
        outfile.write(json.dumps({"queries": [ob for ob in q_array]}, indent=4))

json_dir=r"interim_resources/sample_out_{}.json".format( data['questions'][0]['id'])
def call_galago(json_dir):
    command=['Index\\home\\document_retrieval\\galago-3.10-bin\\bin\\galago',
             'threaded-batch-search',
             '--index=Index\\home\\document_retrieval\\galago-3.10-bin\\bin\\pubmed_only_abstract_galago_index'
             ,'--verbose=False','--requested=25','--scorer=bm25','--defaultTextPart=postings.krovetz',
             '--mode=threaded', json_dir]
    
    
    a=time.time()
    rets=subprocess.Popen(command,stdout=subprocess.PIPE,shell=True)
    
    out,err=rets.communicate()
    print(time.time()-a)
    lines       = out.decode("utf-8").split('\n')
    retrieval_results = defaultdict(list)
    for line in lines:
        if(len(line)>0):
            #print(line.split())
            line_splits = line.split()
            q_id = line_splits[0]
            doc_id = line_splits[2]
            bm25_score = float(line_splits[4])
            retrieval_results[q_id].append((doc_id, bm25_score))
            
    return dict(retrieval_results)
preprocess(data, json_dir)
print(call_galago(json_dir))

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
