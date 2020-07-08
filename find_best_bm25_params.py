

import sys, os, re, json, pickle, ijson
from elasticsearch import Elasticsearch
from tqdm import tqdm
from pprint import pprint

# Modified bioclean: also split on dashes. Works better for retrieval with galago.
bioclean_mod = lambda t: re.sub(
    '[.,?;*!%^&_+():-\[\]{}]', '',
    t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').replace("-", ' ').strip().lower()
).split()
bioclean    = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').strip().lower()).split()

doc_index = 'pubmed_abstracts_index_0_1'
map         = "pubmed_abstracts_mapping_0_1"

def idf_val(w, idf, max_idf):
    if w in idf:
        return idf[w]
    return max_idf

def tokenize(x):
  return bioclean(x)

def GetWords(data, doc_text, words):
  for i in range(len(data['queries'])):
    qwds = tokenize(data['queries'][i]['query_text'])
    for w in qwds:
      words[w] = 1
    for j in range(len(data['queries'][i]['retrieved_documents'])):
      doc_id = data['queries'][i]['retrieved_documents'][j]['doc_id']
      dtext = (
              doc_text[doc_id]['title'] + ' <title> ' + doc_text[doc_id]['abstractText']
              # +
              # ' '.join(
              #     [
              #         ' '.join(mm) for mm in
              #         get_the_mesh(doc_text[doc_id])
              #     ]
              # )
      )
      dwds = tokenize(dtext)
      for w in dwds:
        words[w] = 1

def load_idfs(idf_path, words):
    print('Loading IDF tables')
    #
    # with open(dataloc + 'idf.pkl', 'rb') as f:
    with open(idf_path, 'rb') as f:
        idf = pickle.load(f)
    ret = {}
    for w in words:
        if w in idf:
            ret[w] = idf[w]
    max_idf = 0.0
    for w in idf:
        if idf[w] > max_idf:
            max_idf = idf[w]
    idf = None
    print('Loaded idf tables with max idf {}'.format(max_idf))
    #
    return ret, max_idf

def load_all_data(dataloc, idf_pickle_path):
    print('loading pickle data')
    #
    with open(dataloc+'trainining7b.json', 'r') as f:
        bioasq7_data = json.load(f)
        bioasq7_data = dict((q['id'], q) for q in bioasq7_data['questions'])
    #
    with open(dataloc + 'bioasq7_bm25_top100.dev.pkl', 'rb') as f:
        dev_data = pickle.load(f)
    with open(dataloc + 'bioasq7_bm25_docset_top100.dev.pkl', 'rb') as f:
        dev_docs = pickle.load(f)
    with open(dataloc + 'bioasq7_bm25_top100.train.pkl', 'rb') as f:
        train_data = pickle.load(f)
    with open(dataloc + 'bioasq7_bm25_docset_top100.train.pkl', 'rb') as f:
        train_docs = pickle.load(f)
    print('loading words')
    #
    words               = {}
    GetWords(train_data, train_docs, words)
    GetWords(dev_data,   dev_docs,   words)
    #
    print('loading idfs')
    idf, max_idf    = load_idfs(idf_pickle_path, words)
    return dev_data, dev_docs, train_data, train_docs, idf, max_idf, bioasq7_data

# recall: 0.5099327929645772
def get_multi(qtext,n, max_year=2020):
    tokenized_body  = bioclean_mod(qtext)
    question_tokens = [t for t in tokenized_body if t not in stopwords]
    question        = ' '.join(question_tokens)
    bod = {
    "size": n,
   "query": {
        "multi_match": {
            "query": question,
            "type":       "most_fields",
            "fields": ["AbstractText","ArticleTitle"]
        }
    }
}
        
    res         = es.search(index=doc_index, body=bod, request_timeout=120)
    #print(res)
    return res['hits']['hits']

def get_compound(qtext,n, max_year=2020):
    tokenized_body  = bioclean_mod(qtext)
    question_tokens = [t for t in tokenized_body if t not in stopwords]
    question        = ' '.join(question_tokens)
    bod = {
        "size": n,
        "query": {
            "bool" : {
                "must": [
                    {
                        "range": {
                            "DateCreated": {
                                "gte": "1800",
                                "lte": str(max_year),
                                "format": "dd/MM/yyyy||yyyy"
                            }
                        }
                    }
                ],
                "should": [
                    {"match": {"AbstractText": question}},
                    {"match": {"ArticleTitle": question}},

                    {
                        "multi_match": {
                            "query": question,
                            "type": "most_fields",
                            "fields": ["AbstractText", "ArticleTitle"],
                            "operator": "and"
                        }
                    }
                    ,
                    {
                        "range": {
                            "ArticleDate": {
                                "gte": "1800",
                                "lte": str(max_year),
                                "format": "dd/MM/yyyy||yyyy"
                            }
                        }
                    }
                ],
                "minimum_should_match": 1,
            }
        }
    }
    
        
    res = es.search(index=doc_index, body=bod, request_timeout=120)
    #print(res)
    return res['hits']['hits']
# recall: 0.5138480484607151   
def get_compound_2(qtext,n, max_year=2020):
    tokenized_body  = bioclean_mod(qtext)
    question_tokens = [t for t in tokenized_body if t not in stopwords]
    question        = ' '.join(question_tokens)
    bod = {
        "size": n,
        "query": {
            "bool" : {
                
                "should": [
                    {"match": {"AbstractText": question}},
                    {"match": {"ArticleTitle": question}},
                    {
                        "range": {
                            "DateCreated": {
                                "gte": "1800",
                                "lte": str(max_year),
                                "format": "dd/MM/yyyy||yyyy"
                            }
                        }
                    }
                    ,
                    {
                        "range": {
                            "ArticleDate": {
                                "gte": "1800",
                                "lte": str(max_year),
                                "format": "dd/MM/yyyy||yyyy"
                            }
                        }
                    }
                ],
                "minimum_should_match": 1,
            }
        }
    }
    
        
    res = es.search(index=doc_index, body=bod, request_timeout=120)
    #print(res)
    return res['hits']['hits']
def get_the_recalls():
    recalls1 = []
    recalls2 = []
    
    for q in tqdm(dev_data['queries']):
        qtext = q['query_text']
        #####
        results1 = get_multi(qtext, 100)
        results2 = get_compound_2(qtext, 100)
        
        #####
        retr_pmids1 = [t['_source']['pmid'] for t in results1]
        retr_pmids2 = [t['_source']['pmid'] for t in results2]
        
        #####
        rel_ret1 = sum([1 if (t in q['relevant_documents']) else 0 for t in retr_pmids1])
        rel_ret2 = sum([1 if (t in q['relevant_documents']) else 0 for t in retr_pmids2])
        
        #####
        recall1 = float(rel_ret1) / float(len(q['relevant_documents']))
        recall2 = float(rel_ret2) / float(len(q['relevant_documents']))
       
        #####
        recalls1.append(recall1)
        recalls2.append(recall2)
       
    #################
    r1 = sum(recalls1) / float(len(recalls1))
    r2 = sum(recalls2) / float(len(recalls2))
  
    return r1, r2

def put_b_k1(b, k1):
    print(es.indices.close(index = doc_index))
    print(es.indices.put_settings(
        index = doc_index,
        body  = {
            "similarity": {
                "my_similarity": {
                    "type": "BM25",
                    "b"  : b,
                    "k1" : k1
                }
            }
        }
    ))
    print(es.indices.open(index = doc_index))

def measure_existisng_system(dev_data, dev_docs):
    recalls = []
    for q in dev_data['queries']:
        recalls.append(float(q['num_rel_ret']) / float(len(q['relevant_documents']))) # float(q['num_rel']))
    print(sum(recalls) / float(len(recalls)))



es = Elasticsearch(
   ['localhost:9200'],
    verify_certs        = True,
    timeout             = 150,
    max_retries         = 10,
    retry_on_timeout    = True
)

dataloc='./Data/bioasq_data/'
w2v_bin_path                = './Data/PretrainedWeightsAndVectors/pubmed2018_w2v_30D.bin'
idf_pickle_path             = './Data/PretrainedWeightsAndVectors/idf.pkl'
(dev_data, dev_docs, train_data, train_docs, idf, max_idf, bioasq7_data) = load_all_data(dataloc, idf_pickle_path)


with open('./Data/stopwords.pkl', 'rb') as f:
    stopwords = pickle.load(f)




for b_ in tqdm(range(0, 105, 10)):
    for k1_ in tqdm(range(0, 205, 20)):
        #################
        b   = b_  / 100.0
        k1  = k1_ / 100.0
        #################
        put_b_k1(b, k1)
        #################
        r1, r2 = get_the_recalls()
        print(b, k1, r1, r2)
        sys.stdout.flush()


'''
# TO TUNE BM25 in ELK:


b   : a weight for doc length           default 0.75
k1  : a weight for term frequencies     default 1.2

curl -XPOST 'http://192.168.188.79:9201/pubmed_abstracts_joint_0_1/_close'
curl -XPUT "http://192.168.188.79:9201/pubmed_abstracts_joint_0_1/_settings" -d '
{
    "similarity": {
        "my_similarity": { 
            "type": "BM25",
            "b"                 : 0.1,
            "k1"                : 0.9,
            "discount_overlaps" : true
        }
    }
}'
     

curl -XGET 'http://192.168.188.79:9201/pubmed_abstracts_joint_0_1/_settings'
curl -XPOST 'http://192.168.188.79:9201/pubmed_abstracts_joint_0_1/_open'



'''


