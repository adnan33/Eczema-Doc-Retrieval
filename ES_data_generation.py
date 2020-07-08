
import sys
import os
import re
import json
import pickle
import ijson
import json
from elasticsearch import Elasticsearch
from tqdm import tqdm
from pprint import pprint
import numpy as np


# Modified bioclean: also split on dashes. Works better for retrieval with galago.
bioclean_mod = lambda t: re.sub(
    '[.,?;*!%^&_+():-\[\]{}]', '',
    t.replace('"', '').replace('/', '').replace('\\',
              '').replace("'", '').replace("-", ' ').strip().lower()
).split()
bioclean = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace(
    '/', '').replace('\\', '').replace("'", '').strip().lower()).split()

doc_index = 'pubmed_abstracts_index_0_1'
map = "pubmed_abstracts_mapping_0_1"


def get_multi(qtext, n, max_year=2020):
    tokenized_body = bioclean_mod(qtext)
    question_tokens = [t for t in tokenized_body if t not in stopwords]
    question = ' '.join(question_tokens)
    bod = {
    "size": n,
   "query": {
        "multi_match": {
            "query": question,
            "type":       "most_fields",
            "fields": ["AbstractText", "ArticleTitle"]
        }
    }
}

    res = es.search(index=doc_index, body=bod, request_timeout=120)
    print(json.dumps(res))
    return res['hits']['hits']


def get_compound(qtext, n, max_year=2020):
    tokenized_body = bioclean_mod(qtext)
    #print(tokenized_body)
    question_tokens = [t for t in tokenized_body if t not in stopwords]
    #print(question_tokens)
    question = ' '.join(question_tokens)
    #print(question)
    #print(qtext)
    
    question = qtext
    #print(question)
    bod ={
        "size": n,
        "query": {
            "bool" : {
                
                "should": [
                    {"match": {"AbstractText": question}},
                    {"match": {"ArticleTitle": question}},

                 
                    {
                        "range": {
                            "ArticleDate": {
                                "gte": "1800",
                                "lte": str(max_year),
                                "format": "dd/MM/yyyy||yyyy"
                            }
                        }
                    },
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
                "minimum_should_match": 1
            }
        }
    }
    bod=json.dumps(bod)
    res = es.search(index=doc_index, body=bod, request_timeout=120)
    #print(json.dumps(res))
    return res['hits']['hits']


def get_normalized_scores(results):
    scores = [result["_score"] for result in results]
    #print(scores)
    if np.std(scores) == 0:
        pass
        # print(q)
    scores_mean = np.mean(scores)
    scores_std = np.std(scores)
    if scores_std != 0:
        norm_scores = (scores - scores_mean) / scores_std
    else:
        norm_scores = scores
    #print(norm_scores)
    return norm_scores


def process_mesh(mesh_dict):
    processed_mesh = []
    for mesh in mesh_dict:
        processed_mesh.append(mesh["UI"]+":" + mesh["name"])
    return processed_mesh


def convert_mesh_to_keyword(mesh_dict):
    converted_mesh = []
    for mesh in mesh_dict:
        converted_mesh.append(mesh["name"])
    return converted_mesh


def convert_chemical_to_keyword(chemical_list):
    converted_chemical = []
    for chemical in chemical_list:
        converted_chemical.append(chemical["NameOfSubstance"])
    return converted_chemical


def create_docset(results):
    query_docs = {}
    for result in results:
        query_doc = {}
        rdoc = result["_source"]
        id = rdoc["pmid"]
        query_doc["pmid"] = id
        query_doc["title"] = rdoc["ArticleTitle"]
        query_doc["abstractText"] = rdoc["AbstractText"]
        query_doc["publicationDate"] = rdoc["DateCreated"]

        if "Keywords" in rdoc:
            query_doc["Keywords"] = rdoc["Keywords"]
        else:
            query_doc["Keywords"] = []
        if "MeshHeadings" in rdoc:
            query_doc["meshHeadingsList"] = rdoc["MeshHeadings"]
            query_doc["Keywords"].extend(convert_mesh_to_keyword(rdoc["MeshHeadings"]))
        else:
            query_doc["meshHeadingsList"] = []

        if "Chemicals" in rdoc:
            query_doc["Keywords"].extend(
                convert_chemical_to_keyword(rdoc["Chemicals"]))
        query_doc["journalName"] = rdoc["Title"]

        query_docs[id] = query_doc
    return query_docs
		
def create_data(results,qbody,id):
    query_data = {}
    n_ret = 0
    n_ret_rel = 0
    query_data['query_id'] = id
    query_data['query_text'] = qbody
    query_data['relevant_documents'] = []
    query_data['num_rel'] = 0
    query_data['retrieved_documents'] = []
    norm_scores = get_normalized_scores(results)
    for idx , result in enumerate(results):
        n_ret+=1
        temp_data = {}
        temp_data["doc_id"] = result["_source"]["pmid"]
        temp_data["rank"] = idx+1
        temp_data["bm25_score"] = result["_score"]
        temp_data["norm_bm25_score"] = norm_scores[idx]
        temp_data['is_relevant'] = False
        query_data['retrieved_documents'].append(temp_data)
    query_data['num_ret'] = n_ret
    query_data['num_rel_ret'] = n_ret_rel
    data = {'queries': [query_data]}
    
    return data

def generate_data_es(data, keep_up_to_year):
    qtext = data["questions"][0]["body"]
    qid = data["questions"][0]["id"]
    results = get_compound(qtext, 100, keep_up_to_year)    
    data = create_data(results,qtext,qid)
    docset = create_docset(results)
    return data,docset    
    
es = Elasticsearch(['localhost:9200'],
    verify_certs        = True,
    timeout             = 150,
    max_retries         = 10,
    retry_on_timeout    = True
)

with open('./Data/stopwords.pkl', 'rb') as f:
    stopwords = pickle.load(f)
qid = 15 
qtext = "what are the risk to children under 12 of long term frequent topical steroid use ?"
keep_up_to_year = 2019
data = {
    "questions": [
        {
            "body"      : qtext,
            "id"        : qid,
            "documents" : []
        }
    ]
}

test_data,test_docs = generate_data_es(data, keep_up_to_year)
with open("test_op/test_data_ES.json","w+") as f:
    f.write(json.dumps(test_data))
with open("test_op/test_docs_ES.json","w+") as f:
    f.write(json.dumps(test_docs))

        
	
