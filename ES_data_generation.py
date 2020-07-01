
import sys, os, re, json, pickle, ijson,json
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
    print(res)
    return res['hits']['hits']
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
es = Elasticsearch(
   ['localhost:9200'],
    verify_certs        = True,
    timeout             = 150,
    max_retries         = 10,
    retry_on_timeout    = True
)

with open('./Data/stopwords.pkl', 'rb') as f:
    stopwords = pickle.load(f)
	
results         = get_multi(qtext, 100,2020)	
qid = 15
scores = []
def process_mesh(mesh_dict):
    processed_mesh = []
    for mesh in mesh_dict:
        processed_mesh.append(mesh["UI"]+":"+ mesh["name"])
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
            query_doc["Keywords"].extend(convert_chemical_to_keyword(rdoc["Chemicals"]))
		query_doc["journalName"] = rdoc["Title"]
        
        query_docs[id] = query_doc
    return query_docs
		
def create_data(results):
	query_data = {}
    query_data['query_id'] = qid
	query_data['query_text'] = qtext
	query_data['relevant_documents'] = []
	query_data['num_rel'] = 0
	query_data['retrieved_documents'] = []
    for result in results:
        
	
