from elasticsearch import Elasticsearch



doc_index = 'pubmed_abstracts_index_0_1'


def put_b_k1(b, k1):
    print(es.indices.close(index = doc_index))
    print(es.indices.put_settings(
        index = doc_index,
        body  = {
            "similarity": {
                "my_similarity": {
                    "type": "BM25",
                    "b"  : b,
                    "k1" : k1,
                    "discount_overlaps" : "true"
                }
            }
        }
    ))
    print(es.indices.open(index = doc_index))



es = Elasticsearch(
   ['localhost:9200'],
    verify_certs        = True,
    timeout             = 150,
    max_retries         = 10,
    retry_on_timeout    = True
)


put_b_k1(0.2,0.9)
print(es.indices.get_settings())

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


