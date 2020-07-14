#!/usr/bin/env bash

#touch .interim_resources/sample_bm25_retrieval.txt

echo "{}" > ./interim_resources/sample_bm25_retrieval.txt

#python queries2galago.py interim_resources/sample.json interim_resources/sample_out.json Data/stopwords.pkl

./galago_idx.bat

python generate_bioasq_data.py interim_resources/sample.json interim_resources/sample_bm25_retrieval.txt sample 2018

