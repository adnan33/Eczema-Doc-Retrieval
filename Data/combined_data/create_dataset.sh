#!/usr/bin/env bash

touch train_bm25_retrieval.txt

echo "{}" > ./train_bm25_retrieval.txt

python ../../queries2galago.py traindata.json traindata_out.json ../stopwords.pkl

./get_train_galago_idx.bat

python generate_bioasq_data.py traindata.json train_bm25_retrieval.txt train 2018

touch dev_bm25_retrieval.txt

echo "{}" > ./dev_bm25_retrieval.txt

python ../../queries2galago.py validationdata.json validationdata_out.json ../stopwords.pkl

./get_dev_galago_idx.bat

python generate_bioasq_data.py validationdata.json dev_bm25_retrieval.txt dev 2018