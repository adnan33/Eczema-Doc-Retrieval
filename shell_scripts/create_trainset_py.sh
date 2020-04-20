#!/usr/bin/env bash

cd shell_scripts/

touch train_bm25_retrieval.txt

echo "{}" > ./train_bm25_retrieval.txt

python ../queries2galago.py trainq.json trainq_out.json ../Data/stopwords.pkl

./get_train_galago_idx.bat

python generate_bioasq_data.py trainq.json train_bm25_retrieval.txt train 2018

touch dev_bm25_retrieval.txt

echo "{}" > ./dev_bm25_retrieval.txt

python ../queries2galago.py devq.json devq_out.json ../Data/stopwords.pkl

./get_dev_galago_idx.bat

python generate_bioasq_data.py devq.json dev_bm25_retrieval.txt dev 2018