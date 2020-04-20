import os
import chardet
import nltk
target_file_name=r"ok.txt"
ff_name="interim_resources/sample_bm25_retrieval.txt"

with open(ff_name, 'rb') as source_file:
    contents = source_file.read()
    if(chardet.detect(contents)["encoding"]=="UTF-16"):
        with open(target_file_name, 'w+b') as dest_file:
            dest_file.write(contents.decode('utf-16').encode('utf-8'))
if(chardet.detect(contents)["encoding"]=="UTF-16"):
    os.remove(ff_name)
    os.rename(target_file_name,ff_name)
nltk.download('punkt')