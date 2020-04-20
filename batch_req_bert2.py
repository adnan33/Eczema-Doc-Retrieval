import pandas as pd
import json
import requests
from builtins import enumerate
import time
start_time = time.time()

url = "http://127.0.0.1:9251/bert/search"
response_dir=r"D:\aueb-bioasq7-master\Outputs\Responses\bert_responses\\"
qdir="Eczema questions for BERT-JPDRMM.csv"
meta=[]

def do_request(data):
    file_dir=response_dir+"response{}.json".format(data['id'])
    r=requests.post(url,data=json.dumps(data))
    print(r.request.body)
    meta.append(r.elapsed.total_seconds()/60)
    with open(file_dir,"w+") as f:
        json.dump(json.loads(r.text),f,indent=4)
    
qdf=pd.read_csv(qdir,header=None,engine='python')
ques=qdf[1].values[333:]
ids=qdf[0].values[333:]

for i,qs in enumerate(ques):
    data = {
    'question'  : qs,
    # "question"  : 'Is smoking related to blindness ?',
    'id'        : str(ids[i]),
    }
    print(data)
    do_request(data)
    
    
meta.append((time.time() - start_time)/3600)
metadf = pd.DataFrame(data={"Time per call(min)": meta})
metadf.to_csv("api_time2.csv", sep=',')
print("questions took {:.6f} hours".format((time.time() - start_time)/3600))