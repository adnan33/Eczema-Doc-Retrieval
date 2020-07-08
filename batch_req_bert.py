import pandas as pd
import json
import requests
from builtins import enumerate
import time
start_time = time.time()

url = "http://127.0.0.1:9250/get_eczema_docs"
response_dir=r"D:\aueb-bioasq7-master\Outputs\Responses\ES-response-Psoriasis2\\"
qdir=r"D:\aueb-bioasq7-master\interim_resources\psoriasis_ques.csv"
meta=[]

def do_request(data):
    file_dir=response_dir+"response{}.json".format(data['id'])
    r=requests.post(url,data=json.dumps(data))
    print(r.request.body)
    meta.append(r.elapsed.total_seconds()/60)
    with open(file_dir,"w+") as f:
        json.dump(json.loads(r.text),f,indent=4)
    
qdf=pd.read_csv(qdir,header=None,engine='python')
ques=qdf[1].values
ids=qdf[0].values

for i,qs in enumerate(ques):
    data = {
    'question'  : qs,
    'id'        : str(ids[i]),
    }
    print(data)
    do_request(data)
    
    
meta.append((time.time() - start_time)/3600)
metadf = pd.DataFrame(data={"Time per call(min)": meta})
metadf.to_csv("ES2-api_time.csv", sep=',')
print("questions took {:.6f} hours".format((time.time() - start_time)/3600))