import json
import os,glob 
res_dir=r"D:\aueb-bioasq7-master\Outputs\Responses\bert_responses\\"
mod_res_dir=r"D:\aueb-bioasq7-master\Outputs\Responses\mod_bert_responses\\"
os.makedirs(mod_res_dir, 755, True)
for j in range(422):
    
    with open(res_dir+"response{}.json".format(j+1),"r+") as f:
        data=json.load(f)
        
    
    top10docs=data["results"]["questions"][0]["documents"]
    docdict={str(i+1):doc for i,doc in enumerate(top10docs)}
    print(docdict)
    snips=data["results"]["questions"][0]["snippets"]
    for snip in snips:
      for i,doc in enumerate(top10docs):
          if(snip["document"]==doc):
              snip["docrank"]=str(i+1)
    data["results"]["questions"][0]["snippets"]=snips
    data["results"]["questions"][0]["documents"]=docdict
    
    with open(mod_res_dir+"response{}.json".format(j+1),"w+") as f:
        json.dump(data, f,indent=4)