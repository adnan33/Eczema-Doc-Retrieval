import json
import os,glob 
import ast
import pprint
from collections import OrderedDict as od
from sklearn.model_selection import train_test_split

src_dir=r"D:\aueb-bioasq7-master\Outputs\Responses\sorted_responses\sorted\\"
mods_dir=r"D:\aueb-bioasq7-master\shell_scripts\\"
dlist=[]
def mod_data(fno):
    try:
        with open(src_dir+"response{}.json".format(fno),"r+") as f:
            data=json.load(f)
        
        mod_q=od()
        
        docs=[]
        for i in range(10):
            docs.append(data["results"]["questions"][0]["documents"][str(i+1)])
        mod_q["body"]=data["results"]["questions"][0]["body"]
        mod_q["documents"]=docs
        mod_q["exact_answer"]=[data["results"]["questions"][0]["snippets"][0]["text"]]
        mod_q["type"]="summary"
        mod_q["id"]=data["results"]["questions"][0]["id"]
        mod_q["snippets"]=data["results"]["questions"][0]["snippets"]
        
        new_snipps=[]
        for snippet in mod_q["snippets"]:
            del snippet["article"]
            del snippet["docrank"]
            temp_dict=od()
            temp_dict["offsetInBeginSection"]=snippet["offsetInBeginSection"]
            temp_dict["offsetInEndSection"]=snippet["offsetInEndSection"]
            temp_dict["text"]=snippet["text"]
            temp_dict['beginSection']=snippet["beginSection"]
            temp_dict["document"]=snippet["document"]
            temp_dict["endSection"]=snippet["endSection"]
            new_snipps.append(temp_dict)
            #pprint.pprint(json.dumps(temp_dict))
        mod_q["snippets"]=new_snipps
        return mod_q
    except FileNotFoundError as e:
        pass


if __name__=='__main__':
    frange=421
    for i in range(frange):
        dlist.append(mod_data(i+1))
        
    dlist=list(filter(None,dlist))
    trainset,devset=train_test_split(dlist,test_size=0.1,random_state=42)
    
    print(len(trainset),len(devset))
    final_data={"questions":dlist}
    train={"questions":trainset}
    dev={"questions":devset}
    with open(mods_dir+"trainq.json","w+") as f:
                json.dump(train, f,indent=4)
    with open(mods_dir+"devq.json","w+") as f:
                json.dump(dev, f,indent=4)
    with open(mods_dir+"allq-{}.json".format(frange),"w+") as f:
                json.dump(final_data, f,indent=4)
    os.system(r'D:\aueb-bioasq7-master\shell_scripts\create_trainset_py.sh')

        