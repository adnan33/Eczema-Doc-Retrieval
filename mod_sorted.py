
import json
import os,glob,shutil
import ast
import pprint

ranked_dir=r"D:\aueb-bioasq7-master\Outputs\Responses\sorted_responses\\"
mod_res_dir=r"D:\aueb-bioasq7-master\Outputs\Responses\sorted_responses\sorted\\"
res_dir=r"D:\aueb-bioasq7-master\Outputs\Responses\mod_bert_responses\\"

def sort_response(fno):
    reranks=[]
    print(fno)
    try:
        with open(ranked_dir+"response{}.json".format(fno),"r+") as f:
            for i,line in enumerate(f):
                if(i>=9 and i<=28):
                    reranks.append(line.strip())
        
        reranks=[re.split(":")[0] for re in reranks]
        
        rankdict={}
        
        for rank in reranks:
            rankdict[rank.split("\"")[1]]=rank.split("\"")[0].strip().split("--")[1]
        ok=[int(i) for i in list(rankdict.values())]
        #ok.sort()
        print(ok)
        with open(res_dir+"response{}.json".format(fno),"r+") as f:
            data=json.load(f)
        top10arts=data["results"]["questions"][0]["articles"]
        top10docs=data["results"]["questions"][0]["documents"]
        snippets=data["results"]["questions"][0]["snippets"]
        #mod the article links
        newarts={}
        for i in range(20):
            newarts[str(rankdict[str(i+1)])]=top10arts[str(i+1)]
        newarts2={}
        for i in range(20):
            newarts2[str(i+1)]=newarts[str(i+1)]
        
        #mod the article links
        newdocs={}
        for i in range(20):
            newdocs[str(rankdict[str(i+1)])]=top10docs[str(i+1)]
        newdocs2={}
        for i in range(20):
            newdocs2[str(i+1)]=newdocs[str(i+1)]
        
        #mod the snippets
        newsnips=[]
        for snip in snippets:
            snip["docrank"]=rankdict[str(snip["docrank"])]
            
        for i in range(20):
            for snip in snippets:
                if(int(snip["docrank"])==i+1):
                    newsnips.append(snip)
        
        data["results"]["questions"][0]["articles"]=newarts2
        data["results"]["questions"][0]["documents"]=newdocs2
        data["results"]["questions"][0]["snippets"]=newsnips
        with open(mod_res_dir+"response{}.json".format(fno),"w+") as f:
                json.dump(data, f,indent=4)
    except FileNotFoundError as e:
        print(e)
def copy_skipped(fno):
    fsrc=ranked_dir+"response{}-skipped.json".format(fno)
    fdst=mod_res_dir+"response{}.json".format(fno)
    if(os.path.exists(fsrc)):
        print("skipped_file:",fno)
        shutil.copy2(fsrc, fdst)
        
if __name__== "__main__":
    for i in range(421):
        sort_response(i+1)
        copy_skipped(i+1)