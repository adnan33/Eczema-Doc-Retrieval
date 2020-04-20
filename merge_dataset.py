import json
import numpy
from sklearn.model_selection import train_test_split
from numpy import save

tr_file=r"/aueb-bioasq7-master/Data/bioasq_data/trainining7b.json"
ecz_file=r"/aueb-bioasq7-master/shell_scripts/allq-421.json"
f_dir=r"D:\aueb-bioasq7-master\Data\combined_data\\"

def read_json(file_path):
    try:
        with open(file_path,"r+") as f:
                data=json.load(f)
        return data
    except FileNotFoundError as e:
        print(e)
    
def concat_data(l1,l2):
    l=[]
    l.extend(l1)
    l.extend(l2)
    return l

def save_data(fname,data):
    data_dict={"questions":data}
    with open(f_dir+fname+".json","w+") as f:
                json.dump(data_dict, f,indent=4)
#read the json files    
data_org=read_json(tr_file)
data_ecz=read_json(ecz_file)

#split the data    
trainset,devset=train_test_split(data_ecz['questions'],test_size=0.038,random_state=42)
print(len(trainset),len(devset))

trainorg,devorg=train_test_split(data_org['questions'],test_size=0.0364,random_state=42)
print(len(trainorg),len(devorg))

#concatenate all the data
all=concat_data(data_org['questions'],data_ecz['questions'])
train=concat_data(trainorg, trainset)
valid=concat_data(devorg, devset)

print(len(all),len(train),len(valid))

save_data("alldata",all)
save_data("traindata",train)
save_data("validationdata",valid)
