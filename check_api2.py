import requests
from pprint import pprint
import json
url = "http://127.0.0.1:9250/jpdrmm/search"
data = {
    'question'  : 'What is contact dermatitis?',
    # "question"  : 'Is smoking related to blindness ?',
    'id'        : '114',
}
print(json.dumps(data))

r=requests.post(url,data=json.dumps(data))
print(r.request.body)
print(r.request.headers)
pprint(r.text)
with open("response{}.json".format(data['id']),"w+") as f:
    json.dump(json.loads(r.text),f,indent=4)
print("time taken :{}".format(r.elapsed.total_seconds()))

