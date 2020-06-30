
import sys
from importlib import reload
if sys.version[0] == '2':
    reload(sys)
    sys.setdefaultencoding("utf-8")
# sys.setdefaultencoding() does not exist, here!


import  re, logging, torch, os, json, random, subprocess, pickle, nltk
import  torch.nn.functional         as F
import  torch.nn                    as nn
import  numpy                       as np
import  torch.autograd              as autograd
from    tqdm                        import tqdm
from    pprint                      import pprint
from    nltk.tokenize               import sent_tokenize
from    difflib                     import SequenceMatcher
from    sklearn.preprocessing       import LabelEncoder
from    sklearn.preprocessing       import OneHotEncoder
from    difflib                     import SequenceMatcher
from    collections                 import OrderedDict, defaultdict
from    pymongo                     import MongoClient

from    pytorch_pretrained_bert.tokenization    import BertTokenizer
from    pytorch_pretrained_bert.modeling        import BertForSequenceClassification
from    pytorch_pretrained_bert.file_utils      import PYTORCH_PRETRAINED_BERT_CACHE
from spellchecker import SpellChecker
spell = SpellChecker()

bioclean    = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').strip().lower()).split()
softmax     = lambda z: np.exp(z) / np.sum(np.exp(z))
bioclean_mod = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').replace("-", ' ').strip().lower()).split()

try:
    stopwords   = nltk.corpus.stopwords.words("english")
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')
finally:
    stopwords   = nltk.corpus.stopwords.words("english")
global idf,max_idf

import subprocess
from flask import  Flask
from flask import request
from flask import redirect, url_for, jsonify
from flask_cors import CORS
import traceback

app = Flask(__name__)
CORS(app)



def preprocess(data, path_out):
    
    ############
    
    queries = data['questions']
    ############
    q_array = []
    for query in queries:
        ############
        tokenized_body = bioclean_mod(query['body'])
        tokenized_body = [t for t in tokenized_body if t not in stopwords]
        ############
        body = ' '.join(tokenized_body)
        q_array.append({"text": body, "number": query["id"]})
    with open(path_out, 'w+') as outfile:
        outfile.write(json.dumps({"queries": [ob for ob in q_array]}, indent=4))
    

def create_docset(docs_needed):
    print("Retrieving text for {0} documents".format(len(docs_needed)))
    docset = {}
    client = MongoClient('localhost', 27017)
    db = client.pubmedBaseline2018
    collection = db.articles
    docs_needed = list(docs_needed)
    i = 0
    step = 10000
    pbar = tqdm(total=len(docs_needed))
    while i <= len(docs_needed):
        doc_cursor = collection.find({"pmid": {"$in": docs_needed[i:i + step]}})
        for doc in doc_cursor:
            del doc['_id']
            docset[doc['pmid']] = json.loads((json.dumps(doc)))
        i += step
        pbar.update(step)
    pbar.close()
    not_found = set(docs_needed) - set(docset.keys())
    #print(list(not_found)[:100])
    #print(len(not_found))
    return docset

def create_doc_subset(docset, ret_docs_needed, rel_docs_needed):
    doc_subset = {}
    for doc_id in ret_docs_needed:
        doc_subset[doc_id] = docset[doc_id]
    for doc_id in rel_docs_needed:
        try:
            doc_subset[doc_id] = docset[doc_id]
        except KeyError:
            pass
            #print('Relevant doc {0} not found in docset.'.format(doc_id))
    return doc_subset

def add_normalized_scores(q_ret):
    for q in q_ret:
        scores = [t[1] for t in q_ret[q]]
        if np.std(scores) == 0:
            pass
            #print(q)
        scores_mean = np.mean(scores)
        scores_std = np.std(scores)
        if scores_std != 0:
            norm_scores = (scores - scores_mean) / scores_std
        else:
            norm_scores = scores
        for i in range(len(q_ret[q])):
            q_ret[q][i] += (norm_scores[i],)

def remove_recent_years(q_ret, keep_up_to_year, docset):
    new_q_ret = defaultdict(list)
    for q in q_ret:
        for t in q_ret[q]:
            # print(t)
            doc_id = t[0]
            try:
                pub_year = int(docset[doc_id]['publicationDate'].split('-')[0])
            except ValueError:
                continue
            if pub_year > keep_up_to_year:
                print(pub_year)
                continue
            new_q_ret[q].append(t)
    return new_q_ret


def call_galago(json_dir):
    
    command=['Index\\home\\document_retrieval\\galago-3.10-bin\\bin\\galago',
             'batch-search',
             '--index=Index\\home\\document_retrieval\\galago-3.10-bin\\bin\\pubmed_only_abstract_galago_index'
             ,'--verbose=False','--requested=25','--scorer=bm25','--defaultTextPart=postings.krovetz',
             '--mode=threaded', json_dir]
    rets=subprocess.Popen(command,stdout=subprocess.PIPE,shell=True)
    
    out,err=rets.communicate()
    
    lines       = out.decode("utf-8").split('\n')
    retrieval_results = defaultdict(list)
    for line in lines:
        if(len(line)>0):
            line_splits = line.split()
            q_id = line_splits[0]
            doc_id = line_splits[2]
            bm25_score = float(line_splits[4])
            retrieval_results[q_id].append((doc_id, bm25_score))
            
    return dict(retrieval_results)
    


def load_q_rels(data):
    qrels = defaultdict(list)
    n_qrels = defaultdict(int)
    for i in range(len(data['questions'])):
        q_id = data['questions'][i]['id']
        rel_docs = set([doc.split('/')[-1] for doc in data['questions'][i]['documents']])
        qrels[q_id] = rel_docs
        n_qrels[q_id] = len(rel_docs)
    return dict(qrels), dict(n_qrels)
def load_q_text(data):
    q_text = {}
    for i in range(len(data['questions'])):
        q_id = data['questions'][i]['id']
        text = data['questions'][i]['body']
        q_text[q_id] = text
    return dict(q_text)
def generate_test_data(data, json_dir, keep_up_to_year):
    q_ret = call_galago(json_dir)
    q_rel, n_qrels = load_q_rels(data)
    q_text = load_q_text(data)
    #
    docs_needed = set()
    for q_id in q_rel:
        docs_needed.update(q_rel[q_id])
    for q_id in q_ret:
        docs_needed.update([d[0] for d in q_ret[q_id]])
    docset = create_docset(docs_needed)
    #
    q_ret = remove_recent_years(q_ret, keep_up_to_year, docset)
    #
    for k in [100]:
        #print(k)
        #
        for q in q_ret:
            q_ret[q] = q_ret[q][:k]
        add_normalized_scores(q_ret)
        #
        queries = []
        retrieved_documents_set = set()
        relevant_documents_set = set()
        for q in q_ret:
            query_data = {}
            query_data['query_id'] = q
            query_data['query_text'] = q_text[q]
            query_data['relevant_documents'] = sorted(list(q_rel[q]))
            query_data['num_rel'] = n_qrels[q]
            query_data['retrieved_documents'] = []
            rank = 0
            n_ret_rel = 0
            n_ret = 0
            for t in q_ret[q][:k]:
                n_ret += 1
                doc_id = t[0]
                bm25_score = t[1]
                norm_bm25_score = t[2]
                rank += 1
                #
                retrieved_documents_set.add(doc_id)
                relevant_documents_set.update(q_rel[q])
                #
                doc_data = {}
                doc_data['doc_id'] = doc_id
                doc_data['rank'] = rank
                doc_data['bm25_score'] = bm25_score
                doc_data['norm_bm25_score'] = norm_bm25_score
                if doc_id in q_rel[q]:
                    doc_data['is_relevant'] = True
                    n_ret_rel += 1
                else:
                    doc_data['is_relevant'] = False
                query_data['retrieved_documents'].append(doc_data)
            query_data['num_ret'] = n_ret
            query_data['num_rel_ret'] = n_ret_rel
            queries.append(query_data)
            data = {'queries': queries}
        
        # Create doc subset for the top-k documents (to avoid many queries to mongodb for each k value)
        doc_subset = create_doc_subset(docset, retrieved_documents_set, relevant_documents_set)
    return data,doc_subset
class InputExample(object):
    """A single training/test example for simple sequence classification."""
    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

def fix_bert_tokens(tokens):
    ret = []
    for t in tokens:
        if (t.startswith('##')):
            ret[-1] = ret[-1] + t[2:]
        else:
            ret.append(t)
    return ret

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    ####
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def convert_examples_to_features(examples, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]
        ####
        tokens          = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids     = [0] * len(tokens)
        ####
        if tokens_b:
            tokens      += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)
        input_ids       = tokenizer.convert_tokens_to_ids(tokens)
        ####
        input_mask      = [1] * len(input_ids)
        ####
        padding         = [0] * (max_seq_length - len(input_ids))
        input_ids       += padding
        input_mask      += padding
        segment_ids     += padding
        ####
        assert len(input_ids)   == max_seq_length
        assert len(input_mask)  == max_seq_length
        assert len(segment_ids) == max_seq_length
        ####
        in_f        = InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, label_id=0)
        in_f.tokens = tokens
        features.append(in_f)
    return features

def embed_the_sent(sent):
    eval_examples   = [InputExample(guid='example_dato_1', text_a=sent, text_b=None, label='1')]
    eval_features   = convert_examples_to_features(eval_examples, max_seq_length, bert_tokenizer)
    eval_feat       = eval_features[0]
    input_ids       = torch.tensor([eval_feat.input_ids], dtype=torch.long).to(device)
    input_mask      = torch.tensor([eval_feat.input_mask], dtype=torch.long).to(device)
    segment_ids     = torch.tensor([eval_feat.segment_ids], dtype=torch.long).to(device)
    tokens          = eval_feat.tokens
    with torch.no_grad():
        token_embeds, pooled_output = bert_model.bert(input_ids, segment_ids, input_mask, output_all_encoded_layers=False)
        tok_inds                    = [i for i in range(len(tokens)) if(not tokens[i].startswith('##'))]
        token_embeds                = token_embeds.squeeze(0)
        embs                        = token_embeds[tok_inds,:]
    fixed_tokens = fix_bert_tokens(tokens)
    return fixed_tokens, embs

def tf(term, document):
    tf = 0
    for word in document:
        if word == term:
            tf += 1
    if len(document) == 0:
        return tf
    else:
        return tf / len(document)

def similarity_score(query, document, k1, b, idf_scores, avgdl, normalize, mean, deviation, rare_word):
    score = 0
    for query_term in query:
        if query_term not in idf_scores:
            score += rare_word * (
                    (tf(query_term, document) * (k1 + 1)) /
                    (
                            tf(query_term, document) +
                            k1 * (1 - b + b * (len(document) / avgdl))
                    )
            )
        else:
            score += idf_scores[query_term] * ((tf(query_term, document) * (k1 + 1)) / (
                        tf(query_term, document) + k1 * (1 - b + b * (len(document) / avgdl))))
    if normalize:
        return ((score - mean) / deviation)
    else:
        return score

def compute_avgdl(documents):
    total_words = 0
    for document in documents:
        total_words += len(document)
    avgdl = total_words / len(documents)
    return avgdl

def weighted_binary_cross_entropy(output, target, weights=None):
    if weights is not None:
        assert len(weights) == 2
        loss = weights[1] * (target * torch.log(output)) + weights[0] * ((1 - target) * torch.log(1 - output))
    else:
        loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)
    return torch.neg(torch.mean(loss))

def RemoveTrainLargeYears(data, doc_text):
    for i in range(len(data['queries'])):
        hyear = 1900
        for j in range(len(data['queries'][i]['retrieved_documents'])):
            if data['queries'][i]['retrieved_documents'][j]['is_relevant']:
                doc_id = data['queries'][i]['retrieved_documents'][j]['doc_id']
                year = doc_text[doc_id]['publicationDate'].split('-')[0]
                if year[:1] == '1' or year[:1] == '2':
                    if int(year) > hyear:
                        hyear = int(year)
        j = 0
        while True:
            doc_id = data['queries'][i]['retrieved_documents'][j]['doc_id']
            year = doc_text[doc_id]['publicationDate'].split('-')[0]
            if (year[:1] == '1' or year[:1] == '2') and int(year) > hyear:
                del data['queries'][i]['retrieved_documents'][j]
            else:
                j += 1
            if j == len(data['queries'][i]['retrieved_documents']):
                break
    return data

def RemoveBadYears(data, doc_text, train):
    for i in range(len(data['queries'])):
        j = 0
        while True:
            doc_id = data['queries'][i]['retrieved_documents'][j]['doc_id']
            year = doc_text[doc_id]['publicationDate'].split('-')[0]
            ##########################
            # Skip 2017/2018 docs always. Skip 2016 docs for training.
            # Need to change for final model - 2017 should be a train year only.
            # Use only for testing.
            if year == '2017' or year == '2018' or (train and year == '2016'):
                # if year == '2018' or (train and year == '2017'):
                del data['queries'][i]['retrieved_documents'][j]
            else:
                j += 1
            ##########################
            if j == len(data['queries'][i]['retrieved_documents']):
                break
    return data

def print_params(model, bert_model):
    '''
    It just prints the number of parameters in the model.
    :param model:   The pytorch model
    :return:        Nothing.
    '''
    print(40 * '=')
    print(bert_model)
    print(40 * '=')
    print(model)
    print(40 * '=')
    ##########################################################################################################
    bert_trainable = 0
    bert_untrainable = 0
    for parameter in list(bert_model.parameters()):
        # print(parameter.size())
        v = 1
        for s in parameter.size():
            v *= s
        if (parameter.requires_grad):
            bert_trainable += v
        else:
            bert_untrainable += v
    bert_total_params = bert_trainable + bert_untrainable
    print(40 * '=')
    print('BERT: trainable:{} untrainable:{} total:{}'.format(bert_trainable, bert_untrainable, bert_total_params))
    print(40 * '=')
    ##########################################################################################################
    model_trainable = 0
    model_untrainable = 0
    for parameter in list(model.parameters()):
        v = 1
        for s in parameter.size():
            v *= s
        if (parameter.requires_grad):
            model_trainable += v
        else:
            model_untrainable += v
    model_total_params = model_trainable + model_untrainable
    print(40 * '=')
    print('MODEL: trainable:{} untrainable:{} total:{}'.format(model_trainable, model_untrainable, model_total_params))
    print(40 * '=')

def compute_the_cost(costs, back_prop=True):
    cost_ = torch.stack(costs)
    cost_ = cost_.sum() / (1.0 * cost_.size(0))
    if (back_prop):
        cost_.backward()
        optimizer.step()
        optimizer.zero_grad()
    the_cost = cost_.cpu().item()
    return the_cost

def save_checkpoint(epoch, model, max_dev_map, optimizer, filename='checkpoint.pth.tar'):
    '''
    :param state:       the stete of the pytorch mode
    :param filename:    the name of the file in which we will store the model.
    :return:            Nothing. It just saves the model.
    '''
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'best_valid_score': max_dev_map,
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state, filename)

def get_map_res(fgold, femit):
    trec_eval_res = subprocess.Popen(['python', eval_path, fgold, femit], stdout=subprocess.PIPE, shell=False)
    (out, err) = trec_eval_res.communicate()
    lines = out.decode("utf-8").split('\n')
    map_res = [l for l in lines if (l.startswith('map '))][0].split('\t')
    map_res = float(map_res[-1])
    return map_res

def get_bioasq_res(prefix, data_gold, data_emitted, data_for_revision):
    '''
    java -Xmx10G -cp /home/dpappas/for_ryan/bioasq6_eval/flat/BioASQEvaluation/dist/BioASQEvaluation.jar
    evaluation.EvaluatorTask1b -phaseA -e 5
    /home/dpappas/for_ryan/bioasq6_submit_files/test_batch_1/BioASQ-task6bPhaseB-testset1
    ./drmm-experimental_submit.json
    '''
    jar_path = retrieval_jar_path
    #
    fgold = '{}_data_for_revision.json'.format(prefix)
    fgold = os.path.join(odir, fgold)
    fgold = os.path.abspath(fgold)
    with open(fgold, 'w') as f:
        f.write(json.dumps(data_for_revision, indent=4, sort_keys=True))
        f.close()
    #
    for tt in data_gold['questions']:
        if ('exact_answer' in tt):
            del (tt['exact_answer'])
        if ('ideal_answer' in tt):
            del (tt['ideal_answer'])
        if ('type' in tt):
            del (tt['type'])
    fgold = '{}_gold_bioasq.json'.format(prefix)
    fgold = os.path.join(odir, fgold)
    fgold = os.path.abspath(fgold)
    with open(fgold, 'w') as f:
        f.write(json.dumps(data_gold, indent=4, sort_keys=True))
        f.close()
    #
    femit = '{}_emit_bioasq.json'.format(prefix)
    femit = os.path.join(odir, femit)
    femit = os.path.abspath(femit)
    with open(femit, 'w') as f:
        f.write(json.dumps(data_emitted, indent=4, sort_keys=True))
        f.close()
    #
    bioasq_eval_res = subprocess.Popen(
        [
            'java', '-Xmx10G', '-cp', jar_path, 'evaluation.EvaluatorTask1b',
            '-phaseA', '-e', '5', fgold, femit
        ],
        stdout=subprocess.PIPE, shell=False
    )
    (out, err) = bioasq_eval_res.communicate()
    lines = out.decode("utf-8").split('\n')
    ret = {}
    for line in lines:
        if (':' in line):
            k = line.split(':')[0].strip()
            v = line.split(':')[1].strip()
            ret[k] = float(v)
    return ret

def similar(upstream_seq, downstream_seq):
    upstream_seq = upstream_seq.encode('ascii', 'ignore')
    downstream_seq = downstream_seq.encode('ascii', 'ignore')
    s = SequenceMatcher(None, upstream_seq, downstream_seq)
    match = s.find_longest_match(0, len(upstream_seq), 0, len(downstream_seq))
    upstream_start = match[0]
    upstream_end = match[0] + match[2]
    longest_match = upstream_seq[upstream_start:upstream_end]
    to_match = upstream_seq if (len(downstream_seq) > len(upstream_seq)) else downstream_seq
    r1 = SequenceMatcher(None, to_match, longest_match).ratio()
    return r1

def get_pseudo_retrieved(dato):
    some_ids = [item['document'].split('/')[-1].strip() for item in bioasq7_data[dato['query_id']]['snippets']]
    pseudo_retrieved = [
        {
            'bm25_score': 7.76,
            'doc_id': id,
            'is_relevant': True,
            'norm_bm25_score': 3.85
        }
        for id in set(some_ids)
    ]
    return pseudo_retrieved

def get_snippets_loss(good_sent_tags, gs_emits_, bs_emits_):
    wright = torch.cat([gs_emits_[i] for i in range(len(good_sent_tags)) if (good_sent_tags[i] == 1)])
    wrong = [gs_emits_[i] for i in range(len(good_sent_tags)) if (good_sent_tags[i] == 0)]
    wrong = torch.cat(wrong + [bs_emits_.squeeze(-1)])
    losses = [model.my_hinge_loss(w.unsqueeze(0).expand_as(wrong), wrong) for w in wright]
    return sum(losses) / float(len(losses))

def get_two_snip_losses(good_sent_tags, gs_emits_, bs_emits_):
    bs_emits_ = bs_emits_.squeeze(-1)
    gs_emits_ = gs_emits_.squeeze(-1)
    good_sent_tags = torch.FloatTensor(good_sent_tags)
    tags_2 = torch.zeros_like(bs_emits_)
    if (use_cuda):
        good_sent_tags = good_sent_tags.cuda()
        tags_2 = tags_2.cuda()
    #
    # sn_d1_l = F.binary_cross_entropy(gs_emits_, good_sent_tags, size_average=False, reduce=True)
    # sn_d2_l = F.binary_cross_entropy(bs_emits_, tags_2, size_average=False, reduce=True)
    sn_d1_l = F.binary_cross_entropy(gs_emits_, good_sent_tags, reduction='sum')
    sn_d2_l = F.binary_cross_entropy(bs_emits_, tags_2, reduction='sum')
    return sn_d1_l, sn_d2_l

def init_the_logger(hdlr):
    if not os.path.exists(odir):
        os.makedirs(odir)
    od = odir.split('/')[-1]  # 'sent_posit_drmm_MarginRankingLoss_0p001'
    logger = logging.getLogger(od)
    if (hdlr is not None):
        logger.removeHandler(hdlr)
    hdlr = logging.FileHandler(os.path.join(odir, 'model.log'))
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    return logger, hdlr

def get_words(s, idf, max_idf):
    sl = tokenize(s)
    sl = [s for s in sl]
    sl2 = [s for s in sl if idf_val(s, idf, max_idf) >= 2.0]
    return sl, sl2

def tokenize(x):
    x_tokens = bert_tokenizer.tokenize(x)
    x_tokens = fix_bert_tokens(x_tokens)
    return x_tokens

def idf_val(w, idf, max_idf):
    if w in idf:
        return idf[w]
    return max_idf

def load_idfs(idf_path, words):
    print('Loading IDF tables')
    #
    # with open(dataloc + 'idf.pkl', 'rb') as f:
    with open(idf_path, 'rb') as f:
        idf = pickle.load(f)
    ret = {}
    for w in words:
        if w in idf:
            ret[w] = idf[w]
    max_idf = 0.0
    for w in idf:
        if idf[w] > max_idf:
            max_idf = idf[w]
    idf = None
    print('Loaded idf tables with max idf {}'.format(max_idf))
    #
    return ret, max_idf

def uwords(words):
    uw = {}
    for w in words:
        uw[w] = 1
    return [w for w in uw]

def ubigrams(words):
    uw = {}
    prevw = "<pw>"
    for w in words:
        uw[prevw + '_' + w] = 1
        prevw = w
    return [w for w in uw]

def query_doc_overlap(qwords, dwords, idf, max_idf):
    # % Query words in doc.
    qwords_in_doc = 0
    idf_qwords_in_doc = 0.0
    idf_qwords = 0.0
    for qword in uwords(qwords):
        idf_qwords += idf_val(qword, idf, max_idf)
        for dword in uwords(dwords):
            if qword == dword:
                idf_qwords_in_doc += idf_val(qword, idf, max_idf)
                qwords_in_doc += 1
                break
    if len(qwords) <= 0:
        qwords_in_doc_val = 0.0
    else:
        qwords_in_doc_val = (float(qwords_in_doc) /
                             float(len(uwords(qwords))))
    if idf_qwords <= 0.0:
        idf_qwords_in_doc_val = 0.0
    else:
        idf_qwords_in_doc_val = float(idf_qwords_in_doc) / float(idf_qwords)
    # % Query bigrams  in doc.
    qwords_bigrams_in_doc = 0
    idf_qwords_bigrams_in_doc = 0.0
    idf_bigrams = 0.0
    for qword in ubigrams(qwords):
        wrds = qword.split('_')
        idf_bigrams += idf_val(wrds[0], idf, max_idf) * idf_val(wrds[1], idf, max_idf)
        for dword in ubigrams(dwords):
            if qword == dword:
                qwords_bigrams_in_doc += 1
                idf_qwords_bigrams_in_doc += (idf_val(wrds[0], idf, max_idf) * idf_val(wrds[1], idf, max_idf))
                break
    if len(qwords) <= 0:
        qwords_bigrams_in_doc_val = 0.0
    else:
        qwords_bigrams_in_doc_val = (float(qwords_bigrams_in_doc) / float(len(ubigrams(qwords))))
    if idf_bigrams <= 0.0:
        idf_qwords_bigrams_in_doc_val = 0.0
    else:
        idf_qwords_bigrams_in_doc_val = (float(idf_qwords_bigrams_in_doc) / float(idf_bigrams))
    return [
        qwords_in_doc_val,
        qwords_bigrams_in_doc_val,
        idf_qwords_in_doc_val,
        idf_qwords_bigrams_in_doc_val
    ]

def GetScores(qtext, dtext, bm25, idf, max_idf):
    qwords, qw2 = get_words(qtext, idf, max_idf)
    dwords, dw2 = get_words(dtext, idf, max_idf)
    qd1 = query_doc_overlap(qwords, dwords, idf, max_idf)
    bm25 = [bm25]
    return qd1[0:3] + bm25

def GetWords(data, doc_text, words):
    for i in tqdm(range(len(data['queries'])), ascii=True):
        qwds = tokenize(data['queries'][i]['query_text'])
        for w in qwds:
            words[w] = 1
        for j in range(len(data['queries'][i]['retrieved_documents'])):
            doc_id = data['queries'][i]['retrieved_documents'][j]['doc_id']
            dtext = (
                    doc_text[doc_id]['title'] + ' <title> ' + doc_text[doc_id]['abstractText']
                    # +
                    # ' '.join(
                    #     [
                    #         ' '.join(mm) for mm in
                    #         get_the_mesh(doc_text[doc_id])
                    #     ]
                    # )
            )
            dwds = tokenize(dtext)
            for w in dwds:
                words[w] = 1

def get_gold_snips(quest_id, bioasq6_data):
    gold_snips = []
    if ('snippets' in bioasq6_data[quest_id]):
        for sn in bioasq6_data[quest_id]['snippets']:
            gold_snips.extend(sent_tokenize(sn['text']))
    return list(set(gold_snips))

def prep_extracted_snippets(extracted_snippets, docs, qid, top10docs, quest_body):
    ret = {
        'body'      : quest_body,
        'documents' : top10docs,
        'articles'    : {},
        'id'        : qid,
        'snippets'  : [],
    }
   
    for i,tdoc in enumerate(top10docs):
        tdoc=tdoc.split("/")[-1]
        ret['articles'][i+1]=docs[tdoc]['title'].strip("]").strip("[")
    for esnip in extracted_snippets:
        pid         = esnip[2].split('/')[-1]
        the_text    = esnip[3]
        esnip_res = {
            # 'score'     : esnip[1],
            "article"   :  docs[pid]['title'].strip("]").strip("["),
            "document"  : "http://www.ncbi.nlm.nih.gov/pubmed/{}".format(pid),
            "text"      : the_text
        }
        try:
            ind_from                            = docs[pid]['title'].index(the_text)
            ind_to                              = ind_from + len(the_text)
            
            esnip_res["beginSection"]           = "title"
            esnip_res["endSection"]             = "title"
            esnip_res["offsetInBeginSection"]   = ind_from
            esnip_res["offsetInEndSection"]     = ind_to
        except:
            # print(the_text)
            # pprint(docs[pid])
            ind_from                            = docs[pid]['abstractText'].index(the_text)
            ind_to                              = ind_from + len(the_text)
            esnip_res["beginSection"]           = "abstract"
            esnip_res["endSection"]             = "abstract"
            esnip_res["offsetInBeginSection"]   = ind_from
            esnip_res["offsetInEndSection"]     = ind_to
        ret['snippets'].append(esnip_res)
    return ret

def get_snips(quest_id, gid, bioasq6_data):
    good_snips = []
    if ('snippets' in bioasq6_data[quest_id]):
        for sn in bioasq6_data[quest_id]['snippets']:
            if (sn['document'].endswith(gid)):
                good_snips.extend(sent_tokenize(sn['text']))
    return good_snips

def get_the_mesh(the_doc):
    good_meshes = []
    if ('meshHeadingsList' in the_doc):
        for t in the_doc['meshHeadingsList']:
            t = t.split(':', 1)
            t = t[1].strip()
            t = t.lower()
            good_meshes.append(t)
    elif ('MeshHeadings' in the_doc):
        for mesh_head_set in the_doc['MeshHeadings']:
            for item in mesh_head_set:
                good_meshes.append(item['text'].strip().lower())
    if ('Chemicals' in the_doc):
        for t in the_doc['Chemicals']:
            t = t['NameOfSubstance'].strip().lower()
            good_meshes.append(t)
    good_mesh = sorted(good_meshes)
    good_mesh = ['mesh'] + good_mesh
    # good_mesh = ' # '.join(good_mesh)
    # good_mesh = good_mesh.split()
    # good_mesh = [gm.split() for gm in good_mesh]
    good_mesh = [gm for gm in good_mesh]
    return good_mesh

def snip_is_relevant(one_sent, gold_snips):
    return int(
        any(
            [
                (one_sent.encode('ascii', 'ignore') in gold_snip.encode('ascii', 'ignore'))
                or
                (gold_snip.encode('ascii', 'ignore') in one_sent.encode('ascii', 'ignore'))
                for gold_snip in gold_snips
            ]
        )
    )

def create_one_hot_and_sim(tokens1, tokens2):
    '''
    :param tokens1:
    :param tokens2:
    :return:
    exxample call : create_one_hot_and_sim('c d e'.split(), 'a b c'.split())
    '''
    label_encoder = LabelEncoder()
    onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
    #
    values = list(set(tokens1 + tokens2))
    integer_encoded = label_encoder.fit_transform(values)
    #
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoder.fit(integer_encoded)
    #
    lab1 = label_encoder.transform(tokens1)
    lab1 = np.expand_dims(lab1, axis=1)
    oh1 = onehot_encoder.transform(lab1)
    #
    lab2 = label_encoder.transform(tokens2)
    lab2 = np.expand_dims(lab2, axis=1)
    oh2 = onehot_encoder.transform(lab2)
    #
    ret = np.matmul(oh1, np.transpose(oh2), out=None)
    #
    return oh1, oh2, ret

def prep_data(quest, the_doc, the_bm25, good_snips, idf, max_idf, quest_toks):
    if(emit_only_abstract_sents):
        good_sents = sent_tokenize(the_doc['abstractText'])
    else:
        good_sents      = sent_tokenize(the_doc['title']) + sent_tokenize(the_doc['abstractText'])
    ####
    good_doc_af         = GetScores(quest, the_doc['title'] + the_doc['abstractText'], the_bm25, idf, max_idf)
    good_doc_af.append(len(good_sents) / 60.)
    #
    all_doc_text        = the_doc['title'] + ' ' + the_doc['abstractText']
    doc_toks            = tokenize(all_doc_text)
    tomi                = (set(doc_toks) & set(quest_toks))
    tomi_no_stop        = tomi - set(stopwords)
    BM25score           = similarity_score(quest_toks, doc_toks, 1.2, 0.75, idf, avgdl, True, mean, deviation, max_idf)
    tomi_no_stop_idfs   = [idf_val(w, idf, max_idf) for w in tomi_no_stop]
    tomi_idfs           = [idf_val(w, idf, max_idf) for w in tomi]
    quest_idfs          = [idf_val(w, idf, max_idf) for w in quest_toks]
    features            = [
        len(quest) / 300.,
        len(all_doc_text) / 300.,
        len(tomi_no_stop) / 100.,
        BM25score,
        sum(tomi_no_stop_idfs) / 100.,
        sum(tomi_idfs) / sum(quest_idfs),
    ]
    good_doc_af.extend(features)
    ####
    good_sents_embeds, good_sents_escores, held_out_sents, good_sent_tags, good_oh_sim = [], [], [], [], []
    for good_text in good_sents:
        sent_toks, sent_embeds  = embed_the_sent(' '.join(bioclean(good_text)))
        oh1, oh2, oh_sim        = create_one_hot_and_sim(quest_toks, sent_toks)
        good_oh_sim.append(oh_sim)
        good_escores            = GetScores(quest, good_text, the_bm25, idf, max_idf)[:-1]
        good_escores.append(len(sent_toks) / 342.)
        tomi                    = (set(sent_toks) & set(quest_toks))
        tomi_no_stop            = tomi - set(stopwords)
        BM25score               = similarity_score(quest_toks, sent_toks, 1.2, 0.75, idf, avgdl, True, mean, deviation, max_idf)
        tomi_no_stop_idfs       = [idf_val(w, idf, max_idf) for w in tomi_no_stop]
        tomi_idfs               = [idf_val(w, idf, max_idf) for w in tomi]
        quest_idfs              = [idf_val(w, idf, max_idf) for w in quest_toks]
        features                = [
            len(quest) / 300.,
            len(good_text) / 300.,
            len(tomi_no_stop) / 100.,
            BM25score,
            sum(tomi_no_stop_idfs) / 100.,
            sum(tomi_idfs) / sum(quest_idfs),
        ]
        #
        good_sents_embeds.append(sent_embeds)
        good_sents_escores.append(good_escores + features)
        held_out_sents.append(good_text)
        good_sent_tags.append(snip_is_relevant(' '.join(bioclean(good_text)), good_snips))
    ####
    return {
        'sents_embeds': good_sents_embeds,
        'sents_escores': good_sents_escores,
        'doc_af': good_doc_af,
        'sent_tags': good_sent_tags,
        'held_out_sents': held_out_sents,
        'oh_sims': good_oh_sim
    }

def do_for_one_retrieved(doc_emit_, gs_emits_, held_out_sents, retr, doc_res, gold_snips):
    emition = doc_emit_.cpu().item()
    emitss = gs_emits_.tolist()
    mmax = max(emitss)
    all_emits, extracted_from_one = [], []
    for ind in range(len(emitss)):
        t = (
            snip_is_relevant(held_out_sents[ind], gold_snips),
            emitss[ind],
            "http://www.ncbi.nlm.nih.gov/pubmed/{}".format(retr['doc_id']),
            held_out_sents[ind]
        )
        all_emits.append(t)
        # extracted_from_one.append(t)
        # if(emitss[ind] == mmax):
        #     extracted_from_one.append(t)
        if(emitss[ind]> -1000.0):
            extracted_from_one.append(t)
    doc_res[retr['doc_id']] = float(emition)
    all_emits = sorted(all_emits, key=lambda x: x[1], reverse=True)
    return doc_res, extracted_from_one, all_emits

def get_norm_doc_scores(the_doc_scores):
    ks = list(the_doc_scores.keys())
    vs = [the_doc_scores[k] for k in ks]
    vs = softmax(vs)
    norm_doc_scores = {}
    for i in range(len(ks)):
        norm_doc_scores[ks[i]] = vs[i]
    return norm_doc_scores

def select_snippets_v1(extracted_snippets):
    '''
    :param extracted_snippets:
    :param doc_res:
    :return: returns the best 10 snippets of all docs (0..n from each doc)
    '''
    sorted_snips = sorted(extracted_snippets, key=lambda x: x[1], reverse=True)
    return sorted_snips[:10]

def select_snippets_v2(extracted_snippets):
    '''
    :param extracted_snippets:
    :param doc_res:
    :return: returns the best snippet of each doc  (1 from each doc)
    '''
    # is_relevant, the_sent_score, ncbi_pmid_link, the_actual_sent_text
    ret = {}
    for es in extracted_snippets:
        if (es[2] in ret):
            if (es[1] > ret[es[2]][1]):
                ret[es[2]] = es
        else:
            ret[es[2]] = es
    sorted_snips = sorted(ret.values(), key=lambda x: x[1], reverse=True)
    return sorted_snips[:10]
def select_snippets_v20(extracted_snippets):
    '''
    :param extracted_snippets:
    :param doc_res:
    :return: returns the best snippet of each doc  (1 from each doc)
    '''
    # is_relevant, the_sent_score, ncbi_pmid_link, the_actual_sent_text
    ret = {}
    for es in extracted_snippets:
        if (es[2] in ret):
            if (es[1] > ret[es[2]][1]):
                ret[es[2]] = es
        else:
            ret[es[2]] = es
    sorted_snips = sorted(ret.values(), key=lambda x: x[1], reverse=True)
    return sorted_snips[:20]

def select_snippets_v3(extracted_snippets, the_doc_scores):
    '''
    :param      extracted_snippets:
    :param      doc_res:
    :return:    returns the top 10 snippets across all documents (0..n from each doc)
    '''
    norm_doc_scores = get_norm_doc_scores(the_doc_scores)
    # is_relevant, the_sent_score, ncbi_pmid_link, the_actual_sent_text
    extracted_snippets = [tt for tt in extracted_snippets if (tt[2] in norm_doc_scores)]
    sorted_snips = sorted(extracted_snippets, key=lambda x: x[1] * norm_doc_scores[x[2]], reverse=True)
    return sorted_snips[:10]


def print_the_results(prefix, all_bioasq_gold_data, all_bioasq_subm_data, data_for_revision):
    ###########################################################
    bioasq_snip_res = get_bioasq_res(prefix, all_bioasq_gold_data, all_bioasq_subm_data, data_for_revision)
    pprint(bioasq_snip_res)
    print('{} MAP documents: {}'.format(prefix, bioasq_snip_res['MAP documents']))
    print('{} F1 snippets: {}'.format(prefix, bioasq_snip_res['F1 snippets']))
    print('{} MAP snippets: {}'.format(prefix, bioasq_snip_res['MAP snippets']))
    print('{} GMAP snippets: {}'.format(prefix, bioasq_snip_res['GMAP snippets']))



class Sent_Posit_Drmm_Modeler(nn.Module):
    def __init__(self, embedding_dim=30, k_for_maxpool=5, sentence_out_method='MLP', k_sent_maxpool=1):
        super(Sent_Posit_Drmm_Modeler, self).__init__()
        self.k = k_for_maxpool
        self.k_sent_maxpool = k_sent_maxpool
        self.doc_add_feats = 11
        self.sent_add_feats = 10
        #
        self.embedding_dim = embedding_dim
        self.sentence_out_method = sentence_out_method
        # to create q weights
        self.init_context_module()
        self.init_question_weight_module()
        self.init_mlps_for_pooled_attention()
        self.init_sent_output_layer()
        self.init_doc_out_layer()
        # doc loss func
        self.margin_loss = nn.MarginRankingLoss(margin=1.0)
        if (use_cuda):
            self.margin_loss = self.margin_loss.cuda()
    ###########################################################
    def init_mesh_module(self):
        self.mesh_h0 = autograd.Variable(torch.randn(1, 1, self.embedding_dim))
        self.mesh_gru = nn.GRU(self.embedding_dim, self.embedding_dim)
        if (use_cuda):
            self.mesh_h0 = self.mesh_h0.cuda()
            self.mesh_gru = self.mesh_gru.cuda()
    ###########################################################
    def init_context_module(self):
        self.trigram_conv_1 = nn.Conv1d(self.embedding_dim, self.embedding_dim, 3, padding=2, bias=True)
        self.trigram_conv_activation_1 = torch.nn.LeakyReLU(negative_slope=0.1)
        self.trigram_conv_2 = nn.Conv1d(self.embedding_dim, self.embedding_dim, 3, padding=2, bias=True)
        self.trigram_conv_activation_2 = torch.nn.LeakyReLU(negative_slope=0.1)
        if (use_cuda):
            self.trigram_conv_1 = self.trigram_conv_1.cuda()
            self.trigram_conv_2 = self.trigram_conv_2.cuda()
            self.trigram_conv_activation_1 = self.trigram_conv_activation_1.cuda()
            self.trigram_conv_activation_2 = self.trigram_conv_activation_2.cuda()
    ###########################################################
    def init_question_weight_module(self):
        self.q_weights_mlp = nn.Linear(self.embedding_dim + 1, 1, bias=True)
        if (use_cuda):
            self.q_weights_mlp = self.q_weights_mlp.cuda()
    ###########################################################
    def init_mlps_for_pooled_attention(self):
        self.linear_per_q1 = nn.Linear(3 * 3, 8, bias=True)
        self.my_relu1 = torch.nn.LeakyReLU(negative_slope=0.1)
        self.linear_per_q2 = nn.Linear(8, 1, bias=True)
        if (use_cuda):
            self.linear_per_q1 = self.linear_per_q1.cuda()
            self.linear_per_q2 = self.linear_per_q2.cuda()
            self.my_relu1 = self.my_relu1.cuda()
    ###########################################################
    def init_sent_output_layer(self):
        if (self.sentence_out_method == 'MLP'):
            self.sent_out_layer_1 = nn.Linear(self.sent_add_feats + 1, 8, bias=False)
            self.sent_out_activ_1 = torch.nn.LeakyReLU(negative_slope=0.1)
            self.sent_out_layer_2 = nn.Linear(8, 1, bias=False)
            if (use_cuda):
                self.sent_out_layer_1 = self.sent_out_layer_1.cuda()
                self.sent_out_activ_1 = self.sent_out_activ_1.cuda()
                self.sent_out_layer_2 = self.sent_out_layer_2.cuda()
        else:
            self.sent_res_h0 = autograd.Variable(torch.randn(2, 1, 5))
            self.sent_res_bigru = nn.GRU(input_size=self.sent_add_feats + 1, hidden_size=5, bidirectional=True,
                                         batch_first=False)
            self.sent_res_mlp = nn.Linear(10, 1, bias=False)
            if (use_cuda):
                self.sent_res_h0 = self.sent_res_h0.cuda()
                self.sent_res_bigru = self.sent_res_bigru.cuda()
                self.sent_res_mlp = self.sent_res_mlp.cuda()
    ###########################################################
    def init_doc_out_layer(self):
        self.final_layer_1 = nn.Linear(self.doc_add_feats + self.k_sent_maxpool, 8, bias=True)
        self.final_activ_1 = torch.nn.LeakyReLU(negative_slope=0.1)
        self.final_layer_2 = nn.Linear(8, 1, bias=True)
        self.oo_layer = nn.Linear(2, 1, bias=True)
        if (use_cuda):
            self.final_layer_1 = self.final_layer_1.cuda()
            self.final_activ_1 = self.final_activ_1.cuda()
            self.final_layer_2 = self.final_layer_2.cuda()
            self.oo_layer = self.oo_layer.cuda()
    ###########################################################
    def my_hinge_loss(self, positives, negatives, margin=1.0):
        delta = negatives - positives
        loss_q_pos = torch.sum(F.relu(margin + delta), dim=-1)
        return loss_q_pos
    ###########################################################
    def apply_context_gru(self, the_input, h0):
        output, hn = self.context_gru(the_input.unsqueeze(1), h0)
        output = self.context_gru_activation(output)
        out_forward = output[:, 0, :self.embedding_dim]
        out_backward = output[:, 0, self.embedding_dim:]
        output = out_forward + out_backward
        res = output + the_input
        return res, hn
    ###########################################################
    def apply_context_convolution(self, the_input, the_filters, activation):
        conv_res = the_filters(the_input.transpose(0, 1).unsqueeze(0))
        if (activation is not None):
            conv_res = activation(conv_res)
        pad = the_filters.padding[0]
        ind_from = int(np.floor(pad / 2.0))
        ind_to = ind_from + the_input.size(0)
        conv_res = conv_res[:, :, ind_from:ind_to]
        conv_res = conv_res.transpose(1, 2)
        conv_res = conv_res + the_input
        return conv_res.squeeze(0)
    ###########################################################
    def my_cosine_sim(self, A, B):
        A = A.unsqueeze(0)
        B = B.unsqueeze(0)
        A_mag = torch.norm(A, 2, dim=2)
        B_mag = torch.norm(B, 2, dim=2)
        num = torch.bmm(A, B.transpose(-1, -2))
        den = torch.bmm(A_mag.unsqueeze(-1), B_mag.unsqueeze(-1).transpose(-1, -2))
        dist_mat = num / den
        return dist_mat
    ###########################################################
    def pooling_method(self, sim_matrix):
        sorted_res = torch.sort(sim_matrix, -1)[0]  # sort the input minimum to maximum
        k_max_pooled = sorted_res[:, -self.k:]  # select the last k of each instance in our data
        average_k_max_pooled = k_max_pooled.sum(-1) / float(self.k)  # average these k values
        the_maximum = k_max_pooled[:, -1]  # select the maximum value of each instance
        the_average_over_all = sorted_res.sum(-1) / float(
            sim_matrix.size(1))  # add average of all elements as long sentences might have more matches
        the_concatenation = torch.stack([the_maximum, average_k_max_pooled, the_average_over_all],
                                        dim=-1)  # concatenate maximum value and average of k-max values
        return the_concatenation  # return the concatenation
    ###########################################################
    def get_output(self, input_list, weights):
        temp = torch.cat(input_list, -1)
        lo = self.linear_per_q1(temp)
        lo = self.my_relu1(lo)
        lo = self.linear_per_q2(lo)
        lo = lo.squeeze(-1)
        lo = lo * weights
        sr = lo.sum(-1) / lo.size(-1)
        return sr
    ###########################################################
    def apply_sent_res_bigru(self, the_input):
        output, hn = self.sent_res_bigru(the_input.unsqueeze(1), self.sent_res_h0)
        output = self.sent_res_mlp(output)
        return output.squeeze(-1).squeeze(-1)
    ###########################################################
    def do_for_one_doc_cnn(self, doc_sents_embeds, oh_sims, sents_af, question_embeds, q_conv_res_trigram, q_weights,
                           k2):
        res = []
        for i in range(len(doc_sents_embeds)):
            sim_oh = autograd.Variable(torch.FloatTensor(oh_sims[i]), requires_grad=False)
            sent_embeds = doc_sents_embeds[i]
            gaf = autograd.Variable(torch.FloatTensor(sents_af[i]), requires_grad=False)
            if (use_cuda):
                sim_oh = sim_oh.cuda()
                gaf = gaf.cuda()
            #
            conv_res            = self.apply_context_convolution(sent_embeds, self.trigram_conv_1, self.trigram_conv_activation_1)
            conv_res            = self.apply_context_convolution(conv_res, self.trigram_conv_2, self.trigram_conv_activation_2)
            #
            sim_insens          = self.my_cosine_sim(question_embeds, sent_embeds).squeeze(0)
            sim_sens            = self.my_cosine_sim(q_conv_res_trigram, conv_res).squeeze(0)
            #
            insensitive_pooled  = self.pooling_method(sim_insens)
            sensitive_pooled    = self.pooling_method(sim_sens)
            oh_pooled           = self.pooling_method(sim_oh)
            #
            sent_emit           = self.get_output([oh_pooled, insensitive_pooled, sensitive_pooled], q_weights)
            sent_add_feats      = torch.cat([gaf, sent_emit.unsqueeze(-1)])
            res.append(sent_add_feats)
        res = torch.stack(res)
        if (self.sentence_out_method == 'MLP'):
            res = self.sent_out_layer_1(res)
            res = self.sent_out_activ_1(res)
            res = self.sent_out_layer_2(res).squeeze(-1)
        else:
            res = self.apply_sent_res_bigru(res)
        # ret = self.get_max(res).unsqueeze(0)
        ret = self.get_kmax(res, k2)
        return ret, res
    ###########################################################
    def do_for_one_doc_bigru(self, doc_sents_embeds, sents_af, question_embeds, q_conv_res_trigram, q_weights, k2):
        res = []
        hn = self.context_h0
        for i in range(len(doc_sents_embeds)):
            sent_embeds = autograd.Variable(torch.FloatTensor(doc_sents_embeds[i]), requires_grad=False)
            gaf = autograd.Variable(torch.FloatTensor(sents_af[i]), requires_grad=False)
            if (use_cuda):
                sent_embeds = sent_embeds.cuda()
                gaf = gaf.cuda()
            conv_res, hn = self.apply_context_gru(sent_embeds, hn)
            #
            sim_insens = self.my_cosine_sim(question_embeds, sent_embeds).squeeze(0)
            sim_oh = (sim_insens > (1 - (1e-3))).float()
            sim_sens = self.my_cosine_sim(q_conv_res_trigram, conv_res).squeeze(0)
            #
            insensitive_pooled = self.pooling_method(sim_insens)
            sensitive_pooled = self.pooling_method(sim_sens)
            oh_pooled = self.pooling_method(sim_oh)
            #
            sent_emit = self.get_output([oh_pooled, insensitive_pooled, sensitive_pooled], q_weights)
            sent_add_feats = torch.cat([gaf, sent_emit.unsqueeze(-1)])
            res.append(sent_add_feats)
        res = torch.stack(res)
        if (self.sentence_out_method == 'MLP'):
            res = self.sent_out_layer_1(res)
            res = self.sent_out_activ_1(res)
            res = self.sent_out_layer_2(res).squeeze(-1)
        else:
            res = self.apply_sent_res_bigru(res)
        # ret = self.get_max(res).unsqueeze(0)
        ret = self.get_kmax(res, k2)
        res = torch.sigmoid(res)
        return ret, res
    ###########################################################
    def get_max(self, res):
        return torch.max(res)
    ###########################################################
    def get_kmax(self, res, k):
        res = torch.sort(res, 0)[0]
        res = res[-k:].squeeze(-1)
        if (len(res.size()) == 0):
            res = res.unsqueeze(0)
        if (res.size()[0] < k):
            to_concat = torch.zeros(k - res.size()[0])
            if (use_cuda):
                to_concat = to_concat.cuda()
            res = torch.cat([res, to_concat], -1)
        return res
    ###########################################################
    def get_max_and_average_of_k_max(self, res, k):
        k_max_pooled = self.get_kmax(res, k)
        average_k_max_pooled = k_max_pooled.sum() / float(k)
        the_maximum = k_max_pooled[-1]
        the_concatenation = torch.cat([the_maximum, average_k_max_pooled.unsqueeze(0)])
        return the_concatenation
    ###########################################################
    def get_average(self, res):
        res = torch.sum(res) / float(res.size()[0])
        return res
    ###########################################################
    def get_maxmin_max(self, res):
        res = self.min_max_norm(res)
        res = torch.max(res)
        return res
    ###########################################################
    def apply_mesh_gru(self, mesh_embeds):
        mesh_embeds = autograd.Variable(torch.FloatTensor(mesh_embeds), requires_grad=False)
        if (use_cuda):
            mesh_embeds = mesh_embeds.cuda()
        output, hn = self.mesh_gru(mesh_embeds.unsqueeze(1), self.mesh_h0)
        return output[-1, 0, :]
    ###########################################################
    def get_mesh_rep(self, meshes_embeds, q_context):
        meshes_embeds = [self.apply_mesh_gru(mesh_embeds) for mesh_embeds in meshes_embeds]
        meshes_embeds = torch.stack(meshes_embeds)
        sim_matrix = self.my_cosine_sim(meshes_embeds, q_context).squeeze(0)
        max_sim = torch.sort(sim_matrix, -1)[0][:, -1]
        output = torch.mm(max_sim.unsqueeze(0), meshes_embeds)[0]
        return output
    ###########################################################
    def emit_one(self, doc1_sents_embeds, doc1_oh_sim, question_embeds, q_idfs, sents_gaf, doc_gaf):
        q_idfs          = autograd.Variable(torch.FloatTensor(q_idfs), requires_grad=False)
        doc_gaf         = autograd.Variable(torch.FloatTensor(doc_gaf), requires_grad=False)
        if (use_cuda):
            q_idfs = q_idfs.cuda()
            doc_gaf = doc_gaf.cuda()
        #
        q_context = self.apply_context_convolution(question_embeds, self.trigram_conv_1, self.trigram_conv_activation_1)
        q_context = self.apply_context_convolution(q_context, self.trigram_conv_2, self.trigram_conv_activation_2)
        #
        q_weights = torch.cat([q_context, q_idfs], -1)
        q_weights = self.q_weights_mlp(q_weights).squeeze(-1)
        q_weights = F.softmax(q_weights, dim=-1)
        #
        good_out, gs_emits = self.do_for_one_doc_cnn(
            doc1_sents_embeds,
            doc1_oh_sim,
            sents_gaf,
            question_embeds,
            q_context,
            q_weights,
            self.k_sent_maxpool
        )
        #
        good_out_pp = torch.cat([good_out, doc_gaf], -1)
        #
        final_good_output = self.final_layer_1(good_out_pp)
        final_good_output = self.final_activ_1(final_good_output)
        final_good_output = self.final_layer_2(final_good_output)
        #
        gs_emits = gs_emits.unsqueeze(-1)
        gs_emits = torch.cat([gs_emits, final_good_output.unsqueeze(-1).expand_as(gs_emits)], -1)
        gs_emits = self.oo_layer(gs_emits).squeeze(-1)
        gs_emits = torch.sigmoid(gs_emits)
        #
        return final_good_output, gs_emits
    ###########################################################
    def forward(self, doc1_sents_embeds, doc2_sents_embeds, doc1_oh_sim, doc2_oh_sim,
                question_embeds, q_idfs, sents_gaf, sents_baf, doc_gaf, doc_baf):
        q_idfs = autograd.Variable(torch.FloatTensor(q_idfs), requires_grad=False)
        doc_gaf = autograd.Variable(torch.FloatTensor(doc_gaf), requires_grad=False)
        doc_baf = autograd.Variable(torch.FloatTensor(doc_baf), requires_grad=False)
        if (use_cuda):
            q_idfs = q_idfs.cuda()
            doc_gaf = doc_gaf.cuda()
            doc_baf = doc_baf.cuda()
        #
        q_context = self.apply_context_convolution(question_embeds, self.trigram_conv_1, self.trigram_conv_activation_1)
        q_context = self.apply_context_convolution(q_context, self.trigram_conv_2, self.trigram_conv_activation_2)
        #
        q_weights = torch.cat([q_context, q_idfs], -1)
        q_weights = self.q_weights_mlp(q_weights).squeeze(-1)
        q_weights = F.softmax(q_weights, dim=-1)
        #
        good_out, gs_emits = self.do_for_one_doc_cnn(
            doc1_sents_embeds,
            doc1_oh_sim,
            sents_gaf,
            question_embeds,
            q_context,
            q_weights,
            self.k_sent_maxpool
        )
        bad_out, bs_emits = self.do_for_one_doc_cnn(
            doc2_sents_embeds,
            doc2_oh_sim,
            sents_baf,
            question_embeds,
            q_context,
            q_weights,
            self.k_sent_maxpool
        )
        #
        good_out_pp = torch.cat([good_out, doc_gaf], -1)
        bad_out_pp = torch.cat([bad_out, doc_baf], -1)
        #
        final_good_output = self.final_layer_1(good_out_pp)
        final_good_output = self.final_activ_1(final_good_output)
        final_good_output = self.final_layer_2(final_good_output)
        #
        gs_emits = gs_emits.unsqueeze(-1)
        gs_emits = torch.cat([gs_emits, final_good_output.unsqueeze(-1).expand_as(gs_emits)], -1)
        gs_emits = self.oo_layer(gs_emits).squeeze(-1)
        gs_emits = torch.sigmoid(gs_emits)
        #
        final_bad_output = self.final_layer_1(bad_out_pp)
        final_bad_output = self.final_activ_1(final_bad_output)
        final_bad_output = self.final_layer_2(final_bad_output)
        #
        bs_emits = bs_emits.unsqueeze(-1)
        bs_emits = torch.cat([bs_emits, final_good_output.unsqueeze(-1).expand_as(bs_emits)], -1)
        bs_emits = self.oo_layer(bs_emits).squeeze(-1)
        bs_emits = torch.sigmoid(bs_emits)
        #
        loss1 = self.my_hinge_loss(final_good_output, final_bad_output)
        return loss1, final_good_output, final_bad_output, gs_emits, bs_emits

def load_model_from_checkpoint(resume_from, resume_from_bert):
    global start_epoch, optimizer
    if os.path.isfile(resume_from):
        print("=> loading checkpoint '{}'".format(resume_from))
        checkpoint = torch.load(resume_from, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})".format(resume_from, checkpoint['epoch']))
    if os.path.isfile(resume_from_bert):
        print("=> loading checkpoint '{}'".format(resume_from_bert))
        checkpoint = torch.load(resume_from_bert, map_location=lambda storage, loc: storage)
        bert_model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})".format(resume_from_bert, checkpoint['epoch']))
        
def embed_the_sent(sent):
    eval_examples   = [InputExample(guid='example_dato_1', text_a=sent, text_b=None, label='1')]
    eval_features   = convert_examples_to_features(eval_examples, max_seq_length, bert_tokenizer)
    eval_feat       = eval_features[0]
    input_ids       = torch.tensor([eval_feat.input_ids], dtype=torch.long).to(device)
    input_mask      = torch.tensor([eval_feat.input_mask], dtype=torch.long).to(device)
    segment_ids     = torch.tensor([eval_feat.segment_ids], dtype=torch.long).to(device)
    tokens          = eval_feat.tokens
    with torch.no_grad():
        token_embeds, pooled_output = bert_model.bert(input_ids, segment_ids, input_mask, output_all_encoded_layers=False)
        tok_inds                    = [i for i in range(len(tokens)) if(not tokens[i].startswith('##'))]
        token_embeds                = token_embeds.squeeze(0)
        embs                        = token_embeds[tok_inds,:]
    fixed_tokens = fix_bert_tokens(tokens)
    return fixed_tokens, embs
def do_for_some_retrieved(docs, dato, retr_docs, data_for_revision, ret_data, use_sent_tokenizer):
    emitions                    = {
        'body': dato['query_text'],
        'id': dato['query_id'],
        'documents': []
        
    }
    #
    quest_text                  = dato['query_text']
    quest_text          = ' '.join(bioclean(quest_text.replace('\ufeff', ' ')))
    quest_tokens, quest_embeds  = embed_the_sent(quest_text)
    q_idfs                      = np.array([[idf_val(qw, idf, max_idf)] for qw in quest_tokens], 'float')
    
    doc_res, extracted_snippets         = {}, []
    for retr in retr_docs:
        datum                   = prep_data(quest_text, docs[retr['doc_id']], retr['norm_bm25_score'], [], idf, max_idf, quest_tokens)
        doc_emit_, gs_emits_ = model.emit_one(
            doc1_sents_embeds   = datum['sents_embeds'],
            doc1_oh_sim         = datum['oh_sims'],
            question_embeds     = quest_embeds,
            q_idfs              = q_idfs,
            sents_gaf           = datum['sents_escores'],
            doc_gaf             = datum['doc_af']
        )
        doc_res, extracted_from_one, all_emits = do_for_one_retrieved(
            doc_emit_, gs_emits_, datum['held_out_sents'], retr, doc_res,[]
        )# is_relevant, the_sent_score, ncbi_pmid_link, the_actual_sent_text
        extracted_snippets.extend(extracted_from_one)
        #
        total_relevant = sum([1 for em in all_emits if (em[0] == True)])
        if (dato['query_id'] not in data_for_revision):
            data_for_revision[dato['query_id']] = {'query_text': dato['query_text'],
                                                   'snippets': {retr['doc_id']: all_emits}}
        else:
            data_for_revision[dato['query_id']]['snippets'][retr['doc_id']] = all_emits
    #
    doc_res                                 = sorted([item for item in doc_res.items() if(item[1]>-1000.0)], key = lambda x: x[1], reverse = True)
    the_doc_scores                          = dict([("http://www.ncbi.nlm.nih.gov/pubmed/{}".format(pm[0]), pm[1]) for pm in doc_res[:doclimit]])
    doc_res                                 = ["http://www.ncbi.nlm.nih.gov/pubmed/{}".format(pm[0]) for pm in doc_res]
    emitions['documents']                   = doc_res[:100]
    ret_data['questions'].append(emitions)
    #
    extracted_snippets                      = [tt for tt in extracted_snippets if (tt[2] in doc_res[:doclimit])]
    #extracted_snippets_v3                   = select_snippets_v3(extracted_snippets, the_doc_scores)
    extracted_snippets_v3                   = select_snippets_v20(extracted_snippets)
    #
    snips_res_v3                            = prep_extracted_snippets(extracted_snippets_v3, docs, dato['query_id'], doc_res[:doclimit], dato['query_text'])
    #
    snips_res = {'v3' : snips_res_v3}
    return data_for_revision, ret_data, snips_res

def get_one(data, docs):
    model.eval()
    ##########################################
    ret_data                        = {'questions': []}
    all_bioasq_subm_data_v3         = {"questions": []}
    data_for_revision               = {}
    ##########################################
    for dato in tqdm(data['queries']):
        data_for_revision, ret_data, snips_res = do_for_some_retrieved(docs, dato, dato['retrieved_documents'], data_for_revision, ret_data, True)
        all_bioasq_subm_data_v3['questions'].append(snips_res['v3'])
    ##########################################
    return all_bioasq_subm_data_v3 # data_for_revision #all_bioasq_subm_data_v3

def check_data(data):
    if(len(data)!=2):
        return None
    if('id' not in data):
        return None
    if('question' not in data):
        return None
    return data

def anazitisi(idd, question):
    data = {
        "questions": [
            {
                "body"      : question,
                "id"        : idd,
                "documents" : []
            }
        ]
    }
    
    f_out  = 'interim_resources/sample_out_{}.json'.format(idd)
    preprocess(data,f_out)
    test_data,test_docs=generate_test_data(data, f_out, keep_up_to_year)
    os.remove(f_out)
    
    GetWords(test_data, test_docs, words)
    global idf,max_idf
    print('loading idfs')
    idf, max_idf = load_idfs(idf_pickle_path, words)
    ret = {'results': get_one(test_data, test_docs)}
    return ret

def preprocess_input(data):
    words=data
    words=words.lower().split()
    for i,word in enumerate(words):
       words[i]=spell.correction(word)
    data=" ".join(words)+"?"
    if(data.lower().find("hand eczema")!=-1 or
        data.lower().find("contact eczema")!=-1 or
         data.lower().find("contact dermatitis")!=-1):
        return data
    elif(data.lower().find("atopic eczema")!=-1):
        data=data.lower().replace("atopic eczema","atopic dermatitis")
        return data
    elif(data.lower().find("eczema")!=-1):
        data=data.lower().replace("eczema","atopic dermatitis")
        data=data.lower().replace("atopic atopic","atopic")
        return data
    elif(data.lower().find("eczema")==-1 and data.lower().find("atopic dermatitis")):
        data=data[:-1]
        return data+" in atopic dermatitis ?"
    
@app.route('/search', methods=['GET', 'POST'])
def data_searching():
    try:
        app.logger.debug("JSON received...")
        data=request.get_json(force=True,silent=True)
        app.logger.debug(data)
        if(request.method=="POST"):
            print(request.get_json(force=True,silent=True))
            if data:
                mydata = data
                pprint(mydata)
                mydata = check_data(mydata)
                mydata['question']=preprocess_input(mydata['question'])
                if(mydata is None):
                    ret = {'success': 0, 'message': 'False request'}
                    app.logger.debug(ret)
                    return jsonify(ret)
                else:
                    ret = anazitisi(mydata['id'], mydata['question'])
                    ret['request'] = mydata
                    return jsonify(ret)
            else:
                ret = {'success': 0, 'message': 'request should be json formated'}
                app.logger.debug(ret)
                return jsonify(ret)
           
    except Exception as e:
        app.logger.debug(str(e))
        traceback.print_exc()
        ret = {'success': 0, 'message': str(e)+'\n'+traceback.format_exc()}
        app.logger.debug(ret)
        return jsonify(ret)

min_doc_score               = -1000.
min_sent_score              = -1000.
emit_only_abstract_sents    = False
doclimit=20
###########################################################
use_cuda                    = torch.cuda.is_available()
###########################################################
w2v_bin_path                = r'D:\aueb-bioasq7-master\Data\PretrainedWeightsAndVectors\pubmed2018_w2v_30D.bin'
idf_pickle_path             = r'D:\aueb-bioasq7-master\Data\PretrainedWeightsAndVectors\idf.pkl'
###########################################################
resume_from         = r'D:\aueb-bioasq7-master\Data\PretrainedWeightsAndVectors\bioasq7_bert_jpdrmm_2L_0p01_run_0\best_checkpoint.pth.tar'
resume_from_bert    = r'D:\aueb-bioasq7-master\Data\PretrainedWeightsAndVectors\bioasq7_bert_jpdrmm_2L_0p01_run_0\best_bert_checkpoint.pth.tar'
cache_dir           = r'D:\aueb-bioasq7-master\Data\PretrainedWeightsAndVectors\bert_cache\\'
###########################################################
f_in1                       = 'interim_resources/sample.json'
f_out                       = 'interim_resources/sample_out.json'
f_in2                       = 'interim_resources/bioasq_bm25_top100.sample.pkl'
f_in3                       = 'interim_resources/bioasq_bm25_docset_top100.sample.pkl'

###########################################################
avgdl, mean, deviation      = 21.1907, 0.6275, 1.2210
print(avgdl, mean, deviation)
###########################################################
k_for_maxpool, k_sent_maxpool, embedding_dim = 5, 5, 768
###########################################################
max_seq_length      = 50
my_seed     = 1
random.seed(my_seed)
torch.manual_seed(my_seed)
keep_up_to_year             = 2018
###########################################################
print('Compiling model...')
max_seq_length      = 50
device              = torch.device("cuda") if(use_cuda) else torch.device("cpu")
bert_model          = 'bert-base-uncased'
bert_tokenizer      = BertTokenizer.from_pretrained(bert_model, do_lower_case=True, cache_dir=cache_dir)
bert_model          = BertForSequenceClassification.from_pretrained(bert_model, cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(-1), num_labels=2)
model               = Sent_Posit_Drmm_Modeler(embedding_dim=embedding_dim, k_for_maxpool=k_for_maxpool)
###########################################################
load_model_from_checkpoint(resume_from, resume_from_bert)
print_params(model, bert_model)
bert_model.to(device)
model.to(device)
###########################################################


###########################################################

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9251, debug=True, threaded=True)


'''
rm -rf \
/home/dpappas/sample_bm25_retrieval.txt \
/home/dpappas/sample.json \
/home/dpappas/sample_galago_queries.json \
/home/dpappas/bioasq_bm25_docset_top100.all.pkl \
/home/dpappas/bioasq_bm25_top100.all.pkl
'''
