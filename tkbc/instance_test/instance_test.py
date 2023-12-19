import torch
from torch import optim
from models import ComplEx, TComplEx, TNTComplEx, HGE_TNTComplEx, HGE_TComplEx
from models_TLTs import HGE_TLT_KGE
import argparse
from datasets import TemporalDataset
from regularizers import N3, Lambda3
from optimizers import TKBCOptimizer, IKBCOptimizer
import os
from typing import Dict
import json
import numpy as np
parser = argparse.ArgumentParser(
    description="Temporal ComplEx"
)
parser.add_argument(
    '--dataset', type=str,
    help="Dataset name"
)
models = [
    'ComplEx', 'TComplEx', 'TNTComplEx', 'HGE_TNTComplEx', 'HGE_TComplEx',
]
parser.add_argument(
    '--model', choices=models,
    help="Model in {}".format(models)
)
parser.add_argument(
    '--max_epochs', default=200, type=int,
    help="Number of epochs."
)
parser.add_argument(
    '--valid_freq', default=5, type=int,
    help="Number of epochs between each valid."
)
parser.add_argument(
    '--rank', default=100, type=int,
    help="Factorization rank."
)
parser.add_argument(
    '--batch_size', default=1000, type=int,
    help="Batch size."
)
parser.add_argument(
    '--learning_rate', default=1e-1, type=float,
    help="Learning rate"
)
parser.add_argument(
    '--emb_reg', default=0., type=float,
    help="Embedding regularizer strength"
)
parser.add_argument(
    '--time_reg', default=0., type=float,
    help="Timestamp regularizer strength"
)
parser.add_argument(
    '--no_time_emb', default=False, action="store_true",
    help="Use a specific embedding for non temporal relations"
)

parser.add_argument(
    '--device', default="cpu", help="CPU or cuda device"
)
parser.add_argument(
    '--attention', default=0, type=int,
    help="attention form."
)
parser.add_argument(
    '--num_form', default="split", help="which num form will the model take"
)

parser.add_argument(
    "--hyp_search", action="store_true", help="hyperparameter_search"
)

parser.add_argument(
    "--hrh_filter", action="store_true", help="hrh_filter"
)

parser.add_argument(
    "--kg_setting", action="store_true", help="hrh_filter"
)

parser.add_argument(
    "--rhsODE", action="store_true", help="hrh_filter"
)

parser.add_argument(
    "--use_reverse", action="store_true", help="hrh_filter"
)

parser.add_argument(
    "--print_attention", action="store_true", help="hrh_filter"
)

parser.add_argument(
    '--hidrank', default=10, type=int,
    help="attention form."
)

parser.add_argument(
    '--thidrank', default=10, type=int,
    help="attention form."
)


args = parser.parse_args()

def load_model(path,d):
    #dataset = TemporalDataset(args.dataset)
    sizes = dataset.get_shape()
    model = {
        'ComplEx': ComplEx(sizes, args.rank),
        'TComplEx': TComplEx(sizes, args.rank, no_time_emb=args.no_time_emb),
        'TNTComplEx': TNTComplEx(sizes, args.rank, no_time_emb=args.no_time_emb),
        'HGE_TNTComplEx': HGE_TNTComplEx(sizes, args.rank, no_time_emb=args.no_time_emb,num_form=args.num_form,use_attention=args.attention),
        'HGE_TComplEx': HGE_TComplEx(sizes, args.rank, no_time_emb=args.no_time_emb,num_form=args.num_form,use_attention=args.attention),
        'HGE_TLT_KGE': HGE_TLT_KGE(sizes, args.rank, cycle=args.cycle, num_form=args.num_form, use_reverse=False,
                                   attention=0)
    }[args.model]
    model = model.to(args.device)
    checkpoint = torch.load(path+'/best.pth',map_location='cpu')
    model.load_state_dict(checkpoint['param'],strict=False)
    return model

def avg_both(mrrs: Dict[str, float], hits: Dict[str, torch.FloatTensor]):
    """
    aggregate metrics for missing lhs and rhs
    :param mrrs: d
    :param hits:
    :return:
    """
    m = (mrrs['lhs'] + mrrs['rhs']) / 2.
    h = (hits['lhs'] + hits['rhs']) / 2.
    return {'MRR': m, 'hits@[1,3,10]': h}

def read_file(f_name):
    f=open(f_name,'r')
    data = f.readlines()
    return data

def load_idx(dataset):
    rid = json.load(open((str(dataset)+'_rid.json'),'r'))
    tid = json.load(open((str(dataset)+'_tid.json'),'r'))
    eid = json.load(open((str(dataset) + '_eid.json'), 'r'))
    return rid,tid, eid

def instances_to_id(data, rid, tid, eid):
    examples = []
    for context in data[1:]:
        head, rel, tail, time = context.strip().split('\t')
        try:
            head, tail = eid[head], eid[tail]
            rel = rid[rel]
            time = tid[time]
            examples.append([head,rel,tail,time])
        except:
            print(context)
            continue
    return examples

def run(args,dataset,data,model):
    if dataset.has_intervals():
        test = dataset.eval_instance(model,data, -1)
        print("instance_result: ", test)
    else:
        test=avg_both(*dataset.eval_instance(model,data, -1))
        print("test_MRR: ", test['MRR'])
        print("test_Hits: ", test['hits@[1,3,10]'])
    return test

def write_file(d_name,complex,dual,split,ensemble):
    f = open(d_name+'_flow_hierarchy.txt','r')
    f2 = open(d_name+'_hierarchy_summary.txt','w+')
    rel_hierarchy = f.readlines()
    for i in range(len(rel_hierarchy)):
        hierarchy_score, freq = rel_hierarchy[i].strip().split()
        f2.write(str(hierarchy_score) + '\t' + str(freq) +'\t'+ str(complex[i])+'\t'+str(dual[i])+'\t'+str(split[i])+'\t'+str(ensemble[i])+'\n')
    f2.close()
    f.close()
    return

def hierarchy_test(d_name,dataset,model):
    f = open(d_name+'_flow_hierarchy.txt','r')
    rel_hierarchy = f.readlines()
    test_mrr=[]
    for i in range(len(rel_hierarchy)):
        data = dataset.data['test']
        index = (data[:, 1] == i)
        data = dataset.data['test'][index]
        if len(data)==0:
            test_mrr.append(0)
        else:
            test = run(args,dataset,data,model)
            test_mrr.append(test['MRR'])
    f.close()
    return test_mrr

def hierarchy_test_ins(d_name,dataset,model):
    data = read_file('hierarchy_'+d_name.lower()+'.txt')
    rid, tid, eid = load_idx(d_name)
    data = instances_to_id(data, rid,tid, eid)
    data = np.array(data)
    test = run(args,dataset,data,model)
    return test['MRR']

def symmetry_test(d_name,dataset, model):
    data = read_file('temporal_symmetric_instances_selected_'+d_name.lower()+'.txt')
    rid, tid, eid = load_idx(d_name)
    data = instances_to_id(data, rid,tid, eid)
    data = np.array(data)
    test = run(args,dataset,data,model)
    return test['MRR']

def star_test(d_name,dataset, model):
    data = read_file('missing_instances_'+d_name.lower()+'.txt')
    rid, tid, eid = load_idx(d_name)
    data = instances_to_id(data, rid,tid, eid)
    data = np.array(data)
    test = run(args,dataset,data,model)
    return test['MRR']

d_name='ICEWS14'
test_mrr={}
#for num_form in ['full_rel']:
for num_form in ['full_rel','complex','dual','split']:
    args.num_form = num_form
    #model = load_model('expe_log/ICEWS05-15_HTNTComplEx_1200_0.1_0.003_0.1_1_'+num_form)
    #model = load_model('expe_log/ICEWS14_HTNTComplEx_1200_0.1_0.003_0.03_1_' + num_form)
    dataset = TemporalDataset(d_name)
    model = load_model('/workspace/git/MetaE/expe_log/ICEWS14_HTNTComplEx_1200_0.1_0.003_0.03_1_'+num_form,dataset)
    #test_mrr[num_form] = hierarchy_test(d_name,dataset,model)
    symmetry_mrr = symmetry_test(d_name, dataset, model)
    star_mrr = star_test(d_name, dataset, model)
    hierarchy_mrr = hierarchy_test_ins(d_name,dataset,model)
    print(num_form)
    print(symmetry_mrr)
    print(star_mrr)
    print(hierarchy_mrr)
#write_file(d_name,test_mrr['complex'],test_mrr['dual'],test_mrr['split'],test_mrr['full_rel'])

d_name='ICEWS05-15'
test_mrr={}
#for num_form in ['full_rel']:
args.num_form = num_form
dataset = TemporalDataset(d_name)
model = load_model('/workspace/git/MetaE/expe_log/ICEWS05-15_HTNTComplEx_1200_0.1_0.003_0.1_0_'+num_form,dataset)
symmetry_mrr = symmetry_test(d_name, dataset, model)
star_mrr = star_test(d_name, dataset, model)
hierarchy_mrr = hierarchy_test_ins(d_name,dataset,model)
print(symmetry_mrr)
print(star_mrr)
print(hierarchy_mrr)
write_file(d_name, test_mrr['complex'], test_mrr['dual'], test_mrr['split'], test_mrr['full_rel'])
