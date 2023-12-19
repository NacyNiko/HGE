import torch
from torch import optim
from models import HGE_TNTComplEx, HGE_TComplEx
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

parser.add_argument('--checkpoint', help='checkpoint path')
parser.add_argument('--device', default='cuda:0')


def load_model(path):
    dataset = TemporalDataset(args.dataset)
    sizes = dataset.get_shape()
    model = {
        'HGE_TNTComplEx': HGE_TNTComplEx(sizes, args.rank, no_time_emb=False, num_form=args.num_form,
                                   use_attention=args.attention, use_reverse=False),
        'HGE_ComplEx': HGE_TComplEx(sizes, args.rank, no_time_emb=False, num_form=args.num_form,
                               use_attention=args.attention),
        'HGE_TLT_KGE': HGE_TLT_KGE(sizes, args.rank, cycle=args.cycle, num_form=args.num_form, use_reverse=False, attention=0)
    }[args.model]
    model = model.to(args.device)
    checkpoint = torch.load(path+'/best.pth', map_location=args.device)
    model.load_state_dict(checkpoint['param'], strict=False)
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


if __name__ == '__main__':
    args = parser.parse_args()
    args.cycle=1
    checkpoint = torch.load(args.checkpoint + '/best.pth', map_location=args.device)
    if 'args' in checkpoint:
        for attr in vars(checkpoint['args']):
            setattr(args, attr, getattr(checkpoint['args'], attr))
    else:
        hyper_params = args.checkpoint.split('_')
        if hyper_params[2] == 'TLT':
            args.dataset, m1, m2, m3, args.rank, args.learning_rate, args.emb_reg, args.time_reg, args.cycle, _ = hyper_params
            args.model = '_'.join([m1, m2, m3])
        else:
            if len(hyper_params) == 9:
                args.dataset, m1, m2, args.rank, args.learning_rate, args.emb_reg, args.time_reg, args.attention, args.num_form = hyper_params
                args.model = '_'.join([m1, m2])
            else:
                args.dataset, m1, m2, args.rank, args.learning_rate, args.emb_reg, args.time_reg, args.attention, num_form1, num_form2 = hyper_params
                args.model = '_'.join([m1, m2])
                args.num_form = num_form1 + num_form2
            args.cycle = 1

    args.rank = int(args.rank)
    args.learning_rate = float(args.learning_rate)
    args.emb_reg = float(args.emb_reg)
    args.attention = int(args.attention)
    model = load_model(args.checkpoint)
    dataset = TemporalDataset(args.dataset)
    test_result = dataset.eval(model=model, split='test')
    test = avg_both(*test_result)
    print("test_MRR: ", test['MRR'])
    print("test_Hits: ", test['hits@[1,3,10]'])



