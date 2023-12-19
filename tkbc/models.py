# Copyright (c) Facebook, Inc. and its affiliates.

from abc import ABC, abstractmethod
from typing import Tuple, List, Dict

import math
import torch
from torch import nn
import numpy as np


class TKBCModel(nn.Module, ABC):
    @abstractmethod
    def get_rhs(self, chunk_begin: int, chunk_size: int, queries: torch.Tensor):
        pass

    @abstractmethod
    def get_queries(self, queries: torch.Tensor):
        pass

    @abstractmethod
    def score(self, x: torch.Tensor):
        pass

    @abstractmethod
    def forward_over_time(self, x: torch.Tensor):
        pass

    def batch_mul(self, x, y):
        if len(x.shape) != len(y.shape):
            result = torch.einsum('ij,ikj->ik', [x, y])
        else:
            result = torch.einsum('ij,kj->ik', [x, y])
        return result

    def get_ranking(
            self, queries: torch.Tensor,
            filters: Dict[Tuple[int, int, int], List[int]],
            batch_size: int = 1000, chunk_size: int = -1
    ):
        """
        Returns filtered ranking for each queries.
        :param queries: a torch.LongTensor of quadruples (lhs, rel, rhs, timestamp)
        :param filters: filters[(lhs, rel, ts)] gives the elements to filter from ranking
        :param batch_size: maximum number of queries processed at once
        :param chunk_size: maximum number of candidates processed at once
        :return:
        """
        if chunk_size < 0:
            chunk_size = self.sizes[2]
        ranks = torch.ones(len(queries))
        with torch.no_grad():
            c_begin = 0
            while c_begin < self.sizes[2]:
                b_begin = 0
                while b_begin < len(queries):
                    these_queries = queries[b_begin:b_begin + batch_size]
                    rhs = self.get_rhs(c_begin, chunk_size, these_queries)
                    q = self.get_queries(these_queries)

                    # rhs = rhs.t()
                    # scores = q @ rhs
                    scores = self.batch_mul(q,rhs)
                    targets = self.score(these_queries)
                    assert not torch.any(torch.isinf(scores)), "inf scores"
                    assert not torch.any(torch.isnan(scores)), "nan scores"
                    assert not torch.any(torch.isinf(targets)), "inf targets"
                    assert not torch.any(torch.isnan(targets)), "nan targets"

                    # set filtered and true scores to -1e6 to be ignored
                    # take care that scores are chunked
                    for i, query in enumerate(these_queries):
                        filter_out = filters[(query[0].item(), query[1].item(), query[3].item())]
                        filter_out += [queries[b_begin + i, 2].item()]
                        if chunk_size < self.sizes[2]:
                            filter_in_chunk = [
                                int(x - c_begin) for x in filter_out
                                if c_begin <= x < c_begin + chunk_size
                            ]
                            scores[i, torch.LongTensor(filter_in_chunk)] = -1e6
                        else:
                            scores[i, torch.LongTensor(filter_out)] = -1e6
                    ranks[b_begin:b_begin + batch_size] += torch.sum(
                        (scores >= targets).float(), dim=1
                    ).cpu()

                    b_begin += batch_size

                c_begin += chunk_size
        return ranks

    def get_auc(
            self, queries: torch.Tensor, batch_size: int = 1000
    ):
        """
        Returns filtered ranking for each queries.
        :param queries: a torch.LongTensor of quadruples (lhs, rel, rhs, begin, end)
        :param batch_size: maximum number of queries processed at once
        :return:
        """
        all_scores, all_truth = [], []
        all_ts_ids = None
        with torch.no_grad():
            b_begin = 0
            while b_begin < len(queries):
                these_queries = queries[b_begin:b_begin + batch_size]
                scores = self.forward_over_time(these_queries)
                all_scores.append(scores.cpu().numpy())
                if all_ts_ids is None:
                    if torch.cuda.is_available():
                        all_ts_ids = torch.arange(0, scores.shape[1]).cuda()[None, :]
                    else:
                        all_ts_ids = torch.arange(0, scores.shape[1])[None, :]
                assert not torch.any(torch.isinf(scores) + torch.isnan(scores)), "inf or nan scores"
                truth = (all_ts_ids <= these_queries[:, 4][:, None]) * (all_ts_ids >= these_queries[:, 3][:, None])
                all_truth.append(truth.cpu().numpy())
                b_begin += batch_size

        return np.concatenate(all_truth), np.concatenate(all_scores)

    def get_time_ranking(
            self, queries: torch.Tensor, filters: List[List[int]], chunk_size: int = -1
    ):
        """
        Returns filtered ranking for a batch of queries ordered by timestamp.
        :param queries: a torch.LongTensor of quadruples (lhs, rel, rhs, timestamp)
        :param filters: ordered filters
        :param chunk_size: maximum number of candidates processed at once
        :return:
        """
        if chunk_size < 0:
            chunk_size = self.sizes[2]
        ranks = torch.ones(len(queries))
        with torch.no_grad():
            c_begin = 0
            q = self.get_queries(queries)
            targets = self.score(queries)
            while c_begin < self.sizes[2]:
                rhs = self.get_rhs(c_begin, chunk_size,queries)
                scores = self.batch_mul(q,rhs)
                # set filtered and true scores to -1e6 to be ignored
                # take care that scores are chunked
                for i, (query, filter) in enumerate(zip(queries, filters)):
                    filter_out = filter + [query[2].item()]
                    if chunk_size < self.sizes[2]:
                        filter_in_chunk = [
                            int(x - c_begin) for x in filter_out
                            if c_begin <= x < c_begin + chunk_size
                        ]
                        max_to_filter = max(filter_in_chunk + [-1])
                        assert max_to_filter < scores.shape[1], f"fuck {scores.shape[1]} {max_to_filter}"
                        scores[i, filter_in_chunk] = -1e6
                    else:
                        scores[i, filter_out] = -1e6
                ranks += torch.sum(
                    (scores >= targets).float(), dim=1
                ).cpu()

                c_begin += chunk_size
        return ranks


class HGE_TNTComplEx(TKBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int, int], rank: int,
            no_time_emb=False, init_size: float = 1e-2, use_attention=0, att_weight=0.5, num_form='split',
            use_reverse=False,
    ):
        super(HGE_TNTComplEx, self).__init__()
        self.sizes = sizes
        self.rank = rank
        self.model_name = 'HGE_TNTComplEx'

        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True)
            for s in [sizes[0], sizes[1], sizes[3], sizes[1]]  # last embedding modules contains no_time embeddings
        ])
        self.embeddings[0].weight.data *= init_size   #entity
        self.embeddings[1].weight.data *= init_size   #dynamic_relation
        self.embeddings[2].weight.data *= init_size   #time
        self.embeddings[3].weight.data *= init_size   #static relation
        # self.bn0 = torch.nn.BatchNorm1d(self.rank*2)

        self.no_time_emb = no_time_emb
        self.use_attention = use_attention
        self.context_vec = nn.Embedding(self.sizes[1], self.rank)
        self.context_vec.weight.data = init_size * torch.randn(self.sizes[1], self.rank)
        self.rel_vec = nn.Embedding(self.sizes[1], self.rank)   #weights for relation attention
        self.rel_vec.weight.data = init_size * torch.randn(self.sizes[1], self.rank)
        self.rel_att_weight = att_weight
        self.geo_att_weight = att_weight
        self.act = nn.Softmax(dim=1)
        self.num_form = num_form
        self.use_reverse = use_reverse

    @staticmethod
    def has_time():
        return True

    ##calculates Scoring Vector in Equation 5
    def cal_num_form(self, x, y, input, num='split'):
        if num == 'split':
            return x[0] * y[0] + x[1] * y[1], x[0] * y[1] + x[1] * y[0]
        elif num == 'dual':
            return x[0] * y[0], x[0] * y[1] + x[1] * y[0]
        elif num == 'complex':
            return x[0] * y[0] - x[1] * y[1], x[0] * y[1] + x[1] * y[0]
        elif num == 'equal':
            split, dual, complex = [self.cal_num_form(x, y, input, i) for i in ['split', 'dual', 'complex']]
            return (split[0] + dual[0] + complex[0]) / 3, (split[1] + dual[1] + complex[1]) / 3
        else:
            split, dual, complex = [self.cal_num_form(x, y, input, i) for i in ['split', 'dual', 'complex']]
            if num == 'rel':
                query = self.rel_vec(input[:, 1]).view((-1, 1, self.rank))
            elif num:  ##if use full_rel/product space, use p_st as the query vector for attention
                query = y
            att_l = self.cal_att_num_form(split[0], dual[0], complex[0], query[0])
            att_r = complex[1]
            return att_l, att_r

    # calculates geometric attention in Equation 7, att_lhs refers to attention weights belta_i
    def cal_att_num_form(self, x, y, z, query):
        x = x.view((-1, 1, self.rank))
        y = y.view((-1, 1, self.rank))
        z = z.view((-1, 1, self.rank))
        query = query.view((-1, 1, self.rank))

        cands = torch.cat([x, y, z], dim=1)
        self.geo_att_weight = torch.sum(query * cands / np.sqrt(self.rank), dim=-1, keepdim=True)
        self.geo_att_weight = self.act(self.geo_att_weight)
        att_lhs = torch.sum(self.geo_att_weight * cands, dim=1)
        return att_lhs

    # calculates temporal relational attention in Equation 5
    def cal_rel(self, x):
        lhs = self.embeddings[0](x[:, 0])
        # lhs = self.bn0(lhs)
        rel = self.embeddings[1](x[:, 1])
        rel_no_time = self.embeddings[3](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        # rhs = self.bn0(rhs)
        time = self.embeddings[2](x[:, 3])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]
        rnt = rel_no_time[:, :self.rank], rel_no_time[:, self.rank:]
        rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1]
        rwt = rt[0] + rt[3], rt[1] + rt[2]

        if self.use_attention == 1:
            rwt = rwt[0].view((-1, 1, self.rank)), rwt[1].view((-1, 1, self.rank))
            rnt = rt[0].view((-1, 1, self.rank)), rnt[1].view((-1, 1, self.rank))
            cands = torch.cat([rwt[0], rnt[0]], dim=1), torch.cat([rwt[1], rnt[1]], dim=1)
            context_vec = self.context_vec(x[:, 1]).view((-1, 1, self.rank))
            # if use relational attention, relation is query, and self.rel_att_weight is the trained attention weights
            self.rel_att_weight = torch.sum(context_vec * cands[0] / np.sqrt(self.rank), dim=-1,
                                            keepdim=True), torch.sum(context_vec * cands[1] / np.sqrt(self.rank),
                                                                     dim=-1, keepdim=True)
            self.rel_att_weight = self.act(self.rel_att_weight[0]), self.act(self.rel_att_weight[1])
            # print(self.att_weight[0][:5])
            full_rel = torch.sum(self.rel_att_weight[0] * cands[0], dim=1), torch.sum(self.rel_att_weight[1] * cands[1],
                                                                                      dim=1)
        else:
            full_rel = rwt[0] + rnt[0], rwt[1] + rnt[1]
        return lhs, rhs, rt, rnt, full_rel

    def score(self, x):

        lhs, rhs, rt, rnt, full_rel = self.cal_rel(x)

        real_part, img_part = self.cal_num_form(lhs, full_rel, x, self.num_form)
        lhs_rel_1 = real_part
        lhs_rel_2 = img_part
        return torch.sum(lhs_rel_1 * rhs[0] + lhs_rel_2 * (-rhs[1]), 1, keepdim=True)

    def forward(self, x):
        right = self.embeddings[0].weight
        right = right[:, :self.rank], right[:, self.rank:]
        lhs, rhs, rt, rnt, full_rel = self.cal_rel(x)
        rrt = rt[0] + rt[3], rt[1] + rt[2]
        real_part, img_part = self.cal_num_form(lhs, full_rel, x, self.num_form)
        lhs_rel_1 = real_part
        lhs_rel_2 = img_part
        regularizer = (
            math.pow(2, 1 / 3) * torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
            torch.sqrt(rrt[0] ** 2 + rrt[1] ** 2),
            torch.sqrt(rnt[0] ** 2 + rnt[1] ** 2),
            math.pow(2, 1 / 3) * torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2)
        )
        return ((lhs_rel_1 @ right[0].t() + lhs_rel_2 @ (-right[1].t())
                 ), regularizer,
                self.embeddings[2].weight[:-1] if self.no_time_emb else self.embeddings[2].weight
                )

    def forward_over_time(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2].weight

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]

        rel_no_time = self.embeddings[3](x[:, 1])
        rnt = rel_no_time[:, :self.rank], rel_no_time[:, self.rank:]

        score_time = (
                (lhs[0] * rel[0] * rhs[0] - lhs[1] * rel[1] * rhs[0] -
                 lhs[1] * rel[0] * rhs[1] + lhs[0] * rel[1] * rhs[1]) @ time[0].t() +
                (lhs[1] * rel[0] * rhs[0] - lhs[0] * rel[1] * rhs[0] +
                 lhs[0] * rel[0] * rhs[1] - lhs[1] * rel[1] * rhs[1]) @ time[1].t()
        )
        base = torch.sum(
            (lhs[0] * rnt[0] * rhs[0] - lhs[1] * rnt[1] * rhs[0] -
             lhs[1] * rnt[0] * rhs[1] + lhs[0] * rnt[1] * rhs[1]) +
            (lhs[1] * rnt[1] * rhs[0] - lhs[0] * rnt[0] * rhs[0] +
             lhs[0] * rnt[1] * rhs[1] - lhs[1] * rnt[0] * rhs[1]),
            dim=1, keepdim=True
        )
        return score_time + base

    def get_rhs(self, chunk_begin: int, chunk_size: int, queries: torch.Tensor):
        return self.embeddings[0].weight.data[
               chunk_begin:chunk_begin + chunk_size
               ]

    def get_queries(self, queries: torch.Tensor):

        lhs, _, rt, rnt, full_rel = self.cal_rel(queries)
        real_part, img_part = self.cal_num_form(lhs, full_rel, queries, self.num_form)
        lhs_rel_1 = real_part
        lhs_rel_2 = img_part
        return torch.cat([lhs_rel_1, -lhs_rel_2], 1)

class ComplEx(TKBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int, int], rank: int,
            init_size: float = 1e-3
    ):
        super(ComplEx, self).__init__()
        self.sizes = sizes
        self.rank = rank

        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True)
            for s in [sizes[0], sizes[1]]
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size

    @staticmethod
    def has_time():
        return False

    def forward_over_time(self, x):
        raise NotImplementedError("no.")

    def score(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]

        return torch.sum(
            (lhs[0] * rel[0] - lhs[1] * rel[1]) * rhs[0] +
            (lhs[0] * rel[1] + lhs[1] * rel[0]) * rhs[1],
            1, keepdim=True
        )

    def forward(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]

        right = self.embeddings[0].weight
        right = right[:, :self.rank], right[:, self.rank:]
        return (
                       (lhs[0] * rel[0] - lhs[1] * rel[1]) @ right[0].transpose(0, 1) +
                       (lhs[0] * rel[1] + lhs[1] * rel[0]) @ right[1].transpose(0, 1)
               ), (
                   torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
                   torch.sqrt(rel[0] ** 2 + rel[1] ** 2),
                   torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2)
               ), None

    def get_rhs(self, chunk_begin: int, chunk_size: int, queries):
        return self.embeddings[0].weight.data[
               chunk_begin:chunk_begin + chunk_size
               ]

    def get_queries(self, queries: torch.Tensor):
        lhs = self.embeddings[0](queries[:, 0])
        rel = self.embeddings[1](queries[:, 1])
        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]

        return torch.cat([
            lhs[0] * rel[0] - lhs[1] * rel[1],
            lhs[0] * rel[1] + lhs[1] * rel[0]
        ], 1)

class TComplEx(TKBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int, int], rank: int,
            no_time_emb=False, init_size: float = 1e-2
    ):
        super(TComplEx, self).__init__()
        self.sizes = sizes
        self.rank = rank

        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True)
            for s in [sizes[0], sizes[1], sizes[3]]
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        self.embeddings[2].weight.data *= init_size

        self.no_time_emb = no_time_emb

    @staticmethod
    def has_time():
        return True

    def score(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]

        return torch.sum(
            (lhs[0] * rel[0] * time[0] - lhs[1] * rel[1] * time[0] -
             lhs[1] * rel[0] * time[1] - lhs[0] * rel[1] * time[1]) * rhs[0] +
            (lhs[1] * rel[0] * time[0] + lhs[0] * rel[1] * time[0] +
             lhs[0] * rel[0] * time[1] - lhs[1] * rel[1] * time[1]) * rhs[1],
            1, keepdim=True
        )

    def forward(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]

        right = self.embeddings[0].weight
        right = right[:, :self.rank], right[:, self.rank:]

        rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1]
        full_rel = rt[0] - rt[3], rt[1] + rt[2]

        return (
                       (lhs[0] * full_rel[0] - lhs[1] * full_rel[1]) @ right[0].t() +
                       (lhs[1] * full_rel[0] + lhs[0] * full_rel[1]) @ right[1].t()
               ), (
                   torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
                   torch.sqrt(full_rel[0] ** 2 + full_rel[1] ** 2),
                   torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2)
               ), self.embeddings[2].weight[:-1] if self.no_time_emb else self.embeddings[2].weight

    def forward_over_time(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2].weight

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]

        return (
                (lhs[0] * rel[0] * rhs[0] - lhs[1] * rel[1] * rhs[0] -
                 lhs[1] * rel[0] * rhs[1] + lhs[0] * rel[1] * rhs[1]) @ time[0].t() +
                (lhs[1] * rel[0] * rhs[0] - lhs[0] * rel[1] * rhs[0] +
                 lhs[0] * rel[0] * rhs[1] - lhs[1] * rel[1] * rhs[1]) @ time[1].t()
        )

    def get_rhs(self, chunk_begin, chunk_size, query=0):
        return self.embeddings[0].weight.data[
               chunk_begin:chunk_begin + chunk_size
               ].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        lhs = self.embeddings[0](queries[:, 0])
        rel = self.embeddings[1](queries[:, 1])
        time = self.embeddings[2](queries[:, 3])
        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]
        return torch.cat([
            lhs[0] * rel[0] * time[0] - lhs[1] * rel[1] * time[0] -
            lhs[1] * rel[0] * time[1] - lhs[0] * rel[1] * time[1],
            lhs[1] * rel[0] * time[0] + lhs[0] * rel[1] * time[0] +
            lhs[0] * rel[0] * time[1] - lhs[1] * rel[1] * time[1]
        ], 1)

class TNTComplEx(TKBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int, int], rank: int,
            no_time_emb=False, init_size: float = 1e-2
    ):
        super(TNTComplEx, self).__init__()
        self.sizes = sizes
        self.rank = rank

        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True)
            for s in [sizes[0], sizes[1], sizes[3], sizes[1]]  # last embedding modules contains no_time embeddings
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        self.embeddings[2].weight.data *= init_size
        self.embeddings[3].weight.data *= init_size

        self.no_time_emb = no_time_emb

    @staticmethod
    def has_time():
        return True

    def TimeODE(self, entity_emb, queries):
        hidden = self.time_hidden_embedding(entity_emb)
        time_trans = self.embeddings[2](queries[:, 3])
        time_trans = time_trans.view(time_trans.size()[0], self.rank, self.thidrank)
        if hidden.shape[0] == time_trans.shape[0]:
            timeh = torch.einsum('ijk,ik->ij', [time_trans, hidden])
        else:
            timeh = torch.einsum('ijk,mk->imj', [time_trans, hidden])
        timeh1 = torch.tanh(timeh)
        entity_time_trans = timeh1 + entity_emb

        return entity_time_trans

    def RelODE(self, entity_emb, queries):
        hidden = self.rel_hidden_embedding(entity_emb)
        rel_trans = self.embeddings[1](queries[:, 1])
        relation = rel_trans.view(rel_trans.size()[0], self.rank, self.hidrank)
        relationh = torch.einsum('ijk,ik->ij', [relation, hidden])
        relationh1 = torch.tanh(relationh)
        entity_r_trans = relationh1 + entity_emb
        return entity_r_trans

    def score(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rel_no_time = self.embeddings[3](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]
        rnt = rel_no_time[:, :self.rank], rel_no_time[:, self.rank:]

        rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1]
        full_rel = (rt[0] - rt[3]) + rnt[0], (rt[1] + rt[2]) + rnt[1]

        return torch.sum(
            (lhs[0] * full_rel[0] - lhs[1] * full_rel[1]) * rhs[0] +
            (lhs[1] * full_rel[0] + lhs[0] * full_rel[1]) * rhs[1],
            1, keepdim=True
        )

    def forward(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rel_no_time = self.embeddings[3](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]

        rnt = rel_no_time[:, :self.rank], rel_no_time[:, self.rank:]

        right = self.embeddings[0].weight
        right = right[:, :self.rank], right[:, self.rank:]

        rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1]
        rrt = rt[0] - rt[3], rt[1] + rt[2]
        full_rel = rrt[0] + rnt[0], rrt[1] + rnt[1]

        regularizer = (
           math.pow(2, 1 / 3) * torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
           torch.sqrt(rrt[0] ** 2 + rrt[1] ** 2),
           torch.sqrt(rnt[0] ** 2 + rnt[1] ** 2),
           math.pow(2, 1 / 3) * torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2)
        )
        return ((
               (lhs[0] * full_rel[0] - lhs[1] * full_rel[1]) @ right[0].t() +
               (lhs[1] * full_rel[0] + lhs[0] * full_rel[1]) @ right[1].t()
            ), regularizer,
               self.embeddings[2].weight[:-1] if self.no_time_emb else self.embeddings[2].weight
        )

    def forward_over_time(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2].weight

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]

        rel_no_time = self.embeddings[3](x[:, 1])
        rnt = rel_no_time[:, :self.rank], rel_no_time[:, self.rank:]

        score_time = (
            (lhs[0] * rel[0] * rhs[0] - lhs[1] * rel[1] * rhs[0] -
             lhs[1] * rel[0] * rhs[1] + lhs[0] * rel[1] * rhs[1]) @ time[0].t() +
            (lhs[1] * rel[0] * rhs[0] - lhs[0] * rel[1] * rhs[0] +
             lhs[0] * rel[0] * rhs[1] - lhs[1] * rel[1] * rhs[1]) @ time[1].t()
        )
        base = torch.sum(
            (lhs[0] * rnt[0] * rhs[0] - lhs[1] * rnt[1] * rhs[0] -
             lhs[1] * rnt[0] * rhs[1] + lhs[0] * rnt[1] * rhs[1]) +
            (lhs[1] * rnt[1] * rhs[0] - lhs[0] * rnt[0] * rhs[0] +
             lhs[0] * rnt[1] * rhs[1] - lhs[1] * rnt[0] * rhs[1]),
            dim=1, keepdim=True
        )
        return score_time + base

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.embeddings[0].weight.data[
               chunk_begin:chunk_begin + chunk_size
               ].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        lhs = self.embeddings[0](queries[:, 0])
        rel = self.embeddings[1](queries[:, 1])
        rel_no_time = self.embeddings[3](queries[:, 1])
        time = self.embeddings[2](queries[:, 3])

        #head_time_trans = self.TimeODE(lhs,queries)
        lhs = self.RelODE(lhs, queries)
        #lhs = self.RelODE(head_time_trans,queries)

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]
        rnt = rel_no_time[:, :self.rank], rel_no_time[:, self.rank:]

        rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1]
        full_rel = (rt[0] - rt[3]) + rnt[0], (rt[1] + rt[2]) + rnt[1]

        return torch.cat([
            lhs[0] * full_rel[0] - lhs[1] * full_rel[1],
            lhs[1] * full_rel[0] + lhs[0] * full_rel[1]
        ], 1)

class HGE_TComplEx(TKBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int, int], rank: int,
            no_time_emb=False, init_size: float = 1e-2, use_attention=0, att_weight=0.5, num_form='split'
    ):
        super(HGE_TComplEx, self).__init__()
        self.sizes = sizes
        self.rank = rank
        self.model_name = "HGE_TComplEx"

        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True)
            for s in [sizes[0], sizes[1], sizes[3]]
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        self.embeddings[2].weight.data *= init_size

        self.no_time_emb = no_time_emb
        self.use_attention = use_attention
        self.context_vec = nn.Embedding(self.sizes[1], self.rank)
        self.context_vec.weight.data = init_size * torch.randn(self.sizes[1], self.rank)
        self.att_weight=att_weight
        self.act = nn.Softmax(dim=1)
        self.num_form = num_form

    @staticmethod
    def has_time():
        return True

    ##calculates Scoring Vector in Equation 5
    def cal_num_form(self,x,y,input,num='split'):
        if num=='split':
            return x[0] * y[0] + x[1] * y[1], -(x[0] * y[1]+x[1] * y[0])
        elif num == 'dual':
            return x[0]*y[0], -(x[0]*y[1]+x[1]*y[0])
        elif num == 'complex':
            return x[0] * y[0] - x[1] * y[1], x[0] * y[1]+x[1] * y[0]
        else:
            split, dual, complex = [self.cal_num_form(x, y, input, i) for i in ['split', 'dual', 'complex']]
            if num =='rel':
                query = self.rel_vec(input[:, 1]).view((-1, 1, self.rank))
            else:
                query = y
            att_l, att_r = [self.cal_att_num_form(split[i], dual[i], complex[i], query[i]) for i in [0, 1]]
            return att_l, att_r

    #calculates geometric attention in Equation 7, att_lhs refers to attention weights belta_i
    def cal_att_num_form(self, x, y, z, query):

        x, y, z, query = [i.view((-1, 1, self.rank)) for i in [x,y,z, query]]
        cands = torch.cat([x,y,z], dim=1)
        self.att_weight = torch.sum(query * cands /np.sqrt(self.rank), dim=-1, keepdim=True)
        self.att_weight = self.act(self.att_weight)
        att_lhs = torch.sum(self.att_weight * cands, dim=1)
        return att_lhs

    def score(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]

        rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1]
        if self.num_form == 'dual':
            full_rel = rt[0], rt[1] + rt[2]
        elif self.num_form == 'complex':
            full_rel = rt[0] - rt[3], rt[1] + rt[2]
        else:
            full_rel = rt[0] + rt[3], rt[1] + rt[2]

        real_part, img_part = self.cal_num_form(lhs,full_rel,x,self.num_form)

        return torch.sum(real_part*rhs[0] + img_part*rhs[1], dim =1, keepdim=True)

    def forward(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]

        right = self.embeddings[0].weight
        right = right[:, :self.rank], right[:, self.rank:]

        rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1]

        if self.num_form == 'spilt':
            full_rel = rt[0] + rt[3], rt[1] + rt[2]
        elif self.num_form == 'complex':
            full_rel = rt[0] - rt[3], rt[1] + rt[2]
        else:
            full_rel = rt[0] , rt[1] + rt[2]

        real_part, img_part = self.cal_num_form(lhs,full_rel,x,self.num_form)

        return (
                       (real_part) @ right[0].t() +
                       (img_part) @ right[1].t()
               ), (
                   torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
                   torch.sqrt(full_rel[0] ** 2 + full_rel[1] ** 2),
                   torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2)
               ), self.embeddings[2].weight[:-1] if self.no_time_emb else self.embeddings[2].weight

    def forward_over_time(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2].weight

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]

        return (
                (lhs[0] * rel[0] * rhs[0] - lhs[1] * rel[1] * rhs[0] -
                 lhs[1] * rel[0] * rhs[1] + lhs[0] * rel[1] * rhs[1]) @ time[0].t() +
                (lhs[1] * rel[0] * rhs[0] - lhs[0] * rel[1] * rhs[0] +
                 lhs[0] * rel[0] * rhs[1] - lhs[1] * rel[1] * rhs[1]) @ time[1].t()
        )

    def get_rhs(self, chunk_begin: int, chunk_size: int,queries: torch.Tensor):
        return self.embeddings[0].weight.data[
               chunk_begin:chunk_begin + chunk_size
               ]

    def get_queries(self, queries: torch.Tensor):
        lhs = self.embeddings[0](queries[:, 0])
        rel = self.embeddings[1](queries[:, 1])
        time = self.embeddings[2](queries[:, 3])
        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]

        rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1]

        if self.num_form == 'spilt':
            full_rel = rt[0] + rt[3], rt[1] + rt[2]
        elif self.num_form == 'complex':
            full_rel = rt[0] - rt[3], rt[1] + rt[2]
        else:
            full_rel = rt[0] , rt[1] + rt[2]

        real_part, img_part = self.cal_num_form(lhs,full_rel,queries,self.num_form)

        return torch.cat(
            [(real_part),
             (img_part)],
            1
        )

