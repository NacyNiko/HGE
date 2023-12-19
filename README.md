## HGE: Embedding Temporal Knowledge Graphs in a Product Space of Heterogeneous Geometric Subspaces

This code reproduces results in HGE: Embedding Temporal Knowledge Graphs in
a Product Space of Heterogeneous Geometric Subspaces.

## Installation
Create a conda environment with pytorch and scikit-learn :
```
conda create --name tkbc_env python=3.7
source activate tkbc_env
conda install --file requirements.txt -c pytorch
```

Then install the kbc package to this environment
```
python setup.py install
```

## Datasets
Datasets are provided in tkbc/src_data folder. Due to file size limit, GDELT will be uploaded in the final version.

Process data by running :
```
python tkbc/process_icews.py #for icews14, icews05-15 and gdelt
python tkbc/process_wikidata12k.py  # for wikidata12k
```

This will create the files required to compute the filtered metrics.

## Reproducing results

In order to reproduce the results on the datasets in the paper, run the following commands. Detailed hyperparameters could be found in hyperparameter.pdf

```
python tkbc/learner.py --dataset ICEWS14 --model HGE_TNTComplEx --rank 1200 --emb_reg 3e-3 --time_reg 3e-2 --max_epochs 200  --device cuda:0 --attention 1 --num_form full_rel
python tkbc/learner.py --dataset ICEWS05-15 --model HGE_TNTComplEx --rank 1200 --emb_reg 3e-3 --time_reg 1e-1 --max_epochs 200  --device cuda:0 --attention 1 --num_form full_rel
python tkbc/learner.py --dataset ICEWS14 --model HGE_TComplEx --rank 1200 --emb_reg 3e-3 --time_reg 3e-2 --max_epochs 200  --device cuda:0 --num_form full_rel
python tkbc/learner.py --dataset ICEWS05-15 --model HGE_TComplEx --rank 1200 --emb_reg 3e-3 --time_reg 1e-1 --num_form full_rel --max_epochs 200
python tkbc/learner.py --dataset ICEWS05-15 --model HGE_TLT_KGE --rank 1200 --cycle 1440 --use_reverse --num_form full_rel --max_epochs 200 

```
##Test result

In order to test the results on trained checkpoint, run model_test.py as follows:
```
python model_test.py --model_dir (checkpoint_path) --device ('cpu' or 'cuda:0')
```

##Parameters
```
    '--dataset', type=str, help="Dataset name", ['ICEWS14', 'ICEWS05-15','gdelt','wikidata12k' ]

    '--model', choices=models, models = ['ComplEx', 'TComplEx', 'TNTComplEx', 'HGE_TNTComplEx', 'HGE_TComplEx','HGE_TLT_KGE']
    
    '--max_epochs', default=200, type=int, help="Number of epochs."

    '--valid_freq', default=5, type=int, help="Number of epochs between each valid."

    '--rank', default=100, type=int, help="embedding dimension"
    
    '--batch_size', default=1000, type=int,
    
    '--learning_rate', default=1e-1, type=float,

    '--emb_reg', default=0., type=float, help="Entity/Relation Embedding regularizer weight"
    
    '--time_reg', default=0., type=float, help="Timestamp regularizer weight"

    '--device', default="cpu", help="CPU or cuda:0 device"
    
    '--attention', default=1, type=int, help="use temporal-relational attention or not"

    '--num_form', default="full_rel", help="which space will the model take", select between ['complex','dual', 'split','full_rel'(product space)]
```

##Ablation Study
To test the performance of temporal-relational attention, set --attention = 0.  
To test the performance of temporal-geometric attention, set --num_form = 'complex/dual/split' to get results from single geometry.

##Pattern Subset Test
To get model's performance on individual temporal pattern set,run:
```angular2html
python instance_test.py
```
and replaces relevant dataset/checkpoint name in line 246/250

##LCGE
Please check LCGE_new folder to see details of LCGE's inference error and our implementation. 

## License
tkbc is CC-BY-NC licensed, as found in the LICENSE file.

## Acknowledge
This project is based on tkbc: https://github.com/facebookresearch/tkbc.
