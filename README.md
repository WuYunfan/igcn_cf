# INMO: Inductive Embedding Module for CF

This is our official implementation for the paper:

Yunfan Wu, Qi Cao, Huawei Shen, Shuchang Tao & Xueqi Cheng. 2022. **INMO: A Model-Agnostic and Scalable Module for Inductive
Collaborative Filtering**  , In *Proceedings of SIGIR'22*. 

This paper is available in [ACM digital library](https://dl.acm.org/doi/abs/10.1145/3477495.3532000).

## Environment

Python 3.8

Pytorch >= 1.8

DGL >= 0.8

## Dataset

The processed data of Gowalla, Yelp, and Amazon-book can be downloaded in [Baidu Wangpan](https://pan.baidu.com/s/18VcjV_HLhf9FcKgr3-tusQ) with code 1189, or [Google Drive](https://drive.google.com/file/d/1BAN5MJXtRinHTypsszgpTMIJx2RaSj54/view?usp=sharing).

Please place the processed data like:

```
├─igcn_cf
│  ├─data
│  │  ├─Gowalla
|  |  | |─time
|  |  | |-0
|  |  | └─...
│  │  |─Yelp
│  │  └─Amazon
│  |─run
|  └─...
```

For each dataset, **time** folder contains the dataset splitted by time, which is used to tune hyperparameters. 

**0,1,2,3,4** are five random splitted datasets, which are used to train in the transductive scenario and evaluate. 

**0_dropit** is the reduced version of **0** with fewer interactions, which is used to train in the inductive scenario with new interactions.  

**0_dropui** is the reduced version of **0** with fewer users and fewer items, which is used to train in the inductive scenario with new users/items. 


## Quick Start

To launch the experiment in different scenario settings, you can use:

```
python -u -m run.run
python -u -m run.dropit.igcn_dropit
python -u -m run.dropui.igcn_dropui
```
**We provide the implemented codes of all baseline methods in an unified framework**.

To run diffrent baseline methods in the transductive setting, you can change the this code **dataset_config, model_config, trainer_config = config[2]** with different config indices in run/run.py.

To run diffrent baseline methods in inductive settings, you can use their corresponding scripts like **python -u -m run.dropui.lgcn_dropui**.

To use different dataset splits, you can change the this code **dataset_config['path'] = dataset_config['path'][:-4] + str(1)** with different split indices.

The hyperparameters of all methods can be easily changed in config.py.
