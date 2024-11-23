## SAGCN

**#TODO: modify**

This is the Pytorch implementation for our paper:

>SIGIR 2020. Xiangnan He, Kuan Deng ,Xiang Wang, Yan Li, Yongdong Zhang, Meng Wang(2020). LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation, [Paper in arXiv](https://arxiv.org/abs/2002.02126).

Author: Prof. Xiangnan He (staff.ustc.edu.cn/~hexn/)

## Introduction

In this work, we aim to utilize LLMs to extract information from different aspects and construct graphs accordingly for better recommendation performance. We release corresponding datasets for further study. Our code is built heavily on [LightGCN-Pytorch](https://github.com/gusye1234/LightGCN-PyTorch) and you can resort to it for more details.

## Enviroment Requirement

Follow `envison.sh`.

## Dataset

We provide three processed datasets: Baby, Clothing and Office from Amazon. You can access datasets/readmd.md for more details and download links.

## Examples of SAGCN with 8 aspects

Below command will repeat the training progress of some of our best results. For more options, please check `parse.py`.

* command

```shell
# baby
python main.py \
    --layer 6   \
    --dataset baby  \
    --model sagcn   \
    --mode concat   \
    --explicit_factors "quality" "functionality" "comfort" "ease_of_use" "design" "durability" "size" "price"   \
    --has_implicit 0   \
    --explicit_graph specific

# clothing
python main.py \
    --layer 6   \
    --dataset clothing  \
    --explicit_factors "quality", "comfort", "appearance", "style", "fit", "design", "size", "price"

# office
python main.py \
    --layer 5   \
    --dataset office  \
    --explicit_factors "quality", "functionality", "ease_of_use", "convenience", "comfort", "durability", "design", "price"
```

If you'd like to try LightGCN using our code, you can run the following command:

```shell
# baby
python main.py  \
    --layer 3   \
    --decay 0.01    \
    --dataset baby  \
    --model lgn   \
    --has_implicit 1   \
    --explicit_graph common

# office
python main.py  \
    --layer 5   \
    --dataset office  \
    --model lgn   \
    --has_implicit 1   \
    --explicit_graph common

# clothing
python main.py  \
    --layer 4   \
    --dataset clothing  \
    --model lgn   \
    --has_implicit 1   \
    --explicit_graph common
```

* log output example

```shell
...
[TEST]
{'precision': array([0.00856261, 0.0066701 , 0.00454924]), 'recall': array([0.06362961, 0.09798353, 0.16469945]), 'ndcg': array([0.03518474, 0.04443586, 0.05867013]), 'hr': array([0.08228336, 0.12625354, 0.20776549])}
EPOCH[6/1000] loss0.105-|Sample:1.56|
...
```

## Results
*all metrics is under top-20*

***pytorch* version results** (stop at 1000 epochs, early stop patience is 30 epochs):

(*for seed=0*)

### LightGCN

* baby:

| layer | HR  | NDCG | Precision | Recall | decay  | lr     |
| ----- | ------ | ------- | ------------ | --------- | ------ | ------ |
| 1     | 0.1082 | 0.0385  | 0.0057       | 0.0829    | 0.0100 | 0.0100 |
| 2     | 0.1166 | 0.0409  | 0.0061       | 0.0887    | 0.0100 | 0.0100 |
| 3     | 0.1218 | 0.0420  | 0.0064       | 0.0925    | 0.0100 | 0.0100 |
| 4     | 0.1198 | 0.0411  | 0.0063       | 0.0917    | 0.0100 | 0.0100 |
| 5     | 0.1111 | 0.0400  | 0.0059       | 0.0856    | 0.0010 | 0.0100 |
| 6     | 0.1132 | 0.0405  | 0.0060       | 0.0877    | 0.0010 | 0.0100 |
| 7     | 0.1162 | 0.0412  | 0.0061       | 0.0895    | 0.0010 | 0.0100 |

* office

| layer | HR  | NDCG | Precision | Recall | decay  | lr     |
| ----- | ------ | ------- | ------------ | --------- | ------ | ------ |
| 1     | 0.2222 | 0.0727  | 0.0134       | 0.1402    | 0.0100 | 0.0100 |
| 2     | 0.2249 | 0.0734  | 0.0135       | 0.1432    | 0.0100 | 0.0100 |
| 3     | 0.2255 | 0.0729  | 0.0135       | 0.1439    | 0.0100 | 0.0100 |
| 4     | 0.2186 | 0.0714  | 0.0133       | 0.1387    | 0.0010 | 0.0100 |
| 5     | 0.2277 | 0.0743  | 0.0137       | 0.1447    | 0.0010 | 0.0100 |
| 6     | 0.2336 | 0.0765  | 0.0141       | 0.1498    | 0.0010 | 0.0100 |
| 7     | 0.2290 | 0.0751  | 0.0137       | 0.1463    | 0.0010 | 0.0100 |

* clothing

| layer | HR  | NDCG | Precision | Recall | decay  | lr     |
| ----- | ------ | ------- | ------------ | --------- | ------ | ------ |
| 1     | 0.0711 | 0.0289  | 0.0037       | 0.0618    | 0.0100 | 0.0100 |
| 2     | 0.0740 | 0.0297  | 0.0038       | 0.0645    | 0.0100 | 0.0100 |
| 3     | 0.0757 | 0.0298  | 0.0039       | 0.0658    | 0.0100 | 0.0100 |
| 4     | 0.0739 | 0.0295  | 0.0038       | 0.0644    | 0.0010 | 0.0100 |
| 5     | 0.0754 | 0.0302  | 0.0039       | 0.0657    | 0.0010 | 0.0100 |
| 6     | 0.0757 | 0.0303  | 0.0039       | 0.0658    | 0.0010 | 0.0100 |
| 7     | 0.0757 | 0.0302  | 0.0039       | 0.0656    | 0.0010 | 0.0100 |

### SAGCN with 8 aspects

* baby

| layer | HR  | NDCG | Precision | Recall | decay  | lr     |
| ----- | ------ | ------- | ------------ | --------- | ------ | ------ |
| 1     | 0.1172 | 0.0437  | 0.0062       | 0.0908    | 0.0010 | 0.0010 |
| 2     | 0.1288 | 0.0460  | 0.0068       | 0.0996    | 0.0100 | 0.0010 |
| 3     | 0.1321 | 0.0486  | 0.0070       | 0.1033    | 0.0010 | 0.0010 |
| 4     | 0.1351 | 0.0499  | 0.0071       | 0.1058    | 0.0010 | 0.0010 |
| 5     | 0.1316 | 0.0503  | 0.0069       | 0.1043    | 0.0010 | 0.0100 |
| 6     | 0.1337 | 0.0509  | 0.0070       | 0.1056    | 0.0010 | 0.0100 |
| 7     | 0.1357 | 0.0523  | 0.0071       | 0.1064    | 0.0010 | 0.0100 |

* office

| layer | HR  | NDCG | Precision | Recall | decay  | lr     |
| ----- | ------ | ------- | ------------ | --------- | ------ | ------ |
| 1     | 0.2402 | 0.0831  | 0.0148       | 0.1540    | 0.0100 | 0.0010 |
| 2     | 0.2353 | 0.0810  | 0.0140       | 0.1557    | 0.0100 | 0.0100 |
| 3     | 0.2428 | 0.0865  | 0.0147       | 0.1593    | 0.0010 | 0.0010 |
| 4     | 0.2446 | 0.0855  | 0.0148       | 0.1617    | 0.0010 | 0.0100 |
| 5     | 0.2518 | 0.0884  | 0.0153       | 0.1671    | 0.0010 | 0.0100 |
| 6     | 0.2479 | 0.0894  | 0.0148       | 0.1635    | 0.0010 | 0.0100 |
| 7     | 0.2516 | 0.0888  | 0.0152       | 0.1671    | 0.0010 | 0.0100 |

* clothing

| layer | HR  | NDCG | Precision | Recall | decay  | lr     |
| ----- | ------ | ------- | ------------ | --------- | ------ | ------ |
| 1     | 0.0832 | 0.0356  | 0.0043       | 0.0730    | 0.0010 | 0.0010 |
| 2     | 0.0899 | 0.0380  | 0.0046       | 0.0790    | 0.0010 | 0.0010 |
| 3     | 0.0907 | 0.0381  | 0.0047       | 0.0804    | 0.0010 | 0.0100 |
| 4     | 0.0931 | 0.0405  | 0.0048       | 0.0829    | 0.0010 | 0.0100 |
| 5     | 0.0944 | 0.0417  | 0.0048       | 0.0842    | 0.0010 | 0.0100 |
| 6     | 0.0949 | 0.0420  | 0.0049       | 0.0844    | 0.0010 | 0.0100 |
| 7     | 0.0945 | 0.0424  | 0.0048       | 0.0845    | 0.0010 | 0.0100 |
