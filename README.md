
# Privacy-Preserving Synthetic Data Generation for Recommendation Systems

This is our implementation for the paper:

Fan Liu, Yaqi Liu, Huilin Chen, Zhiyong Cheng, Liqiang Nie, Mohan Kankanhalli. 2025. [Understanding Before Recommendation: Semantic Aspect-Aware Review Exploitation via Large Language Models](https://doi.org/10.1145/3704999). ACM Transactions on Information Systems, 2025, 43(2): 1-26.

Please cite our SIGIR'23 paper if you use our codes. Thanks!

### Table of contents
1. [Requirement](#enviroment-requirement)
2. [Dataset](#Dataset)
3. [Usage LLMSAGCN](#Examples-of-LLMSAGCN-with-8-aspects)
4. [Results](#Results)

## Enviroment Requirement
- Pytorch == 1.12.0
- numpy == 1.24.4
- scipy == 1.10.1
- pandas == 2.0.3

## Dataset

We provide three datasets: Office, Clothing, and Baby with public researchers.

### 1. Statistics of the Experimental Datasets.
|#Interactions|#Users|#Items|#interactions|sparsity|
|:-|:-|:-|:-|:-|
|Office|4,905|2,420|53,258| 99.55%|
|Baby|19,445| 7,050| 160,792| 99.88%|
|Clothing| 39,387 | 23,033 | 278,677 | 99.97%|

### 2. The extracted semantic-aware aspects and the number of their associated interactions.

<table>
	<tr>
      <th colspan="1">Datasets</th>
	    <th colspan="8">Semantic-aware Aspects</th>
	</tr >
  <tr>
	    <td rowspan="2">Office</td>
	    <td>Quality </td>
	    <td>Functionality</td>
      <td>Ease of Use </td>
      <td>Convenience</td>
      <td>Comfort</td>
      <td>Durability</td>
      <td>Design</td>
      <td>Price</td>
	</tr >
  <tr>
	    <td>43,850 </td>
	    <td>43,269 </td>
      <td>42,347 </td>
      <td>41,238 </td>
      <td>40,795 </td>
      <td>37,564 </td>
      <td>23,973 </td>
      <td>23,661</td>
	</tr >
  <tr>
	    <td rowspan="2">Baby</td>
	    <td>Quality </td>
	    <td>Functionality</td>
      <td>Comfort</td>
      <td>Ease of Use </td>
      <td>Design</td>
      <td>Durability</td>
      <td>Size</td>
      <td>Price</td>
	</tr >
  <tr>
	    <td>133,337</td>
	    <td>132,827</td>
      <td>127,938</td>
      <td><125,014</td>
      <td>119,013</td>
      <td>116,212</td>
      <td>86,125</td>
      <td>59,859</td>
	</tr >
  <tr>
	    <td rowspan="2">Clothing</td>
	    <td>Quality </td>
	    <td>Comfort</td>
      <td>Appearance</td>
      <td>Style</td>
      <td>Fit</td>
      <td>Design</td>
      <td>Size</td>
      <td>Price</td>
	</tr >
  <tr>
	    <td>230,162</td>
	    <td>210,254</td>
      <td>205,532</td>
      <td>188,378</td>
      <td>186,132</td>
      <td>181,222</td>
      <td>170,869</td>
      <td>106,416</td>
	</tr >
</table>


The training datasets and LLM-generated results can be downloaded from [Google Ddrive](https://drive.google.com/drive/folders/1kGsm9RaUnhn3ujJoEi3MIZQofHb5XRmt?usp=sharing) 

-`graph_xxxx.txt` Aspect-related Train file. Each line is a user with her/his positive interactions with items: (userID and itemID)
-`test.txt` Test file. Each line is a user with her/his several positive interactions with items: (userID and itemID)
-`data.pkl` LLM-generated results. 
 
## Examples of LLMSAGCN with 8 aspects

The below command will repeat the training progress of some of our best results. For more options, please check `parse.py`.

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

* Office

| layer | HR  | NDCG | Precision | Recall | decay  | lr     |
| ----- | ------ | ------- | ------------ | --------- | ------ | ------ |
| 1     | 0.2222 | 0.0727  | 0.0134       | 0.1402    | 0.0100 | 0.0100 |
| 2     | 0.2249 | 0.0734  | 0.0135       | 0.1432    | 0.0100 | 0.0100 |
| 3     | 0.2255 | 0.0729  | 0.0135       | 0.1439    | 0.0100 | 0.0100 |
| 4     | 0.2186 | 0.0714  | 0.0133       | 0.1387    | 0.0010 | 0.0100 |
| 5     | 0.2277 | 0.0743  | 0.0137       | 0.1447    | 0.0010 | 0.0100 |
| 6     | 0.2336 | 0.0765  | 0.0141       | 0.1498    | 0.0010 | 0.0100 |
| 7     | 0.2290 | 0.0751  | 0.0137       | 0.1463    | 0.0010 | 0.0100 |


* Baby:

| layer | HR  | NDCG | Precision | Recall | decay  | lr     |
| ----- | ------ | ------- | ------------ | --------- | ------ | ------ |
| 1     | 0.1082 | 0.0385  | 0.0057       | 0.0829    | 0.0100 | 0.0100 |
| 2     | 0.1166 | 0.0409  | 0.0061       | 0.0887    | 0.0100 | 0.0100 |
| 3     | 0.1218 | 0.0420  | 0.0064       | 0.0925    | 0.0100 | 0.0100 |
| 4     | 0.1198 | 0.0411  | 0.0063       | 0.0917    | 0.0100 | 0.0100 |
| 5     | 0.1111 | 0.0400  | 0.0059       | 0.0856    | 0.0010 | 0.0100 |
| 6     | 0.1132 | 0.0405  | 0.0060       | 0.0877    | 0.0010 | 0.0100 |
| 7     | 0.1162 | 0.0412  | 0.0061       | 0.0895    | 0.0010 | 0.0100 |


* Clothing

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

* Office

| layer | HR  | NDCG | Precision | Recall | decay  | lr     |
| ----- | ------ | ------- | ------------ | --------- | ------ | ------ |
| 1     | 0.2402 | 0.0831  | 0.0148       | 0.1540    | 0.0100 | 0.0010 |
| 2     | 0.2353 | 0.0810  | 0.0140       | 0.1557    | 0.0100 | 0.0100 |
| 3     | 0.2428 | 0.0865  | 0.0147       | 0.1593    | 0.0010 | 0.0010 |
| 4     | 0.2446 | 0.0855  | 0.0148       | 0.1617    | 0.0010 | 0.0100 |
| 5     | 0.2518 | 0.0884  | 0.0153       | 0.1671    | 0.0010 | 0.0100 |
| 6     | 0.2479 | 0.0894  | 0.0148       | 0.1635    | 0.0010 | 0.0100 |
| 7     | 0.2516 | 0.0888  | 0.0152       | 0.1671    | 0.0010 | 0.0100 |


* Baby

| layer | HR  | NDCG | Precision | Recall | decay  | lr     |
| ----- | ------ | ------- | ------------ | --------- | ------ | ------ |
| 1     | 0.1172 | 0.0437  | 0.0062       | 0.0908    | 0.0010 | 0.0010 |
| 2     | 0.1288 | 0.0460  | 0.0068       | 0.0996    | 0.0100 | 0.0010 |
| 3     | 0.1321 | 0.0486  | 0.0070       | 0.1033    | 0.0010 | 0.0010 |
| 4     | 0.1351 | 0.0499  | 0.0071       | 0.1058    | 0.0010 | 0.0010 |
| 5     | 0.1316 | 0.0503  | 0.0069       | 0.1043    | 0.0010 | 0.0100 |
| 6     | 0.1337 | 0.0509  | 0.0070       | 0.1056    | 0.0010 | 0.0100 |
| 7     | 0.1357 | 0.0523  | 0.0071       | 0.1064    | 0.0010 | 0.0100 |


* Clothing

| layer | HR  | NDCG | Precision | Recall | decay  | lr     |
| ----- | ------ | ------- | ------------ | --------- | ------ | ------ |
| 1     | 0.0832 | 0.0356  | 0.0043       | 0.0730    | 0.0010 | 0.0010 |
| 2     | 0.0899 | 0.0380  | 0.0046       | 0.0790    | 0.0010 | 0.0010 |
| 3     | 0.0907 | 0.0381  | 0.0047       | 0.0804    | 0.0010 | 0.0100 |
| 4     | 0.0931 | 0.0405  | 0.0048       | 0.0829    | 0.0010 | 0.0100 |
| 5     | 0.0944 | 0.0417  | 0.0048       | 0.0842    | 0.0010 | 0.0100 |
| 6     | 0.0949 | 0.0420  | 0.0049       | 0.0844    | 0.0010 | 0.0100 |
| 7     | 0.0945 | 0.0424  | 0.0048       | 0.0845    | 0.0010 | 0.0100 |
