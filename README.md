# ALD-TN
Aggressice Language detection with Joint Text Normalization
This repository implements the dependency parser described in the paper [Aggressive Language Detection with Joint Text Normalization via Adversarial Multi-task Learning](https://arxiv.org/pdf/2009.09174.pdf)

# Model
## bilstm
we use shared-private encoder via bilstm
### data preparing
In this module, we offer two encoder ways, one is bilstm, the other is hierarchical neural network via bilstm. 
The former only needs to get dictionary file, the later needs to split every data into sentences.
The directory 'data' gives examples. 

```angular2html
dict.csv  # merge the TN training data and ALD training data
```
### Training
```angular2html
python multi_task_test.py
```
## transformer
### data preparing
To get prepared data run:
```angular2html
cd transformer
python create_data.py
```
### Training
```angular2html
cd transformer
python multi_task_train.py
```
More details refer to code.

## bert

### data preparing
We need to get dictionary file firstly, please refer to 'transformer'
### Training
```angular2html
cd bert 
python run_task6.py
```
## Reference

Please kindly cite this paper in your publications if it helps your research:

```
@inproceedings{wu2020aggressive,
  title={Aggressive Language Detection with Joint Text Normalization via Adversarial Multi-task Learning},
  author={Wu, Shengqiong and Fei, Hao and Ji, Donghong},
  booktitle={CCF International Conference on Natural Language Processing and Chinese Computing},
  pages={683--696},
  year={2020},
  organization={Springer}
}
```
