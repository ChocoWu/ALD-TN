# ALD-TN
Aggressice Language detection with Joint Text Normalization

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
cd transformer/util
python data_processing
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
