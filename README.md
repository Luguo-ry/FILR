# FILR
Code for paper Document-level Biomedical Relation Extraction Based on Multi-Dimensional Fusion Information and Multi-Granularity Logical Reasoning.
# DataSet
The CDR and GDA datasets can be downloaded from [Google Drive](https://drive.google.com/drive/folders/13RgVm7IfEnm4_dV2UTQKIwJExlhyD4pA).
# File Structure
The expected structure of files is:
```
ATLOP
 |-- dataset
 |    |-- cdr
 |    |    |-- train_filter.data
 |    |    |-- dev_filter.data
 |    |    |-- test_filter.data
 |    |-- gda
 |    |    |-- train.data
 |    |    |-- dev.data
 |    |    |-- test.data
 |-- saved_model
      |-- best.model
 |-- biobert_base
 |-- utils.py
 |-- adj_utils.py
 |-- prepro.py
 |-- long_seq.py
 |-- losses.py
 |-- train_cdr.py
 |-- train_gda.py
 |-- rgcn.py
 |-- model.py
```
The `` biobert_base `` can be downloaded from [Google Drive](https://drive.google.com/drive/folders/13RgVm7IfEnm4_dV2UTQKIwJExlhyD4pA?usp=sharing).
# Training and Evaluation
## Training
Train CDA and GDA model with the following command:
```
>> python train_cdr.py  # for CDR
>> python train_gda.py  # for GDA
```
You can save the model by setting the ``--save_path`` argument before training. The model correponds to the best dev results will be saved. 
# Evaluation
You can download the saved models we reported in paper from [Google Drive](https://drive.google.com/drive/folders/13RgVm7IfEnm4_dV2UTQKIwJExlhyD4pA?usp=sharing) and place them in ``--save_path``.
Then, you can evaluate the saved model by setting the ``--load_path`` argument, then the code will skip training and evaluate the saved model on benchmarks.
