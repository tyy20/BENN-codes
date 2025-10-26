

## Simulations - Models A--C

### Related codes

- Folder: `simulation-model-ABC`
- BENN: `benn-1000.ipynb` `benn-2000.ipynb` `benn-5000.ipynb`
- StoNet: `my_stonet_all.ipynb`
- Data summarization: `results-summary-model123-final.R`

### Notes for running

- Some `.py` files are also provided in case it takes too long to run the `.ipynb` file.
- We conduct preliminary analysis for BENN: `benn-prelim-model123.ipynb`
- Directories need to be changed accordingly in each code.
- Results from GMDDNet and GSIR come from Chen et al. (2024). https://doi.org/10.1214/24-AOS2390





## Simulation - Model D-IV

- Folder: `simulation-model-D-IV`
- Note: In all commands, `n` is replaced by sample size (1000, ..., 8000); `t` is replaced by the index of the data (1, ..., 100).

### BENN-1

- Related codes: `benn-model4-poly.ipynb` `benn-model4-poly.py`
- Preliminary analysis codes: `benn-prelim-model4-1-1.ipynb`
- Command: 
```bash
python benn-model4-poly.py --model1 4 --model2 1 --n n --m 1 --d 2 --t t --ep 100
```


### BENN-2

- Related codes: `benn-model4-poly.ipynb` `benn-model4-poly.py`
- Preliminary analysis codes: `benn-prelim-model4-1-2.ipynb`
- Command: 
```bash
python benn-model4-poly.py --model1 4 --model2 1 --n n --m 2 --d 2 --t t --ep 100
```


### BENN-1000

- Related codes: `benn-all-ens-m-ker.ipynb` `benn-all-ens-m-ker.py`
- Preliminary analysis codes: `benn-prelim-model4-1.ipynb`
- Command: 
```bash
python benn-all-ens-m-ker.py --model1 4 --model2 1 --n n --m 1000 --d 2 --t t
```


### GMDDNet

- Related codes: `my_gmdd_correct.ipynb` `my_gmdd_correct.py`
- Command: 
```bash
python my_gmdd_correct.py --model1 4 --model2 1 --n n --d 2 --t t
```


### StoNet

- Related codes: `my_stonet_all-4.ipynb` `my_stonet_all-4.py`
- Command: 
```bash
python my_stonet_all-4.py --model1 4 --model2 1 --n n --r 25 --d 2 --t t
```

### GSIR

- Related codes: `test-gsir.R`
- Command: 
```bash
Rscript test-gsir.R n t 1
```

### Result summarization

- File: `results-summary-model4-1-final.R`


















## Simulation - Model E-IV

- Folder: `simulation-model-E-IV`
- Note: In all commands, `n` is replaced by sample size (1000, ..., 8000); `t` is replaced by the index of the data (1, ..., 100).

### BENN-1

- Related codes: `benn-model4-poly.ipynb` `benn-model4-poly.py`
- Preliminary analysis codes: `benn-prelim-model4-2-1.ipynb`
- Command: 
```bash
python benn-model4-poly.py --model1 4 --model2 2 --n n --m 1 --d 1 --t t --ep 100
```


### BENN-2

- Related codes: `benn-model4-poly.ipynb` `benn-model4-poly.py`
- Preliminary analysis codes: `benn-prelim-model4-2-2.ipynb`
- Command: 
```bash
python benn-model4-poly.py --model1 4 --model2 2 --n n --m 2 --d 1 --t t --ep 100
```


### BENN-1000

- Related codes: `benn-all-ens-m-ker.ipynb` `benn-all-ens-m-ker.py`
- Preliminary analysis codes: `benn-prelim-model4-2.ipynb`
- Command: 
```bash
python benn-all-ens-m-ker.py --model1 4 --model2 2 --n n --m 1000 --d 1 --t t
```


### GMDDNet

- Related codes: `my_gmdd_correct.ipynb` `my_gmdd_correct.py`
- Command: 
```bash
python my_gmdd_correct.py --model1 4 --model2 2 --n n --d 1 --t t
```


### StoNet

- Related codes: `my_stonet_all-4.ipynb` `my_stonet_all-4.py`
- Command: 
```bash
python my_stonet_all-4.py --model1 4 --model2 2 --n n --r 25 --d 1 --t t
```

### GSIR

- Related codes: `test-gsir-1.R`
- Command: 
```bash
Rscript test-gsir-1.R n t 2
```

### Result summarization

- File: `results-summary-model4-2-final.R`








## Simulation - Sensntivity analysis

- Folder: `sensitivity-analysis`
- Related codes: `benn-all-ens-m-ker-ep.ipynb` `benn-all-ens-m-ker-ep.py`
- Preliminary analysis codes: all codes in `prelim-analysis` folder
- Commands for Model D-IV: 
```bash
python benn-all-ens-m-ker-ep.py --model1 4 --model2 1 --n 5000 --l1 2 --r1 50 --l2 1 --r2 2000 --m 1000 --ep 150 --caseid 0 --d 2 --t t
python benn-all-ens-m-ker-ep.py --model1 4 --model2 1 --n 5000 --l1 1 --r1 100 --l2 1 --r2 2000 --m 1000 --ep 150 --caseid 1 --d 2 --t t
python benn-all-ens-m-ker-ep.py --model1 4 --model2 1 --n 5000 --l1 3 --r1 100 --l2 1 --r2 2000 --m 1000 --ep 100 --caseid 2 --d 2 --t t
python benn-all-ens-m-ker-ep.py --model1 4 --model2 1 --n 5000 --l1 3 --r1 50 --l2 1 --r2 2000 --m 1000 --ep 150 --caseid 3 --d 2 --t t
python benn-all-ens-m-ker-ep.py --model1 4 --model2 1 --n 5000 --l1 2 --r1 50 --l2 1 --r2 1000 --m 500 --ep 150 --caseid 4 --d 2 --t t
python benn-all-ens-m-ker-ep.py --model1 4 --model2 1 --n 5000 --l1 1 --r1 100 --l2 1 --r2 1000 --m 500 --ep 100 --caseid 5 --d 2 --t t
python benn-all-ens-m-ker-ep.py --model1 4 --model2 1 --n 5000 --l1 3 --r1 100 --l2 1 --r2 1000 --m 500 --ep 100 --caseid 6 --d 2 --t t
python benn-all-ens-m-ker-ep.py --model1 4 --model2 1 --n 5000 --l1 3 --r1 50 --l2 1 --r2 1000 --m 500 --ep 150 --caseid 7 --d 2 --t t
python benn-all-ens-m-ker-ep.py --model1 4 --model2 1 --n 5000 --l1 2 --r1 50 --l2 2 --r2 1000 --m 500 --ep 150 --caseid 8 --d 2 --t t
python benn-all-ens-m-ker-ep.py --model1 4 --model2 1 --n 5000 --l1 1 --r1 100 --l2 2 --r2 1000 --m 500 --ep 100 --caseid 9 --d 2 --t t
python benn-all-ens-m-ker-ep.py --model1 4 --model2 1 --n 5000 --l1 3 --r1 100 --l2 2 --r2 1000 --m 500 --ep 100 --caseid 10 --d 2 --t t
python benn-all-ens-m-ker-ep.py --model1 4 --model2 1 --n 5000 --l1 3 --r1 50 --l2 2 --r2 1000 --m 500 --ep 150 --caseid 11 --d 2 --t t
```
- Commands for Model E-IV: 
```bash
python benn-all-ens-m-ker-ep.py --model1 4 --model2 2 --n 5000 --l1 2 --r1 50 --l2 1 --r2 2000 --m 1000 --ep 150 --caseid 0 --d 1 --t t
python benn-all-ens-m-ker-ep.py --model1 4 --model2 2 --n 5000 --l1 1 --r1 100 --l2 1 --r2 2000 --m 1000 --ep 100 --caseid 1 --d 1 --t t
python benn-all-ens-m-ker-ep.py --model1 4 --model2 2 --n 5000 --l1 3 --r1 100 --l2 1 --r2 2000 --m 1000 --ep 100 --caseid 2 --d 1 --t t
python benn-all-ens-m-ker-ep.py --model1 4 --model2 2 --n 5000 --l1 3 --r1 50 --l2 1 --r2 2000 --m 1000 --ep 100 --caseid 3 --d 1 --t t
python benn-all-ens-m-ker-ep.py --model1 4 --model2 2 --n 5000 --l1 2 --r1 50 --l2 1 --r2 1000 --m 500 --ep 150 --caseid 4 --d 1 --t t
python benn-all-ens-m-ker-ep.py --model1 4 --model2 2 --n 5000 --l1 1 --r1 100 --l2 1 --r2 1000 --m 500 --ep 100 --caseid 5 --d 1 --t t
python benn-all-ens-m-ker-ep.py --model1 4 --model2 2 --n 5000 --l1 3 --r1 100 --l2 1 --r2 1000 --m 500 --ep 100 --caseid 6 --d 1 --t t
python benn-all-ens-m-ker-ep.py --model1 4 --model2 2 --n 5000 --l1 3 --r1 50 --l2 1 --r2 1000 --m 500 --ep 100 --caseid 7 --d 1 --t t
python benn-all-ens-m-ker-ep.py --model1 4 --model2 2 --n 5000 --l1 2 --r1 50 --l2 2 --r2 1000 --m 500 --ep 100 --caseid 8 --d 1 --t t
python benn-all-ens-m-ker-ep.py --model1 4 --model2 2 --n 5000 --l1 1 --r1 100 --l2 2 --r2 1000 --m 500 --ep 100 --caseid 9 --d 1 --t t
python benn-all-ens-m-ker-ep.py --model1 4 --model2 2 --n 5000 --l1 3 --r1 100 --l2 2 --r2 1000 --m 500 --ep 100 --caseid 10 --d 1 --t t
python benn-all-ens-m-ker-ep.py --model1 4 --model2 2 --n 5000 --l1 3 --r1 50 --l2 2 --r2 1000 --m 500 --ep 100 --caseid 11 --d 1 --t t
```
- Note: `t` is replaced by the index of the data (1, ..., 100).
- Result summarization: `results-summary-sensitivity.R`





## Simulation - Model F-V

- Folder: `simulation-model-F-V`

### BENN

- Related codes: `benn-linear-ens-all.ipynb` `benn-linear-ens-all.py`
- Preliminary analysis codes: `benn-prelim-linear.ipynb`
- Command: 
```bash
python benn-linear-ens-all.py --model1 5 --model2 3 --n n --m m --d 2 --t t
```
- Note: `n` is replaced by sample size (1000, ..., 10000); `m` is replaced by number of transformations (1,2,3); `t` is replaced by the index of the data (1, ..., 100).
- Result summarization: `results-summary-model5-3-correct.R`



## Comparisons to GMDDNet - Model D-IV

- Folder: `gmdd-comp-D-IV`
- Related codes: `my_gmdd_correct-ymode.ipynb` `my_gmdd_correct-ymode.py`
- Command:
```bash
python my_gmdd_correct-ymode.py --model1 4 --model2 1 --n n --d 2 --y_mode y_mode --t t
```
- Note: `n` is replaced by sample size (1000, ..., 8000); `y_mode` is replaced by corresponding `Theta(v)` function (0,2,3,4,5); `t` is replaced by the index of the data (1, ..., 100).
- Result summarization: `results-summary-gmdd-model4-1-ymode.R`
- Side note: `results-model4-1.Rdata` is the intermediate file from Model D-IV.


## Comparisons to GMDDNet - Model G-IV

- Folder: `gmdd-comp-G-IV`
- Related codes: `my_gmdd_correct-ymode.ipynb` `my_gmdd_correct-ymode.py` `benn-all-ens-m-ker-ep100.ipynb` `benn-all-ens-m-ker-ep100.py`
- Command for BENN:
```bash
python benn-all-ens-m-ker-ep100.py --model1 4 --model2 2 --n n --m 1000 --d 2 --t t
```
- Command for GMDDNet:
```bash
python my_gmdd_correct-ymode.py --model1 4 --model2 4 --n n --d 2 --y_mode y_mode --t t
```
- Note: `n` is replaced by sample size (1000, ..., 8000); `y_mode` is replaced by corresponding `Theta(v)` function (0,2,3,4,5); `t` is replaced by the index of the data (1, ..., 100).
- Result summarization: `results-summary-gmdd-model4-4-ymode.R`






## Application 1: Superconductivty dataset

### Dataset information

- Original data: https://archive.ics.uci.edu/dataset/464/superconductivty+data
- Dataset file: `application-1/train.csv` (downloaded from the original source)

### Related codes

- Folder: `application-1`
- BENN: `benn-all-application-1-x.ipynb` (`x` is replaced by the value of m)
- GMDDNet: `gmdd-application-1-correct.ipynb`
- StoNet: `my_stonet_application-1.ipynb`
- GSIR: `test-gsir-application-1.R` `application_graph_gsir-1.ipynb`
- Running time summarization: `application-1-result-summary.R`

### Note for running

- Run `benn-all-application-1-1.ipynb` first to split the training and testing data.
- Some intermediate files are also provided in the folder.





## Application 2: Geographical origin of music dataset 

### Dataset information

- Original data: https://archive.ics.uci.edu/dataset/315/geographical+original+of+music
- Dataset file: `application-2/default_features_1059_tracks.txt` (downloaded from the original source)

### Related codes

- Folder: `application-2`
- BENN: `benn-all-application-2-x.ipynb` (`x` is replaced by the value of m)
- GMDDNet: `gmdd-application-2-correct.ipynb`
- StoNet: `my_stonet_application.ipynb`
- GSIR: `test-gsir-application.R` `application_graph_gsir.ipynb`

### Note for running

- Run `benn-all-application-2-1.ipynb` first to split the training and testing data.
- Some intermediate files are also provided in the folder.