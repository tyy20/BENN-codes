## Simulation - Model D-IV (Sections 6 and S.6.3)

- Folder: `simulation-model-D-IV`
- **Note**: In all commands, `n` is replaced by sample size `1000, ..., 8000`; `t` is replaced by the index of the data `1, ..., 100`

### Methodology implementation

- **BENN-1**: 
  -  `benn-prelim-model4-1-1.ipynb` for preliminary analysis
  - Run `benn-model4-poly.ipynb` or `benn-model4-poly.py` using the following command:
```bash
python benn-model4-poly.py --model1 4 --model2 1 --n n --m 1 --d 2 --t t --ep 100
```

- **BENN-2**:
  - Run `benn-prelim-model4-1-2.ipynb` for preliminary analysis
  - Run `benn-model4-poly.ipynb` or `benn-model4-poly.py` using the following command:
```bash
python benn-model4-poly.py --model1 4 --model2 1 --n n --m 2 --d 2 --t t --ep 100
```


- **BENN-1000**:
  - Run `benn-prelim-model4-1.ipynb` for preliminary analysis
  - Run `benn-all-ens-m-ker.ipynb` or `benn-all-ens-m-ker.py` using the following command:
```bash
python benn-all-ens-m-ker.py --model1 4 --model2 1 --n n --m 1000 --d 2 --t t
```


- **GMDDNet**
  - Run `my_gmdd_correct.ipynb` or `my_gmdd_correct.py` using the following command:
```bash
python my_gmdd_correct.py --model1 4 --model2 1 --n n --d 2 --t t
```


- **StoNet**
  - Run `my_stonet_all-4.ipynb` or `my_stonet_all-4.py` using the following command:
```bash
python my_stonet_all-4.py --model1 4 --model2 1 --n n --r 25 --d 2 --t t
```

- **GSIR**
  - Run `test-gsir.R` using the following command:
```bash
Rscript test-gsir.R n t 1
```

### Result summarization

- Reproduce Table 1: Run `results-summary-model4-1-final.R`
- Reproduce Table S.3: Run `time-summary-model4-1.R`


### Remarks

- Some `.py` files are also provided in case it takes too long to run the `.ipynb` file.
- Directories need to be changed accordingly in each code.
- Running times for each method under each sample size are summarized in Table S.3.






