## Simulation - Model F-V

- Folder: `simulation-model-F-V`
- Note: In all commands, `n` is replaced by sample size `1000, ..., 10000`; `m` is replaced by number of transformations `1,2,3`; `t` is replaced by the index of the data `1, ..., 100`.

### BENN

- Related codes: `benn-linear-ens-all.ipynb` `benn-linear-ens-all.py`
- Preliminary analysis codes: `benn-prelim-linear.ipynb`
- Command: 
```bash
python benn-linear-ens-all.py --model1 5 --model2 3 --n n --m m --d 2 --t t
```
- Result summarization: `results-summary-model5-3-correct.R`
