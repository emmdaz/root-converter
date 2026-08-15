Usage: 

- Requires `uproot`, `awkward`, `pandas`, `numpy`. Install with: 

```
pip install uproot awkward pandas numpy
```

- If you're not sure of the branch names look inside the .ROOT file first: 

```
python root_to_lhco_csv.py your_file.root --list-branches
```

- The convertion is done running:

```
python root_to_lhco_csv.py your_file.root output.csv
```

An example of the resulting .csv file is given in `example.ipynb`