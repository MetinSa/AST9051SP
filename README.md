# AST9051SP - Statistical Methods in Computational Cosmology

This repository contains the project deliverables for the special curriculum course AST9051SP - Statistical Methods in Computational Cosmology, which i took during my last year as a PhD student at UiO. 

## Main deliverables:
- Jupyter notebook with exercises from [Data analysis recipes: Fitting a model to data](https://arxiv.org/abs/1008.4686)
- Project report in `report_AST9051SP.pdf`
- Source code to reproduce results in report in `src/`


## Steps to reproduce results in the report:
- Create a new Python environment (Python >= 3.8)
- Enter the environment and install the following dependencies:
    - astropy >= 5.0.1
    - numpy
    - scipy
    - jplephem
    - matplotlib
    - h5py
- Clone this repository (Note: it contains a 77MB data file)
- Open the Run `src/main.py` file and turn on/off the various functions in the `if __name__== "__main__":` block at the end of the file. Then run `python main.py` in while being in the `src/` directory.