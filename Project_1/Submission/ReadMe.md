### Submission structure

Structure of submitted directory is:

```
└── NGC
	├── implementations.py
	├── run.py
	├── proj1_helpers.py
	├── report.pdf
	└── ReadMe.md
```

Here `implementations.py` contains all the functions used during project, including preprocessing functions, algorithms functions (that were required to be realized due to the assignment) and their essential compounds, cross-validation functions.

`run.py` contains a final chosen method and creates submission.csv file at the end of it work. `run.py` file inherits `implementations.py` and uses functions written there to produse the result.

`proj1_helpers.py` consists of functions that help to load, predict and submit. Note that it was slightly changed from the one that was provided at the very beggining.

`report.pdf` is a report where is the procedure of obtaining our result is presented.
