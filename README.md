# CSP Sudoku Solver

A Python-based Sudoku solver using Constraint Satisfaction Problems (CSP).

## Features

- Backtracking search
- Minimum Remaining Values (MRV) heuristic
- Degree heuristic
- Least Constraining Value (LCV)
- Forward checking
- Sudoku puzzle generation
- Benchmarking different solving strategies
- Automated tests
- Optional AI-powered explanations using the Anthropic API
- Optional Google OR-Tools CP-SAT comparison

## Technologies

Python, Constraint Satisfaction Problems (CSP), Backtracking, Heuristics, Anthropic API, Google OR-Tools.

## Usage

Run the solver:

```bash
python sudoku_csp_solver.py
```

Run the tests:

```bash
python sudoku_csp_solver.py --test
```

Run benchmarks:

```bash
python sudoku_csp_solver.py --bench
```

## Requirements

Python 3.10 or later.

Anthropic and OR-Tools are optional dependencies for their respective features.
