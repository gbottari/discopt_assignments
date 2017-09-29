#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
sys.path.append('..')

from coloring.solver_tools import *
from tools.solver_tools import MultiSolver, SolverManager


def solve_it(input_data):
    # Modify this code to run your optimization algorithm
    solver = MultiSolver(timeout=10 * 60, solvers=[GreedyChangeUntilSatisfy(), TrivialGCSolver()])
    mgr = SolverManager()
    return mgr.solve(input_data, solver)


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/gc_4_1)')

