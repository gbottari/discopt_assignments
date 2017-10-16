#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
sys.path.append('..')

from tools.solver_tools import MultiSolver, SolverManager
from tsp.solver_tools import *


def solve_it(input_data):
    # Modify this code to run your optimization algorithm
    solver = MultiSolver(timeout=2 * 60, solvers=[InputOrderTSPSolver(), Greedy2OptTSPSolver(1000000)])
    mgr = SolverManager()
    return mgr.solve(input_data, solver)


import sys

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/tsp_51_1)')

