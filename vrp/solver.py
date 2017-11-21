#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
sys.path.append('..')

from tools.solver_tools import MultiSolver, SolverManager
from vrp.solver_tools import *


def solve_it(input_data):
    solver = MultiSolver(timeout=300, solvers=[
        ILSVRPSolver(max_diving_iters=800, max_dives=2000),
        SASolver(t_min=3, t_max=100000.0, alpha=0.99996, improvement_limit=200000),
    ])
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

        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/vrp_5_4_1)')

