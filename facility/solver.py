#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
sys.path.append('..')

from tools.solver_tools import MultiSolver, SolverManager
from facility.solver_tools import *


def solve_it(input_data):
    # SASolver(alpha=0.999995, t0=10000, improvement_limit=1000000)
    solver = MultiSolver(timeout=10 * 60, solvers=[GreedyDistSolver(), GreedyPrefSolver(), TrivialFLSolver(), RandSolver(),
                                                   FLMipSplitter(max_vars=100000)])
    #solver = MultiSolver(timeout=5 * 60, solvers=[FLMipSplitter(max_vars=100000)])
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
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/fl_16_2)')

