#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
sys.path.append('..')

from tools.solver_tools import solve_and_serialize
from solver_tools import *


def solve_it(input_data):
    solver = GreedyMaxValueKSSolver()
    return solve_and_serialize(input_data, solver)


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)')

