#!/usr/bin/python
# -*- coding: utf-8 -*-

from solver_tools import FifoKSSolver


def solve_it(input_data):
    solver = FifoKSSolver()
    solution = solver.solve(input_data)
    return solution.serialize()


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)')

