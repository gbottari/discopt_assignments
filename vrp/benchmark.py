# -*- coding: utf-8 -*-

import sys
sys.path.append('..')

import os
import time
from tools.solver_tools import MultiSolver
from vrp.solver_tools import *


def get_solver():
    #solver = MultiSolver(timeout=None, solvers=[RandomVRPSolver()])
    solver = LS2OptVRPSolver(max_iters=10000)
    #solver = SASolver(improvement_limit=3000)
    return solver


def run_benchmark():
    problems = ('vrp_16_3_1', 'vrp_26_8_1', 'vrp_51_5_1', 'vrp_101_10_1', 'vrp_200_16_1', 'vrp_421_41_1')
    solutions = []
    times = []

    print('Benchmarking solver {}'.format(get_solver()))

    for i, filename in enumerate(problems):
        with open(os.path.join('data', filename)) as f:
            raw_data = f.read()

        print('Solving Problem {}...'.format(i + 1))
        t0 = time.time()
        solver = get_solver()
        solution = solver.solve(raw_data)
        t1 = time.time()
        solutions.append(solution)
        times.append(t1 - t0)

    print('#1 Problem #2 Solution #3 Duration (seconds)')
    for i in range(len(solutions)):
        solution = solutions[i]
        t = times[i]
        solution_value = '{:8.1f}'.format(solution.get_value()) if solution and solution.is_feasible() else 'infeasible'
        print('{} {}     {:.1f}'.format(i + 1, solution_value, t))


if __name__ == '__main__':
    run_benchmark()
