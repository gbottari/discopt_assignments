# -*- coding: utf-8 -*-

import sys
sys.path.append('..')

import os
import time
from tools.solver_tools import MultiSolver
from vrp.solver_tools import *


def get_solver():
    #solver = MultiSolver(timeout=None, solvers=[RandomVRPSolver()])
    #solver = LS2OptVRPSolver(max_iters=1000000)
    #solver = SASolver(improvement_limit=30000)
    solver = SASolver(t0=100000.0, alpha=0.99996, improvement_limit=200000)
    return solver


def run_benchmark():
    problems = ('vrp_16_3_1', 'vrp_26_8_1', 'vrp_51_5_1', 'vrp_101_10_1', 'vrp_200_16_1', 'vrp_421_41_1')
    solutions = []
    times = []

    print('Benchmarking solver {}'.format(get_solver()))

    def print_summary(solution, t):
        solution_value = '{:8.1f}'.format(solution.get_value()) if solution and solution.is_feasible() else 'infeasible'
        print('{} {}     {:.1f}'.format(i + 1, solution_value, t))

    for i, filename in enumerate(problems):
        with open(os.path.join('data', filename)) as f:
            raw_data = f.read()

        print('Solving Problem {}...'.format(i + 1))
        t0 = time.time()
        solver = get_solver()
        solution = solver.solve(raw_data)
        t1 = time.time()
        solutions.append(solution)
        t = t1 - t0
        times.append(t)
        print_summary(solution, t)

    print('\n')
    print('Benchmarking solver {}'.format(get_solver()))
    print('#1 Problem #2 Solution #3 Duration (seconds)')
    for i in range(len(solutions)):
        solution = solutions[i]
        t = times[i]
        print_summary(solution, t)


if __name__ == '__main__':
    run_benchmark()
