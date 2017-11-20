# -*- coding: utf-8 -*-

import sys
sys.path.append('..')

import os
import time
import datetime
from tools.solver_tools import MultiSolver
from vrp.solver_tools import *


def get_solver():
    #solver = MultiSolver(timeout=None, solvers=[RandomVRPSolver()])
    #solver = LS2OptVRPSolver(max_iters=1000000)
    #solver = SASolver(improvement_limit=30000)
    #solver = SASolver(t_min=3, t_max=100000.0, alpha=0.99996, improvement_limit=200000)
    #solver = ILSVRPSolver(max_dives=2000, max_diving_iters=800)
    solver = ILSVRPSolver2(max_failed_dives=10, depth_multiplier=1.1, refinement_loops=30000, initial_depth=1000)
    #solver = ILSVRPSolver2(max_failed_dives=200, depth_multiplier=1.05, refinement_loops=300000, initial_depth=1000)
    #solver = ILSVRPExplorerSolver(max_dives=100, max_diving_iters=3000, max_good_sols=100, max_gap=1.5)
    #solver = ILSVRPBaggerSolver(max_dives=4000, max_diving_iters=500, max_good_sols=100, max_gap=1.5)
    #solver = ILSVRPSolver(max_dives=8, max_diving_iters=200 * 500)
    return solver


def run_benchmark():
    problems = ('vrp_16_3_1', 'vrp_26_8_1', 'vrp_51_5_1', 'vrp_101_10_1', 'vrp_200_16_1', 'vrp_421_41_1')
    #problems = ('vrp_26_8_1',)
    #problems = ('vrp_421_41_1',)
    solutions = []
    times = []

    print('Benchmarking solver {}'.format(get_solver()))

    def print_summary(solution, t):
        solution_value = '{:8.1f}'.format(solution.get_value()) if solution and solution.is_feasible() else 'infeasible'
        print('{} {}     {:.1f}'.format(i + 1, solution_value, t))

    for i, filename in enumerate(problems):
        with open(os.path.join('data', filename)) as f:
            raw_data = f.read()

        print('[{}] Solving Problem {}...'.format(datetime.datetime.now().time(), i + 1))
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
