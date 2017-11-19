# -*- coding: utf-8 -*-

import sys
sys.path.append('..')

import os
from tools.solver_tools import MultiSolver, SolverManager
from vrp.solver_tools import *


def get_solver():
    #solver = MultiSolver(timeout=None, solvers=[RandomVRPSolver()])
    solver = LS2OptVRPSolver()
    return solver


def run_benchmark():
    problems = ('vrp_16_3_1', 'vrp_26_8_1', 'vrp_51_5_1', 'vrp_101_10_1', 'vrp_200_16_1', 'vrp_421_41_1')
    solutions = []

    print('Benchmarking solver {}'.format(get_solver()))

    for i, filename in enumerate(problems):
        with open(os.path.join('data', filename)) as f:
            raw_data = f.read()

        print('Solving Problem {}...'.format(i + 1))
        solver = get_solver()
        solution = solver.solve(raw_data)
        solutions.append(solution)

    print('# Problem # Solution')
    for i, solution in enumerate(solutions):
        solution_value = '{:.1f}'.format(solution.get_value()) if solution and solution.is_feasible() else 'infeasible'
        print('{}  {}'.format(i + 1, solution_value))


if __name__ == '__main__':
    run_benchmark()
