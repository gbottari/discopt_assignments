import unittest
import os
from coloring.solver_tools import *


def get_all_problem_filenames():
    return (f for f in os.listdir('data') if os.path.isfile(os.path.join('data', f)))


def get_problem_by_filename(filename):
    with open(os.path.join('data', filename)) as f:
        return GCSolver()._parse(f.read())


def get_problem_1():
    return get_problem_by_filename('gc_4_1')


class TestSolver(unittest.TestCase):
    def _check_solution(self, solution, optimal):
        self.assertIsInstance(solution, GCSolution)
        self.assertTrue(solution.is_feasible())
        self.assertEqual(solution.is_optimal(), optimal)
        self.assertEqual(len(solution.node_colors), len(solution.problem.nodes))
        self.assertLessEqual(max(solution.node_colors), len(solution.problem.nodes) - 1)

    def test_solution_reading(self):
        problem = get_problem_by_filename('gc_4_1')
        self.assertEqual(problem.sorted_edges, [[0, 1], [1, 2, 3], [2], [3]])

    def test_solver_consistency(self):
        for solver in (TrivialGCSolver(), GreedyChangeUntilSatisfy()):
            for problem in (get_problem_by_filename(fn) for fn in list(get_all_problem_filenames())[-3:]):
                solution = solver._solve(problem)
                self._check_solution(solution, optimal=False)

    def test_greedy_spanning(self):
        problem = get_problem_by_filename('gc_4_1')
        solver = GreedyChangeUntilSatisfy()
        solution = solver._solve(problem)
        self._check_solution(solution, optimal=False)

    def test_trivial_solver_unfeasible(self):
        solver = TrivialGCSolver()
        solution = solver._solve(get_problem_1())
        mid = len(solution.problem.input_edges) // 2
        solution.node_colors[solution.problem.input_edges[mid][0]] = solution.node_colors[solution.problem.input_edges[mid][1]]
        self.assertFalse(solution.is_feasible())
