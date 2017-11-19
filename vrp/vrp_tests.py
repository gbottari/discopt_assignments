import unittest
import os
from vrp.solver_tools import *

def get_problem_by_filename(filename):
    with open(os.path.join('data', filename)) as f:
        return VRPSolver()._parse(f.read())


def get_easy_problem():
    return get_problem_by_filename('vrp_5_4_1')


def get_all_problems():
    for fn in ('vrp_16_3_1', 'vrp_26_8_1', 'vrp_51_5_1', 'vrp_101_10_1', 'vrp_200_16_1', 'vrp_421_41_1'):
        yield get_problem_by_filename(fn)


class TestSolver(unittest.TestCase):
    def _check_customer(self, problem, index, demand, x, y):
        self.assertEqual(problem.customers[index].index, index)
        self.assertEqual(problem.customers[index].demand, demand)
        self.assertEqual(problem.customers[index].location.x, x)
        self.assertEqual(problem.customers[index].location.y, y)

    def test_parse_problem(self):
        problem = get_problem_by_filename('vrp_5_4_1')
        self.assertEqual(len(problem.customers), 5)  # 4 + 1 depot
        self.assertEqual(problem.max_vehicles, 4)
        self._check_customer(problem, 0, 0, 0, 0)
        self._check_customer(problem, 1, 3, 0, 10)
        self._check_customer(problem, 2, 3, -10, 10)
        self._check_customer(problem, 3, 3, 0, -10)
        self._check_customer(problem, 4, 3, 10, -10)

    def test_feasible_solution(self):
        problem = get_problem_by_filename('vrp_5_4_1')
        solution = VRPSolution(problem)
        solution.big_tour = [0, 1, 2, 3, -1, 4, -2, -3, 0]
        self.assertAlmostEqual(solution.get_value(), 80.6, places=1)
        self.assertTrue(solution.is_feasible())
        self.assertFalse(solution.is_optimal())

    def test_infeasible_solution(self):
        problem = get_problem_by_filename('vrp_5_4_1')
        solution = VRPSolution(problem)

        # capacity exceeded
        solution.big_tour = [0, 1, 2, 3, 4, -1, -2, -3, 0]
        self.assertFalse(solution.is_feasible())
        solution.big_tour = solution.big_tour[::-1]
        self.assertFalse(solution.is_feasible())

        # vehicle omission is not ok
        solution.big_tour = [0, 1, 2, 3, -1, 4, 0]
        self.assertFalse(solution.is_feasible())

    def test_solution_serialization(self):
        problem = get_problem_by_filename('vrp_5_4_1')
        solution = VRPSolution(problem)
        solution.big_tour = [0, 1, 2, 3, -1, 4, -2, -3, 0]
        serialized = solution.serialize()
        self.assertEqual(serialized, "80.6 0\n0 1 2 3 0\n0 4 0\n0 0\n0 0")

    def test_random_solver_feasible(self):
        problem = get_easy_problem()
        solution = RandomVRPSolver()._solve(problem)
        self.assertTrue(solution.is_feasible())