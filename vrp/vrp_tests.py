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

    #@unittest.skip('')
    def test_ls2opt_feasible(self):
        problem = get_problem_by_filename('vrp_16_3_1')
        solution = LS2OptVRPSolver(max_iters=10000)._solve(problem)
        self.assertTrue(solution.is_feasible())

    def test_sasolver_feasible(self):
        problem = get_problem_by_filename('vrp_16_3_1')
        solution = SASolver(improvement_limit=100)._solve(problem)
        self.assertTrue(solution.is_feasible())

    def test_next_customer_in_tour(self):
        problem = get_easy_problem()
        solution = VRPSolution(problem)
        solution.from_big_tour([0, 1, 2, -1, 3, 0])
        # tour 1: 0, 1, 2
        # tour 2: -1, 3

        self.assertEqual(solution.next_c_i_in_tour(0), 1)  # 0
        self.assertEqual(solution.next_c_i_in_tour(1), 2)  # 1
        self.assertEqual(solution.next_c_i_in_tour(2), 0)  # 2
        self.assertEqual(solution.next_c_i_in_tour(3), 3)  # -1
        self.assertEqual(solution.next_c_i_in_tour(4), 0)  # 3
        self.assertEqual(solution.next_c_i_in_tour(5), 3)  # 0

        self.assertEqual(solution.prev_c_i_in_tour(0), 2)  # 0
        self.assertEqual(solution.prev_c_i_in_tour(1), 0)  # 1
        self.assertEqual(solution.prev_c_i_in_tour(2), 1)  # 2
        self.assertEqual(solution.prev_c_i_in_tour(3), 0)  # -1
        self.assertEqual(solution.prev_c_i_in_tour(4), -1) # 3
        self.assertEqual(solution.prev_c_i_in_tour(5), 3)  # 0

    def test_demand_check(self):
        problem = get_easy_problem()
        problem.capacity = 6

        solver = LS2OptVRPSolver()
        solution = VRPSolution(problem)
        #                       0  1  2   3  4  5   6   7  8
        solution.from_big_tour([0, 1, 2, -1, 3, 4, -2, -3, 0])
        #                       0  0  0   1  1  1   2   3  3
        self.assertTrue(solution.is_feasible())
        self.assertTrue(solver._check_demand(solution, 1, 2))
        #  0  0  0  0   1  1   2   2  3
        # [0, 1, 4, 3, -1, 2, -2, -3, 0]
        self.assertFalse(solver._check_demand(solution, 2, 5))

        #                       0  1  2  3   4  5   6   7  8
        solution.from_big_tour([0, 1, 2, 3, -1, 4, -2, -3, 0])
        #                       0  0  0  0   1  1   2   3  3
        # [0, 1, 4, -1, 3, 2, -2, -3, 0]
        self.assertTrue(solver._check_demand(solution, 2, 5))

        problem = get_problem_by_filename('vrp_16_3_1')
        solution = VRPSolution(problem)
        solution.from_big_tour([0, 13, 15, 7, 14, 6, -1, 5, 11, 2, 4, 10, -2, 8, 9, 3, 12, 1, 0])
        #                       0   1   2  3   4  5   6  7   8  9
        self.assertFalse(solver._check_demand(solution, 1, 9))

    def test_calc_value(self):
        problem = get_problem_by_filename('vrp_5_4_1')
        solution = VRPSolution(problem)
        solver = LS2OptVRPSolver()
        dist = problem.dist
        get_c = problem.get_customer

        solution.from_big_tour([0, 2, 1, 3, -1, 4, -2, -3, 0])
        value = solver.calc_sol_value(solution, solution.get_value(), 1, 2)
        self.assertAlmostEqual(value, 80.6, places=1)

        solution.from_big_tour([0, 1, 3, 2, -1, 4, -2, -3, 0])
        value = solver.calc_sol_value(solution, solution.get_value(), 2, 3)
        self.assertAlmostEqual(value, 80.6, places=1)

        solution.from_big_tour([0, 1, 3, 2, -1, 4, -2, -3, 0])
        value = solver.calc_sol_value(solution, solution.get_value(), 3, 5)
        expected_value = solution.get_value() \
                         - dist(get_c(3).location, get_c(2).location) \
                         + dist(get_c(3).location, get_c(4).location) \
                         - dist(get_c(4).location, get_c(-2).location) \
                         + dist(get_c(2).location, get_c(-2).location)
        solution.from_big_tour([0, 1, 3, 4, -1, 2, -2, -3, 0])
        actual_value = solution.get_value()
        self.assertEqual(expected_value, actual_value)
        self.assertAlmostEqual(value, actual_value, places=1)