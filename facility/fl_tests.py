import unittest
import os
from facility.solver_tools import *


def get_problem_by_filename(filename):
    with open(os.path.join('data', filename)) as f:
        return FLSolver()._parse(f.read())


def get_easy_problem():
    return get_problem_by_filename('fl_3_1')


class TestSolver(unittest.TestCase):
    def test_get_easy_problem_returns_a_valid_problem(self):
        problem = get_easy_problem()
        self.assertIsInstance(problem, FLProblem)
        self.assertEqual(len(problem.facilities), 3)
        self.assertEqual(len(problem.customers), 4)

        self.assertEqual(problem.facilities[0].index, 0)
        self.assertEqual(problem.facilities[0].setup_cost, 100)
        self.assertEqual(problem.facilities[0].capacity, 100)
        self.assertEqual(problem.facilities[0].location.x, 1065.0)
        self.assertEqual(problem.facilities[0].location.y, 1065.0)

        self.assertEqual(problem.facilities[1].index, 1)
        self.assertEqual(problem.facilities[1].setup_cost, 100)
        self.assertEqual(problem.facilities[1].capacity, 100)
        self.assertEqual(problem.facilities[1].location.x, 1062.0)
        self.assertEqual(problem.facilities[1].location.y, 1062.0)

        self.assertEqual(problem.facilities[2].index, 2)
        self.assertEqual(problem.facilities[2].setup_cost, 100)
        self.assertEqual(problem.facilities[2].capacity, 500)
        self.assertEqual(problem.facilities[2].location.x, 0.0)
        self.assertEqual(problem.facilities[2].location.y, 0.0)

        self.assertEqual(problem.customers[0].index, 0)
        self.assertEqual(problem.customers[0].demand, 50)
        self.assertEqual(problem.customers[0].location.x, 1397.0)
        self.assertEqual(problem.customers[0].location.y, 1397.0)
        self.assertEqual(problem.customers[0].prefs, [0, 1, 2])
        self.assertEqual(len(problem.customers[0].dists), 3)

        self.assertEqual(problem.customers[1].index, 1)
        self.assertEqual(problem.customers[1].demand, 50)
        self.assertEqual(problem.customers[1].location.x, 1398.0)
        self.assertEqual(problem.customers[1].location.y, 1398.0)
        self.assertEqual(problem.customers[1].prefs, [0, 1, 2])
        self.assertEqual(len(problem.customers[1].dists), 3)

        self.assertEqual(problem.customers[2].index, 2)
        self.assertEqual(problem.customers[2].demand, 75)
        self.assertEqual(problem.customers[2].location.x, 1399.0)
        self.assertEqual(problem.customers[2].location.y, 1399.0)
        self.assertEqual(problem.customers[2].prefs, [0, 1, 2])
        self.assertEqual(len(problem.customers[2].dists), 3)

        self.assertEqual(problem.customers[3].index, 3)
        self.assertEqual(problem.customers[3].demand, 75)
        self.assertEqual(problem.customers[3].location.x, 586.0)
        self.assertEqual(problem.customers[3].location.y, 586.0)
        self.assertEqual(problem.customers[3].prefs, [1, 0, 2])
        self.assertEqual(len(problem.customers[3].dists), 3)

    def test_is_optimal(self):
        solution = FLSolution(None)
        solution.optimal = False
        self.assertFalse(solution.is_optimal())
        solution.optimal = True
        self.assertTrue(solution.is_optimal())

    def test_get_value(self):
        problem = get_easy_problem()
        solution = FLSolution(problem)
        solution.selections = [1, 1, 0, 2]
        self.assertAlmostEqual(solution.get_value(), 2550.013, places=2)

    def test_feasible_solution_is_detected(self):
        problem = get_easy_problem()
        solution = FLSolution(problem)
        solution.selections = [1, 1, 0, 2]
        self.assertTrue(solution.is_feasible())

    def test_infeasible_solution_is_detected(self):
        problem = get_easy_problem()
        solution = FLSolution(problem)
        solution.selections = [1, 1, 1, 1]
        self.assertFalse(solution.is_feasible())

    def test_serialize(self):
        problem = get_easy_problem()
        solution = FLSolution(problem)
        solution.selections = [1, 1, 0, 2]
        serialized = solution.serialize()
        expected = "{} {}\n{} {} {} {}".format(2550.013, 0, 1, 1, 0, 2)
        self.assertEqual(serialized, expected)

    def test_is_better(self):
        problem = get_easy_problem()
        solution_1 = FLSolution(problem)
        solution_2 = FLSolution(problem)
        solution_1.selections = [1, 1, 0, 2]
        solution_2.selections = [2, 2, 2, 2]
        self.assertLess(solution_1.get_value(), solution_2.get_value())
        self.assertTrue(solution_1.is_better(solution_2))
        self.assertFalse(solution_2.is_better(solution_1))

        # when optimal, ignore the value
        solution_2.optimal = True
        self.assertTrue(solution_2.is_better(solution_1))

    def test_trivial_solver_returns_feasible(self):
        problem = get_easy_problem()
        solver = TrivialFLSolver()
        solution = solver._solve(problem)
        self.assertTrue(solution.is_feasible())

    