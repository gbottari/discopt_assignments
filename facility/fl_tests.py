import unittest
import os
from facility.solver_tools import *


def get_problem_by_filename(filename):
    with open(os.path.join('data', filename)) as f:
        return FLSolver()._parse(f.read())


def get_easy_problem():
    return get_problem_by_filename('fl_3_1')


def get_all_problems():
    for fn in {'fl_25_2', 'fl_50_6', 'fl_100_7', 'fl_100_1', 'fl_200_7', 'fl_500_7', 'fl_1000_2', 'fl_2000_2'}:
        yield get_problem_by_filename(fn)


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

    def test_f_capacity_relaxation(self):
        problem = get_easy_problem()
        partial_solution = FLSolution(problem)
        partial_solution.selections = [None, None, 2, 2]
        lb = get_f_capacity_relaxation(partial_solution)
        prefs_0 = problem.customers[0].prefs[0]
        prefs_1 = problem.customers[1].prefs[0]
        setup_cost = sum(problem.facilities[f_i].setup_cost for f_i in frozenset([prefs_0, prefs_1, 2]))
        dist_cost = problem.dist(problem.facilities[prefs_0].location, problem.customers[0].location) + \
                    problem.dist(problem.facilities[prefs_1].location, problem.customers[1].location) + \
                    problem.dist(problem.facilities[2].location, problem.customers[2].location) + \
                    problem.dist(problem.facilities[2].location, problem.customers[3].location)
        self.assertAlmostEqual(lb, setup_cost + dist_cost)

    def test_solution2_initialization(self):
        problem = get_easy_problem()
        solution = FLSolution2(problem)
        self.assertEqual(solution.selections, [None, None, None, None])

    def test_solution_2_open_close_facility(self):
        problem = get_easy_problem()
        solution = FLSolution2(problem)
        solution.open_facility(0)
        self.assertAlmostEqual(solution.setup_cost, problem.facilities[0].setup_cost)
        self.assertEqual(solution.open_fs, {0})
        solution.close_facility(0)
        self.assertAlmostEqual(solution.setup_cost, 0)
        self.assertEqual(solution.open_fs, set())

    def test_solution_2_bind_tests(self):
        problem = get_easy_problem()
        solution = FLSolution2(problem)
        setup_cost = 0
        dist_cost = 0
        capacities = [f.capacity for f in problem.facilities]

        # bind first customer
        solution.bind_customer(0, 0)
        setup_cost += problem.facilities[0].setup_cost
        capacities[0] -= problem.customers[0].demand
        self.assertEqual(solution.selections[0], 0)
        self.assertAlmostEqual(solution.setup_cost, setup_cost)
        self.assertEqual(solution.open_fs, {0})
        self.assertEqual(solution.capacities, capacities)
        dist_cost += problem.dist(problem.facilities[0].location, problem.customers[0].location)
        self.assertAlmostEqual(solution.dist_cost, dist_cost)

        # bind another customer to the same facility
        solution.bind_customer(1, 0)
        dist_cost += problem.dist(problem.facilities[0].location, problem.customers[1].location)
        capacities[0] -= problem.customers[1].demand
        self.assertEqual(solution.selections[1], 0)
        self.assertAlmostEqual(solution.setup_cost, setup_cost)
        self.assertEqual(solution.open_fs, {0})
        self.assertEqual(solution.capacities, capacities)
        self.assertAlmostEqual(solution.dist_cost, dist_cost)

        # bind another customer to another facility
        solution.bind_customer(2, 1)
        setup_cost += problem.facilities[2].setup_cost
        dist_cost += problem.dist(problem.facilities[1].location, problem.customers[2].location)
        capacities[1] -= problem.customers[2].demand
        self.assertEqual(solution.selections[2], 1)
        self.assertAlmostEqual(solution.setup_cost, setup_cost)
        self.assertEqual(solution.open_fs, {0, 1})
        self.assertEqual(solution.capacities, capacities)
        self.assertAlmostEqual(solution.dist_cost, dist_cost)

        # move a customer to another facility leaving it non-empty
        solution.bind_customer(0, 1)
        dist_cost -= problem.dist(problem.facilities[0].location, problem.customers[0].location)
        dist_cost += problem.dist(problem.facilities[1].location, problem.customers[0].location)
        capacities[0] += problem.customers[0].demand
        capacities[1] -= problem.customers[0].demand
        self.assertAlmostEqual(solution.setup_cost, setup_cost)
        self.assertEqual(solution.open_fs, {0, 1})
        self.assertEqual(solution.capacities, capacities)
        self.assertAlmostEqual(solution.dist_cost, dist_cost)

        # move the last customer, leaving the facility empty
        solution.bind_customer(1, 1)
        setup_cost -= problem.facilities[0].setup_cost
        dist_cost -= problem.dist(problem.facilities[0].location, problem.customers[1].location)
        dist_cost += problem.dist(problem.facilities[1].location, problem.customers[1].location)
        capacities[0] += problem.customers[1].demand
        capacities[1] -= problem.customers[1].demand
        self.assertAlmostEqual(solution.setup_cost, setup_cost)
        self.assertEqual(solution.open_fs, {1})
        self.assertEqual(solution.capacities, capacities)
        self.assertAlmostEqual(solution.dist_cost, dist_cost)

    def test_df_bnb_is_feasible(self):
        problem = get_easy_problem()
        solver = DFBnBSolver()
        solution = solver._solve(problem)
        self.assertTrue(solution.is_feasible())
        self.assertAlmostEqual(solution.get_value(), 2545.771137048475)

    @unittest.skip('')
    def test_df_bnb_runs_fast(self):
        problem = get_problem_by_filename('fl_16_1')
        solver = DFBnBSolver()
        solution = solver._solve(problem)
        self.assertTrue(solution.is_feasible())

    def test_greedy_preference(self):
        problem = get_easy_problem()
        solver = GreedyPrefSolver()
        solution = solver._solve(problem)
        self.assertTrue(solution.is_feasible())

    @unittest.skip('very slow')
    def test_greedy_contest(self):
        solvers = [TrivialFLSolver(), GreedyPrefSolver()]
        results = [0, 0]
        for i in range(len(solvers)):
            for problem in get_all_problems():
                solution = solvers[i]._solve(problem)
                self.assertTrue(solution.is_feasible())
                results[i] += solution.get_value()

        self.assertLessEqual(results[1], results[0])
