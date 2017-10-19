import unittest
import os
import time
from tsp.solver_tools import *


def get_all_problem_filenames():
    return (f for f in os.listdir('data') if os.path.isfile(os.path.join('data', f)))


def get_problem_by_filename(filename) -> TSPProblem:
    with open(os.path.join('data', filename)) as f:
        return TSPSolver()._parse(f.read())


def get_all_problems():
    for fn in get_all_problem_filenames():
        yield get_problem_by_filename(fn)


def get_easy_problem() -> TSPProblem:
    return get_problem_by_filename('tsp_5_1')


class TestSolver(unittest.TestCase):
    def _check_solution(self, solution, optimal):
        self.assertIsInstance(solution, TSPSolution)
        self.assertTrue(solution.is_feasible())
        self.assertEqual(solution.is_optimal(), optimal)

    def test_input_order_solver(self):
        solver = InputOrderTSPSolver()
        solution = solver._solve(get_easy_problem())
        self._check_solution(solution, optimal=False)

    def test_new_idea(self):
        problem = get_problem_by_filename('tsp_51_1')
        solver = NewIdeaTSPSolver(100)
        solution = solver._solve(problem)
        self._check_solution(solution, optimal=False)

    def test_next_point(self):
        problem = get_easy_problem()
        solution: TSPSolution = TSPSolution(problem)
        solution.sequence = list(range(problem.n))
        random.shuffle(solution.sequence)
        self.assertEqual(solution.point(3), problem.points[solution.sequence[3]])
        self.assertEqual(solution.next_node(3), solution.sequence[3 + 1])
        self.assertEqual(solution.next_node(problem.n - 1), solution.sequence[0])
        self.assertEqual(solution.prev_node(3), solution.sequence[3 - 1])
        self.assertEqual(solution.prev_node(0), solution.sequence[-1])
        self.assertEqual(solution.next_point(3), problem.points[solution.sequence[3 + 1]])
        self.assertEqual(solution.next_point(problem.n - 1), problem.points[solution.sequence[0]])
        self.assertEqual(solution.prev_point(3), problem.points[solution.sequence[3 - 1]])
        self.assertEqual(solution.prev_point(0), problem.points[solution.sequence[-1]])

    def test_check_get_value_matches_expeced_value(self):
        problem = get_easy_problem()
        solution: TSPSolution = TSPSolution(problem)
        solution.sequence = [0, 4, 1, 3, 2]
        self.assertEqual(round(solution.get_value(), 1), 5.2)

    def test_dist_cache(self):
        problem = get_easy_problem()
        self.assertEqual(len(problem.dist_cache), 0)
        problem.dist(problem.points[0], problem.points[1])
        self.assertEqual(len(problem.dist_cache), 1)
        problem.dist(problem.points[1], problem.points[0])
        self.assertEqual(len(problem.dist_cache), 1)

    '''
    def test_dist_speed(self):
        times = []
        for func_str in ('dist', 'dist2'):
            problem = get_easy_problem()
            func = getattr(problem, func_str)
            problem.dist_cache = dict()
            for p in problem.points:
                func(p, p)
            t0_dist = time.time()
            for _ in range(10000):
                for p in problem.points:
                    func(p, p)
            t1_dist = time.time()
            times.append(t1_dist - t0_dist)
        print(times)
        self.assertGreaterEqual(times[0], times[1])
    '''