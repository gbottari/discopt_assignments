import unittest
import os
from setcover.solver_tools import *
from tools.solver_tools import MultiSolver


def get_problem1():
    return SCProblem(sets=[SCSet(index=0, cost=1.0, items={0, 1}), SCSet(index=1, cost=1.0, items={2, 3}),
                           SCSet(index=2, cost=1.0, items={0, 2, 3})], items={0, 1, 2, 3})


def get_problem_byfilename(filename) -> SCProblem:
    with open(os.path.join('data', filename)) as f:
        return SCSolver()._parse(f.read())


class TestSolver(unittest.TestCase):
    def _check_solution(self, solution, optimal):
        self.assertIsInstance(solution, SCSolution)
        self.assertTrue(solution.is_feasible())
        self.assertEqual(solution.is_optimal(), optimal)

    def test_firstfit(self):
        solver = FirstFitSCSolver()
        solution = solver._solve(get_problem1())
        self._check_solution(solution, optimal=False)
        self.assertEqual(solution.get_value(), 2)
        self.assertEqual(solution.selections, [1, 1, 0])

    def test_greedy_max_cover(self):
        solver = GreedyMaxCoverSCSolver()
        problem = get_problem_byfilename('sc_6_1')
        solution = solver._solve(problem)
        self._check_solution(solution, optimal=False)

        # should be better than firstfit:
        self.assertLessEqual(solution.get_value(), FirstFitSCSolver()._solve(problem).get_value())

    def test_greedy_min_cover(self):
        solver = GreedyMinCostSCSolver()
        problem = SCProblem(sets=[SCSet(index=0, cost=3.0, items={0, 1}), SCSet(index=1, cost=2.0, items={1, 3}),
                                  SCSet(index=2, cost=1.0, items={0, 2, 3})], items={0, 1, 2, 3})
        solution = solver._solve(problem)
        self._check_solution(solution, optimal=False)
        self.assertEqual(solution.selections, [0, 1, 1])

    def test_cp(self):
        solver = CPSCSolver2()
        solution = solver._solve(get_problem1())
        self._check_solution(solution, optimal=True)

    def test_cp_better_than_heuristics(self):
        cp = CPSCSolver2()
        small_problem = get_problem_byfilename('sc_6_1')
        cp_solution = cp._solve(small_problem)
        hs = MultiSolver([FirstFitSCSolver(), GreedyMaxCoverSCSolver(), GreedyMaxCoverPerCostSCSolver()])
        hs_solution = hs._solve(small_problem)

        self._check_solution(cp_solution, optimal=True)
        self._check_solution(hs_solution, optimal=False)
        self.assertLessEqual(cp_solution.get_value(), hs_solution.get_value())

if __name__ == '__main__':
    unittest.main()
