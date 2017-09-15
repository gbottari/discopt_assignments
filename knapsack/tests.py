import unittest
from solver_tools import *
from tools.solver_tools import MultiSolver

raw_simple_input_problem = """\
3 10
2 1
5 2
6 2"""


class TestSolver(unittest.TestCase):
    def _check_serialized_sol(self, serialized, problem):
        lines = serialized.splitlines()
        self.assertEqual(len(lines), 2)
        total_value, is_optimal = lines[0].split(' ')
        total_value, is_optimal = int(total_value), int(is_optimal)

        x = [int(i) for i in lines[1].split(' ')]
        self.assertTrue(all(x_i in (0, 1) for x_i in x))
        self.assertEqual(len(x), len(problem.items))

        self.assertTrue(is_optimal in (0, 1))
        self.assertEqual(total_value, sum(x_i * item.value for x_i, item in zip(x, problem.items)))

    def _check_solution(self, solution):
        self.assertIsInstance(solution, KSSolution)
        self.assertTrue(solution.is_feasible())

    def test_solve_fifo(self):
        solver = FifoKSSolver()
        solution = solver.solve(raw_simple_input_problem)
        self._check_solution(solution)
        self.assertFalse(solution.is_optimal)
        serialized = solution.serialize()
        self.assertIsInstance(serialized, str)
        self._check_serialized_sol(serialized, solver._parse(raw_simple_input_problem))

    def test_solve_greedy_max_value(self):
        solver = GreedyMaxValueKSSolver()
        solution = solver.solve(raw_simple_input_problem)
        self._check_solution(solution)
        self.assertFalse(solution.is_optimal)

    def test_solve_greedy_min_weight(self):
        solver = GreedyMinWeightKSSolver()
        solution = solver.solve(raw_simple_input_problem)
        self._check_solution(solution)
        self.assertFalse(solution.is_optimal)

    def test_solve_greedy_max_density(self):
        solver = GreedyMaxDensityKSSolver()
        solution = solver.solve(raw_simple_input_problem)
        self._check_solution(solution)
        self.assertFalse(solution.is_optimal)

    def test_multi_solver(self):
        solver = MultiSolver([GreedyMaxValueKSSolver(), GreedyMinWeightKSSolver()])
        solution = solver.solve(raw_simple_input_problem)
        self._check_solution(solution)
        self.assertFalse(solution.is_optimal)

if __name__ == '__main__':
    unittest.main()

