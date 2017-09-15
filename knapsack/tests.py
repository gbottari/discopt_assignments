import unittest
from solver_tools import *


class TestSolver(unittest.TestCase):
    def setUp(self):
        self.solver = FifoKSSolver()

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


    def test_solve_simple_problem(self):
        raw_input_problem = """\
3 10
2 1
5 2
6 2"""

        solution = self.solver.solve(raw_input_problem)
        self.assertIsInstance(solution, Solution)
        self.assertTrue(solution.is_feasible())
        serialized = solution.serialize()
        self.assertIsInstance(serialized, str)
        self._check_serialized_sol(serialized, self.solver._parse(raw_input_problem))

if __name__ == '__main__':
    unittest.main()

