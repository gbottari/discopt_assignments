import unittest
from solver_tools import *
from tools.solver_tools import MultiSolver


class ExceptionSolver(Solver):
    def _solve(self, input_data):
        raise Exception()

problem_files = ['./data/ks_30_0']
lecture_file = './data/ks_lecture_dp_2'


# items capacity
# value weight
raw_simple_input_problem = """\
3 3
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
        self.assertGreater(solution.get_value(), 0)

    def test_solve_fifo(self):
        solver = FifoKSSolver()
        solution = solver.solve(raw_simple_input_problem)
        self._check_solution(solution)
        self.assertFalse(solution.is_optimal())
        serialized = solution.serialize()
        self.assertIsInstance(serialized, str)
        self._check_serialized_sol(serialized, solver._parse(raw_simple_input_problem))

    def test_solve_greedy_max_value(self):
        solver = GreedyMaxValueKSSolver()
        solution = solver.solve(raw_simple_input_problem)
        self._check_solution(solution)
        self.assertFalse(solution.is_optimal())

    def test_solve_greedy_min_weight(self):
        solver = GreedyMinWeightKSSolver()
        solution = solver.solve(raw_simple_input_problem)
        self._check_solution(solution)
        self.assertFalse(solution.is_optimal())

    def test_solve_greedy_max_density(self):
        solver = GreedyMaxDensityKSSolver()
        solution = solver.solve(raw_simple_input_problem)
        self._check_solution(solution)
        self.assertFalse(solution.is_optimal())

    def test_multi_solver(self):
        solver = MultiSolver([GreedyMaxValueKSSolver(), GreedyMinWeightKSSolver()])
        solution = solver.solve(raw_simple_input_problem)
        self._check_solution(solution)
        self.assertFalse(solution.is_optimal())

    def test_multi_solver_survives_exceptions(self):
        solver = MultiSolver([GreedyMinWeightKSSolver(), ExceptionSolver()])
        solution = solver.solve(raw_simple_input_problem)
        self._check_solution(solution)

    def test_multi_solver_gives_up_if_all_solvers_raises(self):
        solver = MultiSolver([ExceptionSolver(), ExceptionSolver()])
        self.assertRaises(Exception, solver.solve, raw_simple_input_problem)

    def test_pd_solver(self):
        solver = PDKSSolver()
        solution = solver.solve(raw_simple_input_problem)
        self._check_solution(solution)
        self.assertTrue(solution.is_optimal())
        self.assertEqual(solution.get_value(), 8)

    @unittest.skip('')
    def test_pd_solver_must_be_the_best(self):
        solver = PDKSSolver()
        with open(problem_files[0]) as f:
            raw_input = f.read()
        best_solution = solver.solve(raw_input)

        solver = MultiSolver([GreedyMaxDensityKSSolver(), GreedyMaxValueKSSolver(), GreedyMinWeightKSSolver()])
        solution = solver.solve(raw_input)

        self.assertLessEqual(solution.get_value(), best_solution.get_value())

    def test_pd_solver_must_find_optimal(self):
        solver = PDKSSolver()
        with open(lecture_file) as f:
            raw_input = f.read()
        best_solution = solver.solve(raw_input)
        self.assertEqual(best_solution.get_value(), 44)


if __name__ == '__main__':
    unittest.main()

