import unittest
from setcover.solver_tools import *


class TestSolver(unittest.TestCase):
    def test_cpsc(self):
        solver = CPSCSolver()
        problem = SCProblem(sets=[SCSet(index=0, cost=1, items=[0, 1]), SCSet(index=1, cost=1, items=[2, 3]),
                                  SCSet(index=2, cost=1, items=[0, 2])], items=[0, 1, 2, 3])
        solution = solver._solve(problem)
        self.assertIsInstance(solution, SCSolution)
        self.assertEqual(solution.get_value(), 2)
        self.assertEqual(solution.selections, [1, 1, 0])

if __name__ == '__main__':
    unittest.main()
