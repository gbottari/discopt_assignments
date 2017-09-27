import sys
sys.path.append('..')

import logging
import pymzn
#from ortools.constraint_solver import pywrapcp  # todo: remove this
from collections import namedtuple
from tools.solver_tools import Solution, Solver
from typing import List

SCSet = namedtuple("Set", ["index", "cost", "items"])
SCProblem = namedtuple("Problem", ["sets", "items"])
Selections = List[int]


class SCSolution(Solution):
    def __init__(self, problem: SCProblem):
        self.problem: SCProblem = problem
        self.selections: Selections = None
        self.optimal = False

    def is_feasible(self) -> bool:
        # check that every item is covered
        solution_items = set()
        for i in range(len(self.selections)):
            if self.selections[i]:
                solution_items.update(self.problem.sets[i].items)
        return solution_items == self.problem.items

    def serialize(self):
        value = self.get_value()
        optimal = int(self.is_optimal())
        return "{} {}\n{}\n".format(value, optimal, " ".join(str(x) for x in self.selections))

    def get_value(self):
        return sum(self.selections[i] * self.problem.sets[i].cost for i in range(len(self.problem.sets)))

    def is_optimal(self):
        return self.optimal

    def copy(self):
        sol = SCSolution(self.problem)
        sol.selections = self.selections.copy()
        sol.optimal = self.optimal
        return sol


class SCSolver(Solver):
    def _parse(self, raw_input_data: str) -> SCProblem:
        lines = raw_input_data.split('\n')

        parts = lines[0].split()
        # item_count = int(parts[0])
        set_count = int(parts[1])

        problem = SCProblem(sets=[], items=set())
        for i in range(set_count):
            parts = lines[i + 1].split()
            # WARNING: we changed the cast here from FLOAT to INT
            items = [int(p) for p in parts[1:]]
            problem.sets.append(SCSet(i, int(parts[0]), items))
            problem.items.update(items)
        return problem

    def _solve(self, input_data):
        raise NotImplementedError()


# class OrToolsCPSCSolver(SCSolver):
#     def _solve(self, problem: SCProblem):
#         logger = logging.getLogger('solver')
#
#         solver: pywrapcp.Solver = pywrapcp.Solver("set_cover")
#         set_count = len(problem.sets)
#         item_count = len(problem.items)
#
#         # Decision vars for selecting the set i
#         x = [solver.IntVar(0, 1, "x_{:02d}".format(i)) for i in range(set_count)]
#         c = [solver.IntConst(problem.sets[i].cost, "c_{:02d}".format(i)) for i in range(set_count)]
#
#         # Minimizes the cost of the sum of the selected sets.
#         solver.Minimize(solver.Sum([x[i] * c[i] for i in range(set_count)]), step=1)
#
#         # Each item must be covered by at least one of the sets specified
#         covered_by = [[] for _ in range(item_count)]
#         for i in range(set_count):
#             for item in problem.sets[i].items:
#                 covered_by[item].append(x[i])
#
#         for i in range(item_count):
#             solver.Add(solver.Max(covered_by[i]) == 1)
#
#         # Dumb search:
#         db = solver.Phase(x, solver.CHOOSE_FIRST_UNBOUND, solver.ASSIGN_MIN_VALUE)
#
#         solution = SCSolution(problem=problem)
#         solution.optimal = False
#         # best_solution: SCSolution = None
#
#         # solver.NewSearch(db)
#         # while solver.NextSolution():
#         #     solution.selections = [x[i].Value() for i in range(set_count)]
#         #     if not best_solution or best_solution.get_value() > solution.get_value():
#         #         best_solution = solution.copy()
#         # solver.EndSearch()
#
#         #assignment = solver.Assignment()
#         collector: pywrapcp.SolutionCollector = solver.BestValueSolutionCollector(False)
#         solver.Solve(db, [collector])
#         print(collector.Solution(0))
#
#         return solution


class CPSCSolver(SCSolver):
    def _solve(self, problem: SCProblem):
        set_count = len(problem.sets)
        item_count = len(problem.items)

        # Each item must be covered by at least one of the sets specified
        covered_by = [set() for _ in range(item_count)]
        for i in range(set_count):
            for item in problem.sets[i].items:
                covered_by[item].add(i + 1)

        costs = [s.cost for s in problem.sets]

        data = dict(nsets=set_count, nitems=item_count, covered_by=covered_by, costs=costs)
        sol_stream: pymzn.SolnStream = pymzn.minizinc('setcover.mzn', data=data, parallel=8)
        solution = SCSolution(problem=problem)
        solution.selections = [int(x) for x in sol_stream._solns[0]['x']]
        solution.optimal = sol_stream.complete
        return solution
