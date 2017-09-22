import sys
sys.path.append('..')

import logging
from ortools.constraint_solver import pywrapcp
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


class CPSCSolver(SCSolver):
    def _solve(self, problem: SCProblem):
        logger = logging.getLogger('solver')

        solver = pywrapcp.Solver("set_cover")
        set_count = len(problem.sets)
        item_count = len(problem.items)

        # Decision vars for selecting the set i
        x = [solver.IntVar(0, 1, "x_{:02d}".format(i)) for i in range(set_count)]
        c = [solver.IntConst(problem.sets[i].cost, "c_{:02d}".format(i)) for i in range(set_count)]

        # Minimizes the cost of the sum of the selected sets.
        solver.Minimize(solver.Sum([x[i] * c[i] for i in range(set_count)]), 1)

        # Each item must be covered by at least one of the sets specified
        covered_by = [[] for _ in range(item_count)]
        for i in range(set_count):
            for item in problem.sets[i].items:
                covered_by[item].append(x[i])

        for i in range(item_count):
            solver.Add(solver.Max(covered_by[i]) == 1)

        # Dumb search:
        db = solver.Phase(x, solver.CHOOSE_FIRST_UNBOUND, solver.ASSIGN_MIN_VALUE)

        solution = SCSolution(problem=problem)
        solution.optimal = True

        solver.NewSearch(db)
        if not solver.NextSolution():
            raise Exception("No solution found!")
        solution.selections = [x[i].Value() for i in range(set_count)]
        solver.EndSearch()
        return solution
