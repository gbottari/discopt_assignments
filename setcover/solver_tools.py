import multiprocessing
import random
import logging
import copy
import pymzn
from collections import namedtuple
from typing import List

import sys
sys.path.append('..')

from tools.solver_tools import Solution, Solver


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

    def is_better(self, other: 'SCSolution'):
        return self.get_value() < other.get_value()

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
            items = [int(p) for p in parts[1:]]
            problem.sets.append(SCSet(i, float(parts[0]), items))
            problem.items.update(items)
        return problem

    def _solve(self, input_data):
        raise NotImplementedError()


class FirstFitSCSolver(SCSolver):
    """
    Adds sets until the problem is satisfied sequentially.
    """
    def _solve(self, problem: SCProblem):
        set_count = len(problem.sets)
        item_count = len(problem.items)

        solution = SCSolution(problem)
        solution.selections = [0] * set_count
        covered = set()

        for s in problem.sets:
            solution.selections[s.index] = 1
            covered |= set(s.items)
            if len(covered) >= item_count:
                break

        return solution


class RandomFitSCSolver(SCSolver):
    """
    Adds sets until the problem is satisfied randomly.
    """
    def _solve(self, problem: SCProblem):
        set_count = len(problem.sets)
        item_count = len(problem.items)

        solution = SCSolution(problem)
        solution.selections = [0] * set_count
        covered = set()
        sets = problem.sets.copy()
        random.shuffle(sets)

        for s in sets:
            solution.selections[s.index] = 1
            covered |= set(s.items)
            if len(covered) >= item_count:
                break

        return solution


class GreedyMaxCoverSCSolver(SCSolver):
    """
    Order each set by the number of items that are not yet covered, and chooses that set.
    """
    def _solve(self, problem: SCProblem):
        set_count = len(problem.sets)
        item_count = len(problem.items)

        sets = copy.deepcopy(problem.sets)
        solution = SCSolution(problem)
        solution.selections = [0] * set_count
        covered = set()

        while len(covered) < item_count:
            # finds the sets with the largest covers:
            max_cover = max(len(s.items) for s in sets)
            best_sets = [s for s in sets if len(s.items) == max_cover]
            lowest_cost = min(s.cost for s in best_sets)
            choice: SCSet = next(s for s in best_sets if s.cost == lowest_cost)
            solution.selections[choice.index] = 1

            # remove that set:
            sets.remove(choice)

            # update the items in the set given the choice
            for s in sets:
                if choice in s.items:
                    s.items.remove(choice)

            covered.update(choice.items)

            # removes empty sets
            sets = [s for s in sets if s.items]

        return solution


class GreedyMaxCoverPerCostSCSolver(SCSolver):
    """
    Order each set by the number of items that are not yet covered, and chooses that set.
    """
    def _solve(self, problem: SCProblem):
        set_count = len(problem.sets)
        item_count = len(problem.items)

        sets = copy.deepcopy(problem.sets)
        solution = SCSolution(problem)
        solution.selections = [0] * set_count
        covered = set()

        while len(covered) < item_count:
            # finds the sets with the largest covers:
            best_sets = sorted(sets, key=lambda s_: len(s_.items) / s_.cost, reverse=True)
            choice: SCSet = best_sets[0]
            solution.selections[choice.index] = 1

            # remove that set:
            sets.remove(choice)

            # update the items in the set given the choice
            for s in sets:
                if choice in s.items:
                    s.items.remove(choice)

            covered.update(choice.items)

            # removes empty sets
            sets = [s for s in sets if s.items]

        return solution


class GreedyMinCostSCSolver(SCSolver):
    """
    Selects sets with minimum cost until everything is covered.
    """
    def _solve(self, problem: SCProblem):
        set_count = len(problem.sets)
        item_count = len(problem.items)

        solution = SCSolution(problem)
        solution.selections = [0] * set_count
        covered = set()

        for s in sorted(problem.sets, key=lambda s_: s_.cost):
            solution.selections[s.index] = 1
            covered |= set(s.items)
            if len(covered) >= item_count:
                break

        return solution


class CPSCSolver(SCSolver):
    def __init__(self):
        self.timeout = None

    def cleanup(self):
        import os
        os.system('taskkill /f /im mzn2fzn.exe')
        os.system('taskkill /f /im fzn-gecode.exe')

    def set_timeout(self, timeout: int):
        self.timeout = timeout // 5

    def _get_minizinc_params(self):
        physical_cores = multiprocessing.cpu_count() // 2
        return dict(parallel=physical_cores - 1, timeout=self.timeout)

    def _solve(self, problem: SCProblem):
        set_count = len(problem.sets)
        item_count = len(problem.items)

        # Each item must be covered by at least one of the sets specified
        covered_by = [set() for _ in range(item_count)]
        for i in range(set_count):
            for item in problem.sets[i].items:
                covered_by[item].add(i + 1)

        costs = [s.cost for s in problem.sets]
        cost_values = set(costs)

        data = dict(nsets=set_count, nitems=item_count, covered_by=covered_by, costs=costs, COSTS=cost_values)
        sol_stream: pymzn.SolnStream = pymzn.minizinc('setcover.mzn', data=data, **self._get_minizinc_params())

        solution = SCSolution(problem=problem)
        solution.selections = [int(x) for x in sol_stream._solns[0]['x']]
        solution.optimal = sol_stream.complete
        return solution


class CPSCSolver2(CPSCSolver):
    def set_timeout(self, timeout: int):
        self.timeout = timeout // 2

    def _get_minizinc_params(self):
        cores = multiprocessing.cpu_count()
        return dict(parallel=cores - 1, timeout=self.timeout)

    def _solve(self, problem: SCProblem):
        set_count = len(problem.sets)
        item_count = len(problem.items)

        costs = [s.cost for s in problem.sets]
        covers = [set(s.items) for s in problem.sets]

        data = dict(nsets=set_count, nitems=item_count, covers=covers, costs=costs)
        sol_stream: pymzn.SolnStream = pymzn.minizinc('setcover2.mzn', data=data, **self._get_minizinc_params())

        solution = SCSolution(problem=problem)
        solution.selections = [int(x) for x in sol_stream._solns[0]['x']]
        solution.optimal = sol_stream.complete
        return solution
