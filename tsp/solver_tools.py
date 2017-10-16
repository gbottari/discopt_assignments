import math
import random
import logging
from collections import namedtuple
from typing import List, Any, Tuple

import sys
sys.path.append('..')

from tools.solver_tools import Solution, Solver


Point = namedtuple("Point", ['x', 'y'])
Sequence = List[int]


class Stats:
    def __init__(self):
        self.iterations = 0
        self.improvements: List[int] = []
        self.initial_value = 0.0
        self.final_value = 0.0

    def __repr__(self):
        s = ['=== Summary ===',
             '\titerations = {}, initial_value = {}'.format(self.iterations, self.initial_value),
             'Improvements:'] + ['{: 5d}: {: 3.2f} %'.format(i, p) for i, p in self._get_percent_improvements()]
        return '\n'.join(s)

    def _get_percent_improvements(self):
        last_imp = self.initial_value
        last_iter = 0
        for i, imp in self.improvements:
            yield i - last_iter, (1. - imp / last_imp) * 100.
            last_imp = imp
            last_iter = i


class TSPProblem:
    def __init__(self, points: List[Point]):
        self.points = points
        self.dist_cache = dict()
        self.n = len(self.points)

    def dist(self, p1: Point, p2: Point) -> float:
        # We avoid recalculating the distance for (p2, p1) by imposing an ordering:
        if p2.x < p1.x or p2.x == p1.x and p2.y < p1.y:
            p1, p2 = p2, p1

        key = p1, p2
        if key not in self.dist_cache:
            dist = math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
            self.dist_cache[key] = dist
        else:
            dist = self.dist_cache[key]
        return dist


class TSPSolution(Solution):
    def __init__(self, problem: TSPProblem):
        self.problem = problem
        self.sequence: Sequence = []
        self.optimal = False

    def is_feasible(self) -> bool:
        # check that the sequence has all the nodes in the problem and that nothing is repeated
        return frozenset(list(range(len(self.problem.points)))) == frozenset(self.sequence) and \
               len(self.problem.points) == len(self.sequence)

    def is_better(self, other: 'TSPSolution') -> bool:
        return self.is_optimal() or self.get_value() < other.get_value()

    def serialize(self):
        value = self.get_value()
        optimal = int(self.is_optimal())
        return "{} {}\n{}\n".format(value, optimal, " ".join(str(x) for x in self.sequence))

    def next_node(self, index: int) -> int:
        return self.sequence[(index + 1) % self.problem.n]

    def prev_node(self, index: int) -> int:
        return self.sequence[index - 1]

    def next_point(self, index: int) -> Point:
        return self.problem.points[self.next_node(index)]

    def prev_point(self, index: int) -> Point:
        return self.problem.points[self.prev_node(index)]

    def point(self, index: int) -> Point:
        return self.problem.points[self.sequence[index]]

    def get_value(self):
        total = 0.0
        for i in range(self.problem.n):
            total += self.problem.dist(self.point(i), self.next_point(i))
        return total

    def is_optimal(self):
        return self.optimal


class TSPSolver(Solver):
    def _parse(self, raw_input_data: str) -> TSPProblem:
        # parse the input
        lines = raw_input_data.split('\n')

        node_count = int(lines[0])
        points = []
        for i in range(1, node_count + 1):
            line = lines[i]
            parts = line.split()
            points.append(Point(float(parts[0]), float(parts[1])))
        return TSPProblem(points)

    def _solve(self, input_data: TSPProblem) -> TSPSolution:
        raise NotImplementedError()


class InputOrderTSPSolver(TSPSolver):
    def _solve(self, input_data: TSPProblem) -> TSPSolution:
        solution = TSPSolution(input_data)
        solution.sequence = list(range(len(input_data.points)))
        return solution


def filter_neighbors_in_range(center: Point, r_low: float, r_high: float, neighbors: List[Point]):
    for p in neighbors:
        if p.x - r_low <= center.x <= p.x + r_high and p.y - r_low <= center.y <= p.y + r_high:
            yield p


class MultiRandomInitializer(TSPSolver):
    def __init__(self, n):
        self.n = n

    def _solve(self, input_data: TSPProblem) -> TSPSolution:
        seq = list(range(len(input_data.points)))

        solution = InputOrderTSPSolver()._solve(input_data)
        best_value = solution.get_value()

        for _ in range(self.n):
            new_solution = TSPSolution(input_data)
            new_solution.sequence = seq.copy()
            random.shuffle(new_solution.sequence)
            new_value = new_solution.get_value()
            if new_value < best_value:
                solution = new_solution
                best_value = new_value

        return solution


class GreedyRandomSwapTSPSolver(TSPSolver):
    def __init__(self, max_swaps=100):
        self.max_swaps = max_swaps
        self.best_solution = None

    def stop(self) -> TSPSolution:
        return self.best_solution

    def _solve(self, input_data: TSPProblem):
        solution = MultiRandomInitializer(n=3)._solve(input_data)
        self.best_solution = solution
        best_value = solution.get_value()
        seq = list(range(len(solution.sequence)))

        # Try a number of random swaps:
        for _ in range(self.max_swaps):
            i = random.choice(seq[:-1])
            j = random.choice(seq[i + 1:])

            point_i = solution.point(i)
            point_j = solution.point(j)
            next_i = solution.next_point(i)
            prev_i = solution.prev_point(i)

            if next_i == point_j or prev_i == point_j:
                continue

            next_j = solution.next_point(j)
            prev_j = solution.prev_point(j)

            new_value = best_value
            new_value -= solution.problem.dist(point_i, next_i)
            new_value -= solution.problem.dist(prev_i, point_i)
            new_value -= solution.problem.dist(point_j, next_j)
            new_value -= solution.problem.dist(prev_j, point_j)

            new_value += solution.problem.dist(point_j, next_i)
            new_value += solution.problem.dist(prev_i, point_j)
            new_value += solution.problem.dist(point_i, next_j)
            new_value += solution.problem.dist(prev_j, point_i)

            if new_value < best_value:
                # Actually do the swap:
                solution.sequence[i], solution.sequence[j] = solution.sequence[j], solution.sequence[i]
                best_value = new_value
                self.best_solution = solution

        return solution


class Greedy2OptTSPSolver(TSPSolver):
    def __init__(self, max_swaps=100):
        self.max_swaps = max_swaps
        self.best_solution = None

    def stop(self) -> TSPSolution:
        return self.best_solution

    def _solve(self, input_data: TSPProblem):
        solution = MultiRandomInitializer(n=3)._solve(input_data)
        solution.stats = Stats()
        best_value = solution.get_value()
        self.best_solution = solution
        solution.stats.initial_value = best_value

        seq = list(range(len(solution.sequence)))

        for k in range(self.max_swaps):
            solution.stats.iterations += 1
            i = random.choice(seq[:-1])
            j = random.choice(seq[i + 1:])

            if i == 0 and j == len(seq) - 1:
                continue  # otherwise will discount twice and the solution is equivalent to the current one

            point_i = solution.point(i)
            point_j = solution.point(j)
            prev_i = solution.prev_point(i)
            next_j = solution.next_point(j)

            new_value = best_value
            new_value -= solution.problem.dist(prev_i, point_i)
            new_value -= solution.problem.dist(point_j, next_j)
            new_value += solution.problem.dist(prev_i, point_j)
            new_value += solution.problem.dist(point_i, next_j)

            if new_value < best_value:
                # Actually do the 2-OPT
                solution.stats.improvements.append((k, new_value))
                new_seq = solution.sequence[:i] + list(reversed(solution.sequence[i:j + 1])) + solution.sequence[j + 1:]
                solution.sequence = new_seq
                best_value = new_value
                solution.stats.final_value = best_value
                self.best_solution = solution

        return solution