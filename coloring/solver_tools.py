import logging
from collections import namedtuple
from typing import List
import sys
sys.path.append('..')

from tools.solver_tools import Solution, Solver


GCProblem = namedtuple("GCProblem", ["input_edges", "sorted_edges", "nodes"])


class GCSolution(Solution):
    def __init__(self, problem: GCProblem):
        self.problem = problem
        self.optimal = False
        self.node_colors: List[int] = []

    def is_feasible(self):
        for n1, n2 in self.problem.input_edges:
            if self.node_colors[n1] == self.node_colors[n2]:
                return False
        return True

    def is_better(self, other: 'GCSolution'):
        return self.get_value() < other.get_value()

    def serialize(self):
        # Format:
        # value optimal
        # colors
        return '{} {}\n{}\n'.format(self.get_value(), int(self.optimal), ' '.join(str(c) for c in self.node_colors))

    def get_value(self):
        return max(self.node_colors) + 1

    def is_optimal(self):
        return self.optimal

    def copy(self):
        raise NotImplementedError()


class GCSolver(Solver):
    def _parse(self, raw_input_data: str) -> GCProblem:
        lines = raw_input_data.split('\n')

        first_line = lines[0].split()
        node_count = int(first_line[0])
        edge_count = int(first_line[1])

        edges = []
        nodes = set()
        for i in range(1, edge_count + 1):
            line = lines[i]
            parts = line.split()
            n1, n2 = int(parts[0]), int(parts[1])
            nodes.add(n1)
            nodes.add(n2)
            edges.append((min(n1, n2), max(n1, n2)))

        sorted_edges = [[] for _ in range(len(nodes))]
        for n1, n2 in edges:
            sorted_edges[n1].append(n2)
            sorted_edges[n2].append(n1)

        for i in range(len(sorted_edges)):
            sorted_edges[i].sort()

        return GCProblem(input_edges=edges, nodes=nodes, sorted_edges=sorted_edges)

    def _solve(self, problem: GCProblem):
        raise NotImplementedError()


class TrivialGCSolver(GCSolver):
    def _solve(self, problem: GCProblem):
        solution = GCSolution(problem)
        solution.node_colors = list(range(len(problem.nodes)))
        return solution


class GreedyChangeUntilSatisfy(GCSolver):
    """
    Assign different colors until the solution is feasible.
    """
    def _solve(self, problem: GCProblem):
        solution = GCSolution(problem)
        solution.node_colors = [0] * len(problem.nodes)

        last_color = 0
        stop = False

        while not stop:
            stop = True
            for n1, n2 in problem.input_edges:
                c1 = solution.node_colors[n1]
                c2 = solution.node_colors[n2]
                if c1 == c2:
                    stop = False
                    last_color += 1
                    solution.node_colors[n2] = last_color

        return solution


class GreedyBlackList(GCSolver):
    """
    Assigns the first color that it is available for a node, respecting its neighbors.
    """
    def _solve(self, problem: GCProblem):
        solution = GCSolution(problem)
        solution.node_colors = [None] * len(problem.nodes)

        black_lists = [set() for _ in range(len(problem.nodes))]

        for node, neighbors in enumerate(problem.sorted_edges):
            color = 0
            while color in black_lists[node]:
                color += 1
            solution.node_colors[node] = color
            for neighbor in neighbors:
                black_lists[neighbor].add(color)

        return solution
