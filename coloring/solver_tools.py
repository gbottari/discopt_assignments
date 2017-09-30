import logging
import multiprocessing
import pymzn
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


class GreedyMostNeighbors(GCSolver):
    """
    Assigns the first color that it is available for a node respecting its neighbors. Select colors for the nodes with
    the most neighbors first.
    """
    def _solve(self, problem: GCProblem):
        solution = GCSolution(problem)
        solution.node_colors = [None] * len(problem.nodes)

        black_lists = [set() for _ in range(len(problem.nodes))]

        for node, neighbors in sorted(enumerate(problem.sorted_edges), key=lambda t: len(t[1]), reverse=True):
            color = 0
            while color in black_lists[node]:
                color += 1
            solution.node_colors[node] = color
            for neighbor in neighbors:
                black_lists[neighbor].add(color)

        return solution


class GreedyMostRestrictionsThenMostNeighbors(GreedyMostNeighbors):
    """
    Sort by restrictions and then neighbors.
    """
    def _solve(self, problem: GCProblem):
        solution = GCSolution(problem)
        solution.node_colors = [None] * len(problem.nodes)

        black_lists = [set() for _ in range(len(problem.nodes))]
        unassigned_nodes = set(range(len(problem.nodes)))

        while unassigned_nodes:
            # Find a node that:
            # 1. Has the most restrictions on the blacklist
            # 2. Has the most neighbors
            best_node = (None, 0, 0)
            for n in unassigned_nodes:
                restrictions = len(black_lists[n])
                neighbors = len(problem.sorted_edges[n])
                if restrictions > best_node[1] or (restrictions == best_node[1] and neighbors > best_node[2]):
                    best_node = (n, restrictions, neighbors)

            node = best_node[0]
            neighbors = problem.sorted_edges[node]
            unassigned_nodes.remove(node)

            color = 0
            while color in black_lists[node]:
                color += 1
            solution.node_colors[node] = color
            for neighbor in neighbors:
                black_lists[neighbor].add(color)

        return solution


class GreedyMostNeighborsThenMostRestrictions(GreedyMostNeighbors):
    """
    Sort by restrictions and then neighbors.
    """
    def _solve(self, problem: GCProblem):
        solution = GCSolution(problem)
        solution.node_colors = [None] * len(problem.nodes)

        black_lists = [set() for _ in range(len(problem.nodes))]
        unassigned_nodes = set(range(len(problem.nodes)))

        while unassigned_nodes:
            # Find a node that:
            # 1. Has the most restrictions on the blacklist
            # 2. Has the most neighbors
            best_node = (None, 0, 0)
            for n in unassigned_nodes:
                restrictions = len(black_lists[n])
                neighbors = len(problem.sorted_edges[n])
                if neighbors > best_node[2] or (neighbors == best_node[2] and restrictions > best_node[1]):
                    best_node = (n, restrictions, neighbors)

            node = best_node[0]
            neighbors = problem.sorted_edges[node]
            unassigned_nodes.remove(node)

            color = 0
            while color in black_lists[node]:
                color += 1
            solution.node_colors[node] = color
            for neighbor in neighbors:
                black_lists[neighbor].add(color)

        return solution


class CPGCSolver(GCSolver):
    def __init__(self):
        self.timeout = None

    def cleanup(self):
        import os
        os.system('taskkill /f /im mzn2fzn.exe')
        os.system('taskkill /f /im fzn-gecode.exe')

    def set_timeout(self, timeout: int):
        self.timeout = timeout // 2

    def _get_minizinc_params(self):
        physical_cores = multiprocessing.cpu_count() // 2
        return dict(parallel=physical_cores - 1, timeout=self.timeout)

    def _solve(self, problem: GCProblem):
        neighbors = [set([n + 1 for n in ns]) for ns in problem.sorted_edges]
        data = dict(n_nodes=len(problem.nodes), neighbors=neighbors)
        sol_stream: pymzn.SolnStream = pymzn.minizinc('coloring.mzn', data=data, **self._get_minizinc_params())

        solution = GCSolution(problem=problem)
        solution.node_colors = [c - 1 for c in sol_stream._solns[0]['colors']]
        solution.optimal = sol_stream.complete
        return solution
