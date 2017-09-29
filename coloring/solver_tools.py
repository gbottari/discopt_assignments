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
        # Check it two adjacent vertices share the same color
        for sorted_nodes in self.problem.sorted_edges:
            SN = len(sorted_nodes)
            for j in range(SN - 1):
                sn_j = sorted_nodes[j] - 1
                node_color = self.node_colors[sn_j]
                for k in range(j + 1, SN):
                    sn_k = sorted_nodes[k] - 1
                    if self.node_colors[sn_k] == node_color:
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
        return max(self.node_colors)

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

        sorted_edges = [[n] for n in range(len(nodes))]
        for n1, n2 in edges:
            sorted_edges[n1].append(n2)
            #sorted_edges[n2].append(n1)

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
