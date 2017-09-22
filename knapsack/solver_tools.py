import logging
import sys
sys.path.append('..')

from collections import namedtuple
from tools.solver_tools import Solution, Solver


KSItem = namedtuple("Item", ['index', 'value', 'weight'])
KSProblem = namedtuple("Problem", ['capacity', 'items'])


class KSSolution(Solution):
    def __init__(self, problem):
        self.problem = problem
        self.selected_items = set()
        self.optimal = False

    def get_value(self):
        return sum(item.value for item in self.selected_items)

    def add_item(self, item: KSItem):
        self.selected_items.add(item)

    def is_feasible(self):
        # Check that we have problem items
        return all(item in self.problem.items for item in self.selected_items)

    def is_optimal(self):
        return self.optimal

    def serialize(self):
        result = '{} {}'.format(self.get_value(), int(self.is_optimal()))
        x = [0] * len(self.problem.items)
        for item in self.selected_items:
            x[item.index] = 1
        result += '\n' + (' '.join(str(x_i) for x_i in x))
        return result


class KSSolver(Solver):
    def _parse(self, raw_input_data: str) -> KSProblem:
        lines = raw_input_data.split('\n')
        first_line = lines[0].split()
        item_count = int(first_line[0])
        capacity = int(first_line[1])

        items = []
        for i in range(1, item_count + 1):
            line = lines[i]
            parts = line.split()
            items.append(KSItem(index=i - 1, value=int(parts[0]), weight=int(parts[1])))

        return KSProblem(capacity=capacity, items=items)

    def _solve(self, input_data: KSProblem) -> KSSolution:
        raise NotImplementedError()


class FifoKSSolver(KSSolver):
    def _solve(self, input_data):
        capacity = 0
        solution = KSSolution(input_data)
        for item in input_data.items:
            if capacity + item.weight <= input_data.capacity:
                capacity += item.weight
                solution.add_item(item)
                if capacity == input_data.capacity:
                    break

        return solution


class GreedyMaxValueKSSolver(KSSolver):
    def _solve(self, input_data):
        capacity = 0
        solution = KSSolution(input_data)
        for item in sorted(input_data.items, key=lambda item: item.value, reverse=True):
            if capacity + item.weight <= input_data.capacity:
                capacity += item.weight
                solution.add_item(item)
                if capacity == input_data.capacity:
                    break

        return solution


class GreedyMinWeightKSSolver(KSSolver):
    def _solve(self, input_data):
        capacity = 0
        solution = KSSolution(input_data)
        for item in sorted(input_data.items, key=lambda item: item.weight):
            if capacity + item.weight <= input_data.capacity:
                capacity += item.weight
                solution.add_item(item)
                if capacity == input_data.capacity:
                    break

        return solution


class GreedyMaxDensityKSSolver(KSSolver):
    def _solve(self, input_data):
        capacity = 0
        solution = KSSolution(input_data)
        for item in sorted(input_data.items, key=lambda item: item.value / item.weight, reverse=True):
            if capacity + item.weight <= input_data.capacity:
                capacity += item.weight
                solution.add_item(item)
                if capacity == input_data.capacity:
                    break

        return solution


class PDKSSolver(KSSolver):
    MAX_SPACE_GB = 2

    def _solve(self, input_data: KSProblem):
        solution = KSSolution(input_data)
        solution.optimal = True

        table_size_in_gb = (input_data.capacity + 1) * (len(input_data.items) + 1) * 4 / (2 ** 30)
        if table_size_in_gb > self.MAX_SPACE_GB:
            raise Exception('This solution would require {:.1f} GB of space when the maximum is {} GB.'.format(
                table_size_in_gb, self.MAX_SPACE_GB))

        logging.getLogger('solver').debug('Table size: {:.1f} GB.'.format(table_size_in_gb))
        table = [[0] * (input_data.capacity + 1) for _ in range(len(input_data.items) + 1)]

        # Fill the table
        for i in range(1, len(input_data.items) + 1):
            item = input_data.items[i - 1]
            for w in range(1, input_data.capacity + 1):
                table[i][w] = max(table[i - 1][w], (item.value + table[i - 1][w - item.weight]) if item.weight <= w else 0)

        # Find the solution
        w = input_data.capacity
        for i in range(len(input_data.items), 0, -1):
            if table[i][w] > table[i - 1][w]:
                item = input_data.items[i - 1]
                solution.add_item(item)
                w -= item.weight

        return solution

