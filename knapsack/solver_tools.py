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
        self.is_optimal = False

    def _get_total_value(self):
        return sum(item.value for item in self.selected_items)

    def add_item(self, item: KSItem):
        self.selected_items.add(item)

    def is_feasible(self):
        # Check that we have problem items
        return all(item in self.problem.items for item in self.selected_items)

    def serialize(self):
        result = '{} {}'.format(self._get_total_value(), int(self.is_optimal))
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