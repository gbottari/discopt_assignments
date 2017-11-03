import math
import random
from tools.solver_tools import Solution, Solver
from collections import namedtuple
from typing import Optional

Point = namedtuple("Point", ['x', 'y'])
Facility = namedtuple("Facility", ['index', 'setup_cost', 'capacity', 'location'])
Customer = namedtuple("Customer", ['index', 'demand', 'location', 'prefs', 'dists'])


class FLProblem:
    def __init__(self, facilities, customers):
        self.facilities = facilities
        self.customers = customers
        self.dist_cache = dict()

    def dist(self, p1: Point, p2: Point) -> float:
        key = frozenset((p1, p2))
        if key not in self.dist_cache:
            dist = math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)
            self.dist_cache[key] = dist
        else:
            dist = self.dist_cache[key]
        return dist


class FLSolution(Solution):
    def __init__(self, problem):
        super().__init__()
        self.problem = problem
        self.selections = []
        self.optimal = False

    def get_value(self):
        facility_indexes = frozenset(self.selections)
        total_setup_cost = sum(self.problem.facilities[f_i].setup_cost for f_i in facility_indexes)
        total_dist = sum(self.problem.customers[c_i].dists[f_i] for c_i, f_i in enumerate(self.selections))
        return total_setup_cost + total_dist

    def is_feasible(self):
        # check capacity
        capacities = [self.problem.facilities[f_i].capacity for f_i in range(len(self.problem.facilities))]
        for c_i, f_i in enumerate(self.selections):
            demand = self.problem.customers[c_i].demand
            capacities[f_i] -= demand
            if capacities[f_i] < 0:
                return False
        return True

    def is_optimal(self):
        return self.optimal

    def serialize(self) -> str:
        # note: truncates the value
        return "{:.3f} {}\n{}".format(int(self.get_value() * 1000) / 1000, int(self.is_optimal()),
                                      " ".join(str(f_i) for f_i in self.selections))

    def is_better(self, other: 'Solution') -> bool:
        if self.optimal:
            return True
        if other.is_optimal():
            return False
        return self.get_value() < other.get_value()


class FLSolution2(FLSolution):
    def __init__(self, problem):
        super().__init__(problem)
        self.setup_cost = 0.0
        self.dist_cost = 0.0
        self.selections = [None for _ in range(len(problem.customers))]
        self.open_fs = set()  # do we need this?
        self.customer_count = [0 for _ in range(len(problem.facilities))]
        self.capacities = [f.capacity for f in problem.facilities]

    def copy(self):
        inst = FLSolution2(self.problem)
        inst.setup_cost = self.setup_cost
        inst.dist_cost = self.dist_cost
        inst.selections = self.selections.copy()
        inst.open_fs = self.open_fs.copy()
        inst.customer_count = self.customer_count.copy()
        inst.capacities = self.capacities.copy()
        return inst

    def open_facility(self, f_i):
        self.setup_cost += self.problem.facilities[f_i].setup_cost
        self.open_fs.add(f_i)
        self.capacities[f_i] = self.problem.facilities[f_i].capacity

    def close_facility(self, f_i):
        self.setup_cost -= self.problem.facilities[f_i].setup_cost
        self.open_fs.remove(f_i)
        self.capacities[f_i] = self.problem.facilities[f_i].capacity

    def bind_customer(self, c_i, f_i):
        old_f_i: Optional[int] = self.selections[c_i]
        customer = self.problem.customers[c_i]

        if old_f_i is not None:  # we need to reassign the customer
            self.dist_cost -= self.problem.customers[c_i].dists[old_f_i]
            self.customer_count[old_f_i] -= 1
            self.capacities[old_f_i] += customer.demand
            if self.customer_count[old_f_i] == 0:
                self.close_facility(old_f_i)

        if f_i is not None:
            if self.customer_count[f_i] == 0:
                self.open_facility(f_i)
            self.customer_count[f_i] += 1
            self.capacities[f_i] -= customer.demand
            self.dist_cost += self.problem.customers[c_i].dists[f_i]

        self.selections[c_i] = f_i


class FLSolver(Solver):
    def _solve(self, input_data):
        raise NotImplementedError()

    def _parse(self, raw_input_data: str):
        # parse the input
        lines = raw_input_data.split('\n')

        parts = lines[0].split()
        facility_count = int(parts[0])
        customer_count = int(parts[1])

        facilities = []
        customers = []
        problem = FLProblem(facilities, customers)

        for i in range(1, facility_count + 1):
            parts = lines[i].split()
            facilities.append(Facility(i - 1, float(parts[0]), int(parts[1]), Point(float(parts[2]), float(parts[3]))))

        for i in range(facility_count + 1, facility_count + 1 + customer_count):
            parts = lines[i].split()
            location = Point(float(parts[1]), float(parts[2]))
            dists = [problem.dist(location, fl.location) for fl in facilities]
            prefs = [index for index, dist in sorted(enumerate(dists), key=lambda t: t[1])]
            customers.append(Customer(i - 1 - facility_count, int(parts[0]), location, prefs, dists))

        return problem


class TrivialFLSolver(FLSolver):
    def _solve(self, problem: FLProblem) -> FLSolution2:
        # pack the facilities one by one until all the customers are served
        solution = FLSolution2(problem)

        facility_index = 0
        for customer in problem.customers:
            if solution.capacities[facility_index] < customer.demand:
                facility_index += 1
            solution.bind_customer(customer.index, facility_index)

        return solution


class GreedyPrefSolver(FLSolver):
    def _solve(self, problem: FLProblem) -> FLSolution2:
        solution = FLSolution2(problem)

        for customer in problem.customers:
            for f_i in customer.prefs:
                if solution.capacities[f_i] > customer.demand:
                    solution.bind_customer(customer.index, f_i)
                    break
            if solution.selections[customer.index] is None:
                raise Exception()

        return solution


def get_f_capacity_relaxation(partial_solution: FLSolution):
    dist_cost = 0
    problem = partial_solution.problem
    facilities = set()
    for c_i, f_i in enumerate(partial_solution.selections):
        if f_i is None:
            f_i = problem.customers[c_i].prefs[0]
        facilities.add(f_i)
        dist_cost += problem.customers[c_i].dists[f_i]
    setup_cost = sum(problem.facilities[f_i].setup_cost for f_i in facilities)
    return setup_cost + dist_cost


def get_f_capacity_relaxation2(solution: FLSolution2):
    setup_cost = solution.setup_cost
    dist_cost = solution.dist_cost

    problem = solution.problem
    open_fs = solution.open_fs.copy()
    for c_i in range(len(solution.selections)):
        if solution.selections[c_i] is not None:
            continue
        customer = problem.customers[c_i]
        for f_i in customer.prefs:
            if solution.capacities[f_i] < customer.demand:
                continue
            dist_cost += problem.customers[c_i].dists[f_i]
            if f_i not in open_fs:
                open_fs.add(f_i)
                setup_cost += problem.facilities[f_i].setup_cost
            break

    return setup_cost + dist_cost


class DFBnBSolver(FLSolver):
    def __init__(self):
        super().__init__()
        self.best_solution = None
        self.best_value = None
        self._stop = False

    def stop(self):
        self._stop = True

    def _bnb(self, solution, c_i) -> None:
        if self._stop:
            return

        if c_i == len(solution.problem.customers):
            # check if the new solution is better:
            if solution.is_better(self.best_solution):
                self.best_solution = solution.copy()
                self.best_value = self.best_solution.get_value()
            return

        c = solution.problem.customers[c_i]
        best_value = self.best_value
        for f in solution.problem.facilities:
            if best_value > self.best_value:
                # this means that a new solution was found and maybe we can discard this branch now
                lb = get_f_capacity_relaxation2(solution)
                if lb > self.best_value:
                    return
                best_value = self.best_value

            if solution.capacities[f.index] < c.demand:  # can't use this factory
                continue

            solution.bind_customer(c_i, f.index)

            lb = get_f_capacity_relaxation2(solution)
            if lb > self.best_value:  # this binding is not optimal
                solution.bind_customer(c_i, None)
                continue

            # go to the next customer
            self._bnb(solution, c_i + 1)
            solution.bind_customer(c_i, None)

    def _solve(self, problem: FLProblem) -> FLSolution2:
        self.best_solution = GreedyPrefSolver()._solve(problem)
        self.best_value = self.best_solution.get_value()
        solution = FLSolution2(problem)
        self._bnb(solution, c_i=0)
        return self.best_solution

