import math
import time
import random
import psutil
import logging
from collections import namedtuple
from itertools import chain
from typing import Optional, List, Sequence, Iterable
from pyscipopt import Model, quicksum, SCIP_PARAMEMPHASIS

import sys
sys.path.append('..')

from tools.solver_tools import Solution, Solver, MultiSolver

sys.setrecursionlimit(2000)

Point = namedtuple("Point", ['x', 'y'])
Facility = namedtuple("Facility", ['index', 'setup_cost', 'capacity', 'location'])
Customer = namedtuple("Customer", ['index', 'demand', 'location', 'prefs', 'dists'])


def split_problem(problem: 'FLProblem'):
    top_left = next(iter(problem.customers.values())).location
    bottom_right = top_left

    # find the rect coordinates
    for point in (obj.location for obj in chain(problem.customers.values(), problem.facilities.values())):
        if top_left.x > point.x:
            top_left = Point(x=point.x, y=top_left.y)
        if top_left.y < point.y:
            top_left = Point(x=top_left.x, y=point.y)
        if bottom_right.x < point.x:
            bottom_right = Point(x=point.x, y=bottom_right.y)
        if bottom_right.y > point.y:
            bottom_right = Point(x=bottom_right.x, y=point.y)

    height = top_left.y - bottom_right.y
    width = bottom_right.x - top_left.x

    # divide along the biggest side:
    rects: List[Point] = [None, None]
    if height > width:
        mid_y = (top_left.y + bottom_right.y) / 2
        rects[0] = (top_left, Point(x=bottom_right.x, y=mid_y))
        rects[1] = (Point(x=top_left.x, y=mid_y), bottom_right)
    else:
        mid_x = (top_left.x + bottom_right.x) / 2
        rects[0] = (top_left, Point(x=mid_x, y=bottom_right.y))
        rects[1] = (Point(x=top_left.x, y=mid_x), bottom_right)

    facilities = [{}, {}]
    customers = [{}, {}]

    top_left, bottom_right = rects[0]
    for customer in problem.customers.values():
        if top_left.x <= customer.location.x <= bottom_right.x and bottom_right.y <= customer.location.y <= top_left.y:
            customers[0][customer.index] = customer
        else:
            customers[1][customer.index] = customer
    for facility in problem.facilities.values():
        if top_left.x <= facility.location.x <= bottom_right.x and bottom_right.y <= facility.location.y <= top_left.y:
            facilities[0][facility.index] = facility
        else:
            facilities[1][facility.index] = facility

    return FLProblem(facilities=facilities[0], customers=customers[0]), \
           FLProblem(facilities=facilities[1], customers=customers[1])


def merge_problems(problems: Iterable['FLProblem']):
    facilities = {}
    customers = {}

    for p in problems:
        for facility in p.facilities.values():
            facilities[facility.index] = facility
        for customer in p.customers.values():
            customers[customer.index] = customer

    return FLProblem(facilities=facilities, customers=customers)


def merge_solutions(solutions: Sequence['FLSolution']):
    merged_problem = merge_problems(s.problem for s in solutions)
    merged_solution = FLSolution(merged_problem)
    merged_solution.selections = {}
    for solution in solutions:
        merged_solution.selections.update(solution.selections)
    return merged_solution


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


class Stats:
    def __init__(self):
        self.iterations = 0
        self.improvements_x = []
        self.improvements_y = []
        self.temperature = []
        self.probs = []
        self.initial_value = 0.0
        self.final_value = 0.0


class FLSolution(Solution):
    def __init__(self, problem):
        super().__init__()
        self.problem = problem
        self.selections = {}  # customer: facility index the facility index for each customer
        self.optimal = False

    def get_value(self):
        facility_indexes = frozenset(self.selections.values())
        total_setup_cost = sum(self.problem.facilities[f_i].setup_cost for f_i in facility_indexes)
        total_dist = sum(self.problem.customers[c_i].dists[f_i] for c_i, f_i in self.selections.items())
        return total_setup_cost + total_dist

    def is_feasible(self):
        # check capacity
        capacities = {f.index: f.capacity for f in self.problem.facilities.values()}
        for c_i, f_i in self.selections.items():
            capacities[f_i] -= self.problem.customers[c_i].demand
            if capacities[f_i] < 0:
                return False
        return True

    def is_optimal(self):
        return self.optimal

    def serialize(self) -> str:
        # note: truncates the value
        serialized_selections = [f for _, f in sorted(self.selections.items(), key=lambda item: item[0])]
        return "{:.3f} {}\n{}".format(int(self.get_value() * 1000) / 1000, int(self.is_optimal()),
                                      " ".join(str(f_i) for f_i in serialized_selections))

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
        self.selections = {c_i: None for c_i in problem.customers}
        self.open_fs = set()  # do we need this?
        self.customer_count = {f_i: 0 for f_i in problem.facilities}
        self.capacities = {f.index: f.capacity for f in problem.facilities.values()}
        self.total_demand = sum(c.demand for c in problem.customers.values())
        self.total_capacity = sum(self.capacities)
        self.demand_covered = 0

    def get_value(self):
        return self.setup_cost + self.dist_cost

    def copy(self):
        inst = FLSolution2(self.problem)
        inst.setup_cost = self.setup_cost
        inst.dist_cost = self.dist_cost
        inst.selections = self.selections.copy()
        inst.open_fs = self.open_fs.copy()
        inst.customer_count = self.customer_count.copy()
        inst.capacities = self.capacities.copy()
        inst.total_demand = self.total_demand
        inst.total_capacity = self.total_capacity
        inst.demand_covered = self.demand_covered
        inst.stats = self.stats
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
            old_dist = self.problem.customers[c_i].dists[old_f_i]
            self.dist_cost -= old_dist
            self.customer_count[old_f_i] -= 1
            self.capacities[old_f_i] += customer.demand
            if self.customer_count[old_f_i] == 0:
                self.close_facility(old_f_i)
        else:
            self.demand_covered += customer.demand

        if f_i is not None:
            if self.customer_count[f_i] == 0:
                self.open_facility(f_i)
            self.customer_count[f_i] += 1
            self.capacities[f_i] -= customer.demand
            self.dist_cost += self.problem.customers[c_i].dists[f_i]
        else:
            self.demand_covered -= customer.demand

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

        facilities = {}
        customers = {}
        problem = FLProblem(facilities, customers)

        for i in range(1, facility_count + 1):
            parts = lines[i].split()
            index = i - 1
            facilities[index] = Facility(index, float(parts[0]), int(parts[1]), Point(float(parts[2]), float(parts[3])))

        for i in range(facility_count + 1, facility_count + 1 + customer_count):
            parts = lines[i].split()
            location = Point(float(parts[1]), float(parts[2]))
            dists = [problem.dist(location, fl.location) for fl in facilities.values()]
            prefs = [index for index, dist in sorted(enumerate(dists), key=lambda t: t[1])]
            index = i - 1 - facility_count
            customers[index] = Customer(index, int(parts[0]), location, prefs, dists)

        return problem


class TrivialFLSolver(FLSolver):
    def _solve(self, problem: FLProblem) -> FLSolution2:
        # pack the facilities one by one until all the customers are served
        solution = FLSolution2(problem)

        facility_index_iter = iter(problem.facilities.keys())
        facility_index = next(facility_index_iter)
        for customer in problem.customers.values():
            while True:
                if solution.capacities[facility_index] < customer.demand:
                    facility_index = next(facility_index_iter)
                else:
                    solution.bind_customer(customer.index, facility_index)
                    break

        return solution


class GreedyPrefSolver(FLSolver):
    def _solve(self, problem: FLProblem) -> FLSolution2:
        solution = FLSolution2(problem)

        for customer in problem.customers.values():
            for f_i in customer.prefs:
                if solution.capacities[f_i] > customer.demand:
                    solution.bind_customer(customer.index, f_i)
                    break
            if solution.selections[customer.index] is None:
                raise Exception()

        return solution


class GreedyDistSolver(FLSolver):
    def _solve(self, problem: FLProblem) -> FLSolution2:
        solution = FLSolution2(problem)

        for customer in problem.customers.values():
            prefs = sorted(solution.problem.facilities.values(), reverse=False,
                           key=lambda f: (f.setup_cost if f not in solution.open_fs else 0) + customer.dists[f.index])
            for f in prefs:
                f_i = f.index
                if solution.capacities[f_i] >= customer.demand:
                    solution.bind_customer(customer.index, f_i)
                    break
            if solution.selections[customer.index] is None:
                raise Exception()

        return solution


class RandSolver(FLSolver):
    def _solve(self, problem: FLProblem) -> FLSolution2:
        solution = FLSolution2(problem)

        while solution.demand_covered < solution.total_demand:
            customer = random.choice(problem.customers)
            facility = random.choice(problem.facilities)
            if solution.capacities[facility.index] >= customer.demand:
                solution.bind_customer(customer.index, facility.index)

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
    dist_cost = solution.dist_cost
    problem = solution.problem

    for c_i in range(len(solution.selections)):
        if solution.selections[c_i] is not None:
            continue
        customer = problem.customers[c_i]
        for f_i in customer.prefs:
            if solution.capacities[f_i] >= customer.demand:
                dist_cost += problem.customers[c_i].dists[f_i]
                break

    return dist_cost


def get_remaining_setup_cost_estimation(solution: FLSolution2):
    setup_cost = solution.setup_cost
    problem = solution.problem
    remaining_demand = solution.total_demand - solution.demand_covered
    current_capacity = sum(solution.capacities[f_i] for f_i in range(len(problem.facilities)) if f_i in solution.open_fs)

    if current_capacity < remaining_demand:
        remaining_capacity = remaining_demand
        # order the facilities by capacity / setup_cost and estimates based on that
        for f in sorted((f for f in problem.facilities.values() if f not in solution.open_fs), reverse=True,
                        key=lambda f: f.capacity / max(f.setup_cost, 0.001)):
            if f.capacity > remaining_capacity:
                setup_cost += f.setup_cost * remaining_capacity / f.capacity
                break
            else:
                setup_cost += f.capacity
                remaining_capacity -= f.capacity

    return setup_cost


def get_relaxation2(solution: FLSolution2):
    return get_remaining_setup_cost_estimation(solution) + get_f_capacity_relaxation2(solution)


class DFBnBSolver(FLSolver):
    def __init__(self):
        super().__init__()
        self.best_solution = None
        self.best_value = None
        self._stop = False
        self.customers = None

    def stop(self):
        self._stop = True
        return self.best_solution

    def _bnb(self, solution, index) -> None:
        if self._stop:
            return

        if index == len(solution.problem.customers):
            # check if the new solution is better:
            if solution.is_better(self.best_solution):
                self.best_solution = solution.copy()
                self.best_value = self.best_solution.get_value()
            return

        c_i = self.customers[index].index
        c = solution.problem.customers[c_i]
        best_value = self.best_value

        # Try the best combination of distance and setup_cost
        prefs = sorted(solution.problem.facilities.values(), reverse=False,
                       key=lambda f: (f.setup_cost if f not in solution.open_fs else 0) + c.dists[f.index])

        for f in prefs:
            f_i = f.index
            if best_value > self.best_value:
                # this means that a new solution was found and maybe we can discard this branch now
                lb = get_relaxation2(solution)
                if lb > self.best_value:
                    return
                best_value = self.best_value

            if solution.capacities[f_i] < c.demand:  # can't use this factory
                continue

            solution.bind_customer(c_i, f_i)

            lb = get_relaxation2(solution)
            if lb > self.best_value:  # this binding is not optimal
                solution.bind_customer(c_i, None)
                continue

            # go to the next customer
            self._bnb(solution, index + 1)
            solution.bind_customer(c_i, None)

    def _solve(self, problem: FLProblem) -> FLSolution2:
        self.customers = sorted(problem.customers.values(), key=lambda c: c.demand, reverse=True)
        self.best_solution = MultiSolver(solvers=[GreedyPrefSolver(), GreedyDistSolver()])._solve(problem)
        self.best_value = self.best_solution.get_value()
        solution = FLSolution2(problem)
        self._bnb(solution, index=0)
        #self.best_solution.optimal = True  # I think that the relaxation can't be trusted
        return self.best_solution


class SASolver(FLSolver):
    def __init__(self, t0=100000000.0, alpha=0.999, improvement_limit=100000, debug=False):
        super().__init__()
        self.t0 = t0
        self.alpha = alpha
        self._stop = False
        self.best_solution = None
        self.debug = debug
        self.improvement_limit = improvement_limit

    def stop(self):
        self._stop = True
        return self.best_solution

    def _solve(self, problem: FLProblem):
        k = 0
        last_improvement = 0
        solution = RandSolver()._solve(problem) #MultiSolver(solvers=[GreedyPrefSolver(), GreedyDistSolver()])._solve(problem)
        solution_value = solution.get_value()
        solution.stats = Stats()
        solution.stats.undos = ([], [])
        self.best_solution = solution.copy()
        best_value = solution_value
        t = self.t0
        logger = logging.getLogger('solver')
        if self.debug:
            solution.stats.final_value = best_value

        while not self._stop and (k - last_improvement < self.improvement_limit):
            move_p = random.random()

            if move_p < 0.44:
                move_code = 0
            else:
                move_code = 1

            if move_code == 0:
                # random bind
                c = random.choice(problem.customers)
                f_i = random.choice(problem.facilities).index

                # Check capacity first
                if solution.capacities[f_i] < c.demand:
                    continue

                prev_f_i = solution.selections[c.index]
                solution.bind_customer(c.index, f_i)
                undo_params = (c.index, prev_f_i)
            elif move_code == 1:
                c_0 = random.choice(problem.customers).index
                c_1 = random.choice(problem.customers).index
                if c_0 == c_1:
                    continue

                c_0_f_i = solution.selections[c_0]
                c_1_f_i = solution.selections[c_1]

                # check if possible
                if solution.capacities[c_0_f_i] - problem.customers[c_0].demand < problem.customers[c_1].demand or \
                   solution.capacities[c_1_f_i] - problem.customers[c_1].demand < problem.customers[c_0].demand:
                    continue

                solution.bind_customer(c_0, c_1_f_i)
                solution.bind_customer(c_1, c_0_f_i)
                undo_params = (c_0, c_1, c_0_f_i, c_1_f_i)

            new_value = solution.get_value()
            prob = min(1 if round(solution_value - new_value, 1) > 0.0 else math.exp(-(new_value - solution_value) / t), 1)

            if prob >= random.random():
                if self.debug:
                    solution.stats.improvements_x.append(k)
                    solution.stats.improvements_y.append(new_value)

                # accept move
                solution_value = new_value

                if round(best_value - solution_value, 1) > 0.0:
                    self.best_solution = solution.copy()
                    best_value = new_value
                    last_improvement = k
                    if self.debug:
                        solution.stats.final_value = best_value

            else:  # undo move
                if self.debug:
                    solution.stats.undos[move_code].append(k)
                if move_code == 0:
                    solution.bind_customer(*undo_params)
                elif move_code == 1:
                    c_0, c_1, c_0_f_i, c_1_f_i = undo_params
                    solution.bind_customer(c_0, c_0_f_i)
                    solution.bind_customer(c_1, c_1_f_i)

            t = max(t * self.alpha, 0.1)
            k += 1

            if self.debug:
                solution.stats.probs.append(prob)
                solution.stats.temperature.append(t)

        logger.debug('alpha = {}, t0 = {}, k = {}, t = {}, k - last_improvement = {}'.format(
            self.alpha, self.t0, k, t, k - last_improvement))
        return self.best_solution


class FLMipSolver(FLSolver):
    def __init__(self, hide_output=False):
        super().__init__()
        self.timeout = None
        self.solution = None
        self._stop = False
        self.max_mem_mb = max(psutil.virtual_memory().free * 90 / (100 * 1024 ** 2), 1024)
        self.logger = logging.getLogger('solver')
        self.max_vars = 1000 * 1000
        self.hide_output = hide_output

    def stop(self):
        self._stop = True
        time.sleep(3)  # give it a little time for the solution to appear
        return self.solution

    def set_timeout(self, timeout: int):
        self.timeout = timeout

    @staticmethod
    def _get_vars_estimate(problem):
        return len(problem.customers) * len(problem.facilities)

    def _solve(self, problem: FLProblem):
        n_vars = self._get_vars_estimate(problem)
        self.logger.debug('{} vars estimated'.format(n_vars))
        if n_vars > self.max_vars:
            self.logger.error('Problem is too big for me!')
            return

        model = Model('facility')
        model.setRealParam("limits/memory", self.max_mem_mb)
        #model.setRealParam("limits/gap", 0.01)
        model.setEmphasis(SCIP_PARAMEMPHASIS.FEASIBILITY)  # detect feasibility fast

        if self.hide_output:
            model.hideOutput()

        if self.timeout:
            model.setRealParam("limits/time", self.timeout)

        x = {}  # facility f is open or closed
        y = {}  # facility f serves customer c

        for facility in problem.facilities.values():
            x[facility.index] = model.addVar(name='x_{}'.format(facility.index), vtype='B')

            for customer in problem.customers.values():
                y[facility.index, customer.index] = model.addVar(name='y_{},{}'.format(facility.index, customer.index),
                                                                 vtype='B')

        # a facility can serve a customer only if it is open
        for facility in problem.facilities.values():
            for customer in problem.customers.values():
                model.addCons(y[facility.index, customer.index] <= x[facility.index])

        # a customer must be served by exactly one facility
        for customer in problem.customers.values():
            model.addCons(quicksum(y[facility.index, customer.index] for facility in problem.facilities.values()) == 1)

        # the total demand must not exceed the facility capacity
        for facility in problem.facilities.values():
            f = facility.index
            model.addCons(quicksum(y[f, customer.index] * customer.demand for customer in problem.customers.values()) <=
                          facility.capacity)

        model.setObjective(quicksum(x[facility.index] * facility.setup_cost for facility in problem.facilities.values()) +
                           quicksum(
                               y[facility.index, customer.index] * customer.dists[facility.index]
                               for facility in problem.facilities.values() for customer in problem.customers.values()), "minimize")

        model.optimize()

        self.solution = FLSolution(problem)
        self.solution.selections = {}
        for c in problem.customers:
            assigned_customer = False
            for f in problem.facilities:
                val = model.getVal(y[f, c])
                if val > 0.5:
                    self.solution.selections[c] = f
                    assigned_customer = True
                    break
            assert assigned_customer

        #self.solution.optimal = not self._stop
        return self.solution


class FLMipSplitter(FLMipSolver):
    def __init__(self, *, hide_output=False, max_vars=None, is_root=True):
        super().__init__(hide_output=hide_output)
        self.is_root = is_root
        if max_vars is not None:
            self.max_vars = max_vars

    def set_timeout(self, timeout: int):
        self.timeout = max(timeout * 0.9, timeout - 3)
        self.logger.debug('timeout: {}'.format(self.timeout))

    def stop(self):
        if self.is_root:
            return super().stop()
        else:
            self._stop = True
            return self.solution

    def _solve(self, problem: FLProblem):
        t0 = time.time()
        n_vars = self._get_vars_estimate(problem)
        if n_vars > self.max_vars:
            self.logger.debug('{} vars is too much, splitting...'.format(n_vars))
            problems = split_problem(problem)
            solutions = []
            if self.timeout:
                time_budget = self.timeout / len(problems)
            else:
                time_budget = None

            for p in problems:
                solver = FLMipSplitter(hide_output=self.hide_output, max_vars=self.max_vars, is_root=False)
                solver.set_timeout(time_budget)
                solutions.append(solver._solve(p))
                time_used = time.time() - t0
                t0 = time.time()
                time_budget += time_budget - time_used if self.timeout else None
            merged = merge_solutions(solutions)
            return merged
        else:
            return super()._solve(problem)
