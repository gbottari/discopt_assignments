import math
import time
import random
import logging
import itertools
from collections import namedtuple
from itertools import chain
from typing import Optional, List, Sequence, Iterable, Set
#from pyscipopt import Model, quicksum, SCIP_PARAMEMPHASIS

import sys
sys.path.append('..')

from tools.solver_tools import Solution, Solver, MultiSolver

Point = namedtuple("Point", ['x', 'y'])
Customer = namedtuple("Customer", ['index', 'demand', 'location'])


def is_warehouse(c_i):
    return c_i <= 0


def is_not_warehouse(c_i):
    return c_i > 0


class VRPProblem:
    def __init__(self, customers: List[Customer], capacity, max_vehicles):
        self.customers = customers
        self.capacity = capacity
        self.max_vehicles = max_vehicles
        self.dist_cache = dict()

    def dist(self, p1: Point, p2: Point) -> float:
        key = frozenset((p1, p2))
        if key not in self.dist_cache:
            dist = math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)
            self.dist_cache[key] = dist
        else:
            dist = self.dist_cache[key]
        return dist

    def get_customer(self, index):
        # trick to make the big circuit work, every <= 0 customer points to the warehouse
        return self.customers[max(index, 0)]


class Stats:
    def __init__(self):
        self.iterations = 0
        self.improvements_x = []
        self.improvements_y = []
        self.temperature = []
        self.probs = []
        self.initial_value = 0.0
        self.final_value = 0.0


class VRPSolution(Solution):
    def __init__(self, problem: VRPProblem):
        super().__init__()
        self.problem = problem
        self.big_tour = []
        self.tour_ids = []
        self.capacities = [problem.capacity] * problem.max_vehicles
        self.optimal = False

    def get_small_tours(self):
        i = 1
        while i < len(self.big_tour):
            tour = [0] + [c_i for c_i in itertools.takewhile(is_not_warehouse, self.big_tour[i:])] + [0]
            i += len(tour) - 1
            yield tour

    def from_small_tours(self, small_tours):
        i = 0
        self.big_tour = [0]
        self.tour_ids = []
        self.capacities = []
        tour_id = 0
        for small_tour in small_tours:
            self.big_tour.extend(small_tour[1:])
            self.tour_ids.extend([tour_id] * (len(small_tour) - 1))
            tour_id += 1
            i -= 1
            self.big_tour[-1] = i
            self.capacities.append(self.problem.capacity -
                                   sum(self.problem.customers[c_i].demand for c_i in small_tour))
        self.tour_ids.append(tour_id - 1)
        self.big_tour[-1] = 0

    def from_big_tour(self, big_tour):
        self.big_tour = big_tour
        self.tour_ids = [0] * len(big_tour)
        self.capacities = [self.problem.capacity] * self.problem.max_vehicles
        tour_id = 0
        for i in range(len(big_tour)):
            c_i = big_tour[i]
            if is_warehouse(c_i) and c_i != 0:
                tour_id += 1
            self.tour_ids[i] = tour_id
            self.capacities[tour_id] -= self.problem.get_customer(c_i).demand

    def next_c_i_in_tour(self, index) -> int:
        curr_tour_id = self.tour_ids[index]
        if index + 1 == len(self.tour_ids):
            start_index = self.tour_ids.index(curr_tour_id) + 1
            return self.big_tour[start_index]

        next_tour_id = self.tour_ids[index + 1]
        if next_tour_id != curr_tour_id:
            start_index = self.tour_ids.index(curr_tour_id)
            return self.big_tour[start_index]
        else:
            return self.big_tour[index + 1]

    def prev_c_i_in_tour(self, index) -> int:
        curr_tour_id = self.tour_ids[index]
        if index == 0:
            end_index = len(self.tour_ids) - self.tour_ids[::-1].index(curr_tour_id) - 1
            return self.big_tour[end_index]

        prev_tour_id = self.tour_ids[index - 1]
        if prev_tour_id != curr_tour_id:
            if curr_tour_id == self.problem.max_vehicles - 1:
                end_index = self.problem.max_vehicles - 1
            else:
                end_index = len(self.tour_ids) - self.tour_ids[::-1].index(curr_tour_id) - 1
            return self.big_tour[end_index]
        else:
            return self.big_tour[index - 1]

    def get_value(self):
        get_c = self.problem.get_customer
        return sum(self.problem.dist(get_c(self.big_tour[i]).location, get_c(self.big_tour[i + 1]).location)
                   for i in range(len(self.big_tour) - 1))

    def is_feasible(self):
        # every big_tour must start and end with <= 0 (the warehouse)
        if len(self.big_tour) < 2 or is_not_warehouse(self.big_tour[0]) or is_not_warehouse(self.big_tour[-1]):
            return False

        get_cs = self.problem.get_customer
        tour_count = 0
        for small_tour in self.get_small_tours():
            # demand must not exceed capacity
            total_demand = sum(get_cs(c_i).demand for c_i in small_tour)
            if total_demand > self.problem.capacity:
                return False
            tour_count += 1

        if tour_count != self.problem.max_vehicles:
            return False

        return True

    def is_optimal(self):
        return self.optimal

    def serialize(self) -> str:
        # note: truncates the value
        obj_value = self.get_value()
        return "{:.1f} {}\n{}".format((obj_value * 10) / 10, int(self.optimal), "\n".join(" ".join(
            str(max(c_i, 0)) for c_i in tour) for tour in self.get_small_tours()))

    def is_better(self, other: 'VRPSolution') -> bool:
        if self.is_optimal():
            return True
        if other.is_optimal():
            return False
        return self.get_value() < other.get_value()

    def copy(self):
        inst = VRPSolution(self.problem)
        inst.big_tour = self.big_tour.copy()
        inst.capacities = self.capacities.copy()
        inst.tour_ids = self.tour_ids.copy()
        inst.optimal = self.optimal
        return inst


class VRPSolver(Solver):
    def _parse(self, raw_input_data: str):
        # parse the input
        lines = raw_input_data.split('\n')

        parts = lines[0].split()
        customer_count = int(parts[0])
        vehicle_count = int(parts[1])
        vehicle_capacity = int(parts[2])

        customers = []
        for i in range(1, customer_count + 1):
            line = lines[i]
            parts = line.split()
            customers.append(Customer(index=i - 1, demand=int(parts[0]),
                                      location=Point(x=float(parts[1]), y=float(parts[2]))))

        return VRPProblem(customers=customers, capacity=vehicle_capacity, max_vehicles=vehicle_count)


class RandomVRPSolver(VRPSolver):
    def _solve(self, problem: VRPProblem):
        tours: List[List[int]] = [[0] for _ in range(problem.max_vehicles)]
        remaining_capacity = [problem.capacity] * problem.max_vehicles
        customers = problem.customers[1:]
        random.shuffle(customers)

        for customer in customers:
            for i in range(problem.max_vehicles):
                # check capacity
                if customer.demand <= remaining_capacity[i]:
                    remaining_capacity[i] -= customer.demand
                    tours[i].append(customer.index)
                    break

        # finish tours
        for tour in tours:
            tour.append(0)

        solution = VRPSolution(problem)
        solution.from_small_tours(tours)
        return solution


class LS2OptVRPSolver(VRPSolver):
    def __init__(self, max_iters=1000):
        self._stop = False
        self.best_solution = None
        self.best_value = None
        self.max_iters = max_iters

    def __repr__(self):
        return '<{}(max_iters={})>'.format(self.__class__.__name__, self.max_iters)

    def stop(self) -> VRPSolution:
        self._stop = True
        time.sleep(0.2)
        return self.best_solution

    def perform_2opt(self, solution, i, j, same_tour=False):
        length = (j - i + 1) // 2
        for k in range(length):
            solution.big_tour[i + k], solution.big_tour[j - k] = solution.big_tour[j - k], solution.big_tour[i + k]

        # the tour ids will remain the same if same_tour
        if not same_tour:
            solution.from_big_tour(solution.big_tour)

    def _check_demand(self, solution, i, j):
        get_c = solution.problem.get_customer
        new_capacities = solution.capacities.copy()

        t_id_1 = solution.tour_ids[i]
        t_id_2 = solution.tour_ids[j]

        if t_id_1 == t_id_2:
            return True

        for k in range(i, j):
            if is_warehouse(solution.big_tour[k]):
                break

            c = get_c(solution.big_tour[k])
            new_capacities[t_id_1] += c.demand
            new_capacities[t_id_2] -= c.demand

        for k in range(j, i - 1, -1):
            if is_warehouse(solution.big_tour[k]):
                break

            c = get_c(solution.big_tour[k])
            new_capacities[t_id_2] += c.demand
            new_capacities[t_id_1] -= c.demand

        # unfeasible by demand
        return all(c >= 0 for c in new_capacities)

    def get_random_swap_indexes(self, solution):
        n = len(solution.problem.customers)

        while True:
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)

            if i > j:
                i, j = j, i

            c_i = solution.big_tour[i]
            c_j = solution.big_tour[j]
            if i == j or is_warehouse(c_i) or is_warehouse(c_j):
                continue

            return i, j

    def calc_sol_value(self, solution, solution_value, i, j):
        get_c = solution.problem.get_customer
        dist = solution.problem.dist

        c_i = solution.big_tour[i]
        c_j = solution.big_tour[j]

        next_j = solution.next_c_i_in_tour(j)
        prev_i = solution.prev_c_i_in_tour(i)

        point_i = get_c(c_i).location
        point_j = get_c(c_j).location
        point_next_j = get_c(next_j).location
        point_prev_i = get_c(prev_i).location

        solution_value -= dist(point_prev_i, point_i)
        solution_value -= dist(point_j, point_next_j)
        solution_value += dist(point_prev_i, point_j)
        solution_value += dist(point_i, point_next_j)

        return solution_value

    def improve(self, solution: VRPSolution, solution_value, i, j):
        # we don't need to perform 2-OPT to know the value
        new_value = self.calc_sol_value(solution, solution_value, i, j)

        # don't waste time performing 2-OPT if the solution would be worse
        if new_value > solution_value:
            return solution_value

        inside_same_tour = solution.tour_ids[i] == solution.tour_ids[j]

        if not inside_same_tour:
            if not self._check_demand(solution, i, j):
                return solution_value

        self.perform_2opt(solution, i, j, same_tour=inside_same_tour)

        return new_value

    def _solve(self, problem: VRPProblem):
        self.best_solution = RandomVRPSolver()._solve(problem)
        self.best_value = self.best_solution.get_value()

        for _ in range(self.max_iters):
            i, j = self.get_random_swap_indexes(self.best_solution)
            new_value = self.improve(self.best_solution, self.best_value, i, j)
            if new_value < self.best_value:
                self.best_value = new_value
            if self._stop:
                break

        return self.best_solution


class SASolver(VRPSolver):
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

    def _solve(self, problem: VRPProblem):
        k = 0
        last_improvement = 0
        solution = RandomVRPSolver()._solve(problem)
        solution.stats = Stats()

        self.best_solution = solution.copy()
        ls_solver = LS2OptVRPSolver()

        solution_value = best_value = solution.get_value()
        t = self.t0
        logger = logging.getLogger('solver')
        if self.debug:
            solution.stats.final_value = best_value

        while not self._stop and (k - last_improvement < self.improvement_limit):
            i, j = ls_solver.get_random_swap_indexes(solution)
            inside_same_tour = solution.tour_ids[i] == solution.tour_ids[j]

            if not inside_same_tour and not ls_solver._check_demand(solution, i, j):
                continue

            new_value = ls_solver.calc_sol_value(solution, solution_value, i, j)
            prob = min(1 if round(solution_value - new_value, 1) > 0.0 else math.exp(-(new_value - solution_value) / t), 1)

            if prob >= random.random():
                if self.debug:
                    solution.stats.improvements_x.append(k)
                    solution.stats.improvements_y.append(new_value)

                # accept move
                solution_value = new_value
                ls_solver.perform_2opt(solution, i, j, same_tour=inside_same_tour)

                if round(best_value - solution_value, 1) > 0.0:
                    self.best_solution = solution.copy()
                    best_value = new_value
                    last_improvement = k
                    if self.debug:
                        solution.stats.final_value = best_value

            t = max(t * self.alpha, 0.1)
            k += 1

            if self.debug:
                solution.stats.probs.append(prob)
                solution.stats.temperature.append(t)

        logger.debug('alpha = {}, t0 = {}, k = {}, t = {}, k - last_improvement = {}'.format(
            self.alpha, self.t0, k, t, k - last_improvement))
        return self.best_solution
