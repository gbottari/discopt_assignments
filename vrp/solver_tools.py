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
        customer_set = set()
        for small_tour in self.get_small_tours():
            # demand must not exceed capacity
            total_demand = sum(get_cs(c_i).demand for c_i in small_tour)
            if total_demand > self.problem.capacity:
                return False
            tour_count += 1

            for c_i in small_tour:
                if is_not_warehouse(c_i) and c_i in customer_set:
                    return False
            customer_set.update(small_tour)
            customer_set.remove(0)

        if len(customer_set) + 1 != len(self.problem.customers):
            return False

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
        inst.stats = self.stats
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
            found_vehicle = False
            vehicles = list(range(problem.max_vehicles))
            random.shuffle(vehicles)
            for i in vehicles:
                # check capacity
                if customer.demand <= remaining_capacity[i]:
                    remaining_capacity[i] -= customer.demand
                    tours[i].append(customer.index)
                    found_vehicle = True
                    break
            if not found_vehicle:
                return self._solve(problem)

        # finish tours
        for tour in tours:
            tour.append(0)

        solution = VRPSolution(problem)
        solution.from_small_tours(tours)
        return solution


class RandomStableVRPSolver(VRPSolver):
    def _solve(self, problem: VRPProblem):
        tours: List[List[int]] = [[] for _ in range(problem.max_vehicles)]
        remaining_capacity = [problem.capacity] * problem.max_vehicles
        vehicles = list(range(problem.max_vehicles))

        for customer in sorted(problem.customers[1:], key=lambda c: c.demand, reverse=True):
            found_vehicle = False
            random.shuffle(vehicles)
            for i in vehicles:
                # check capacity
                if customer.demand <= remaining_capacity[i]:
                    remaining_capacity[i] -= customer.demand
                    tours[i].append(customer.index)
                    found_vehicle = True
                    break
            if not found_vehicle:
                return self._solve(problem)

        # finish tours
        for tour in tours:
            random.shuffle(tour)
            tour.insert(0, 0)
            tour.append(0)

        solution = VRPSolution(problem)
        solution.from_small_tours(tours)
        return solution


# class TrueRandomVRPSolver(VRPSolver):
#     def _solve(self, problem: VRPProblem):
#         remaining_capacity = [problem.capacity] * problem.max_vehicles
#         unallocated_customers = problem.customers[1:]
#         allocated_customers = {}  # customer : vehicle
#
#         while True:
#             customer = random.choice(unallocated_customers)
#             found_vehicle = False
#             vehicles = list(range(problem.max_vehicles))
#             random.shuffle(vehicles)
#             for i in vehicles:
#                 # check capacity
#                 if customer.demand <= remaining_capacity[i]:
#                     remaining_capacity[i] -= customer.demand
#                     allocated_customers[customer] = i
#                     found_vehicle = True
#                     break
#             if not found_vehicle:
#                 # unallocate someone
#                 customer, v_i = random.choice(list(allocated_customers.items()))
#                 remaining_capacity[v_i] += customer.demand
#                 del allocated_customers[customer]
#                 unallocated_customers.append(customer)
#
#         tours = []
#
#         solution = VRPSolution(problem)
#         solution.from_small_tours(tours)
#         return solution


class LS2OptVRPSolver(VRPSolver):
    def __init__(self, max_iters=1000, initial_solution=None, debug=False):
        self._stop = False
        self.best_solution = initial_solution
        self.best_value = initial_solution.get_value() if initial_solution else None
        self.max_iters = max_iters
        self.debug = debug

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
        if self.best_solution is None:
            self.best_solution = RandomStableVRPSolver()._solve(problem)
            self.best_value = self.best_solution.get_value()
            self.best_solution.stats = Stats()

        last_x = 0 if not self.best_solution.stats.improvements_x else self.best_solution.stats.improvements_x[-1]

        for k in range(self.max_iters):
            i, j = self.get_random_swap_indexes(self.best_solution)
            new_value = self.improve(self.best_solution, self.best_value, i, j)
            if new_value < self.best_value:
                self.best_value = new_value
                if self.debug:
                    stats = self.best_solution.stats
                    stats.improvements_x.append(last_x + k)
                    stats.improvements_y.append(new_value)
                    stats.final_value = new_value

            if self._stop:
                break

        return self.best_solution


class SASolver(VRPSolver):
    def __init__(self, t_min=3, t_max=100000000.0, alpha=0.999, improvement_limit=100000, debug=False):
        super().__init__()
        self.t_max = t_max
        self.t_min = t_min
        self.t_factor = -math.log(self.t_max / self.t_min)
        self.alpha = alpha
        self._stop = False
        self.best_solution = None
        self.debug = debug
        self.improvement_limit = improvement_limit

    def __repr__(self):
        return '<{}(t_min={}, t_max={}, alpha={}, improvement_limit={})>'.format(
            self.__class__.__name__, self.t_min, self.t_max, self.alpha, self.improvement_limit)

    def stop(self):
        self._stop = True
        return self.best_solution

    def _solve(self, problem: VRPProblem):
        k = 0
        last_improvement = 0
        solution = RandomStableVRPSolver()._solve(problem)
        solution.stats = Stats()

        self.best_solution = solution.copy()
        ls_solver = LS2OptVRPSolver()

        solution_value = best_value = solution.get_value()
        t = self.t_max
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

            # interesting, like re-heats
            #step = k - last_improvement
            #max_steps = self.improvement_limit / 10
            #t = self.t_max * math.exp(self.t_factor * step / max_steps)

            #step = k
            #max_steps = 100000
            #t = max(t * math.exp(self.t_factor * (1 - self.alpha)), self.t_min)

            t = max(t * self.alpha, self.t_min)

            k += 1

            if self.debug:
                solution.stats.probs.append(prob)
                solution.stats.temperature.append(t)

        logger.debug('alpha = {}, t0 = {}, k = {}, t = {}, k - last_improvement = {}'.format(
            self.alpha, self.t_max, k, t, k - last_improvement))
        return self.best_solution


class ILSVRPSolver(VRPSolver):
    def __init__(self, max_diving_iters=10000, max_dives=10, debug=False):
        self.max_dives = max_dives
        self.max_diving_iters = max_diving_iters
        self._stop = False
        self.best_solution = None
        self.best_value = None
        self.debug = debug

    def __repr__(self):
        return '<{}(max_diving_iters={}, max_dives={})>'.format(self.__class__.__name__, self.max_diving_iters,
                                                                self.max_dives)

    def stop(self):
        self._stop = True
        return self.best_solution

    def _solve(self, problem: VRPProblem):
        stats = Stats()

        for k in range(self.max_dives):
            solver = LS2OptVRPSolver(max_iters=self.max_diving_iters)
            solution = solver._solve(problem)
            solution.stats = stats
            solution_value = solution.get_value()
            if self.best_solution is None or solution_value < self.best_value:
                self.best_solution = solution
                self.best_value = solution_value
                if self.debug:
                    stats.improvements_x.append(k)
                    stats.improvements_y.append(solution_value)
                    stats.final_value = solution_value

        #TourImprover(initial_solution=self.best_solution)._solve(problem)
        #assert self.best_solution.is_feasible()

        return self.best_solution


class ILSVRPExplorerSolver(VRPSolver):
    def __init__(self, max_diving_iters=10000, max_dives=10, max_gap=1.10, max_good_sols=3, debug=False):
        self.max_dives = max_dives
        self.max_diving_iters = max_diving_iters
        self._stop = False
        self.best_solution = None
        self.best_value = None
        self.good_solutions = []
        self.debug = debug
        self.max_gap = max_gap
        self.max_good_sols = max_good_sols

    def __repr__(self):
        return '<{}(max_diving_iters={}, max_dives={}, max_gap={}, max_good_sols={})>'.format(self.__class__.__name__,
            self.max_diving_iters, self.max_dives, self.max_gap, self.max_good_sols)

    def stop(self):
        self._stop = True
        return self.best_solution

    def _solve(self, problem: VRPProblem):
        stats = Stats()

        self.best_solution = RandomStableVRPSolver()._solve(problem)
        self.best_value = self.best_solution.get_value()

        for k in range(self.max_dives):
            if self._stop:
                break

            solver = LS2OptVRPSolver(max_iters=self.max_diving_iters)
            solution = solver._solve(problem)
            solution.stats = stats
            solution_value = solution.get_value()
            if solution_value / self.best_value < self.max_gap:
                self.good_solutions.append((solution_value, solution))
                if len(self.good_solutions) > self.max_good_sols:
                    # look for a solution with the highest value
                    highest_index = 0
                    for i in range(1, len(self.good_solutions)):
                        if self.good_solutions[i][0] > self.good_solutions[highest_index][0]:
                            highest_index = i
                    self.good_solutions.pop(highest_index)

                if solution_value < self.best_value:
                    self.best_solution = solution
                    self.best_value = solution_value
                    if self.debug:
                        stats.improvements_x.append(k)
                        stats.improvements_y.append(solution_value)
                        stats.final_value = solution_value

        # dive deep on the good solutions found
        print('good sols = {}; best_value = {:.1f}'.format(len(self.good_solutions), self.best_value))
        #print('good sols = {}'.format([slv for slv, _ in self.good_solutions]))
        init_val_winner = self.best_value
        for init_val, solution in self.good_solutions:
            if self._stop:
                break
            solver = LS2OptVRPSolver(max_iters=min(self.max_diving_iters * 20, 80000), initial_solution=solution)
            solution = solver._solve(problem)
            solution_value = solution.get_value()

            if solution_value < self.best_value:
                self.best_value = solution_value
                self.best_solution = solution
                init_val_winner = init_val

        print('solution with the value that turn out to be the best had: {}'.format(init_val_winner))

        return self.best_solution


class ILSVRPSolver2(VRPSolver):
    def __init__(self, initial_depth=1000, max_failed_dives=100, depth_multiplier=1.1, refinement_loops=10000, debug=False):
        self.max_failed_dives = max_failed_dives
        self._stop = False
        self.best_solution = None
        self.best_value = None
        self.debug = debug
        self.depth_multiplier = depth_multiplier
        self.refinement_loops = refinement_loops
        self.initial_depth = initial_depth

    def __repr__(self):
        return '<{}(max_failed_dives={}, depth_multiplier={}, refinement_loops={}, initial_depth={})>'.format(
            self.__class__.__name__, self.max_failed_dives, self.depth_multiplier, self.refinement_loops,
            self.initial_depth)

    def _solve(self, problem: VRPProblem):
        stats = Stats()
        stats.dives_x = []
        stats.dives_y = []
        depth = self.initial_depth
        last_x = 0

        # main loop with dives
        failed_dives = 0
        k = 0
        while not self._stop and failed_dives < self.max_failed_dives:
            dive_depth = int(depth)
            solver = LS2OptVRPSolver(max_iters=dive_depth, debug=self.debug)
            solution = solver._solve(problem)
            solution_value = solution.get_value()

            if self.debug and self.best_value is not None:
                stats.dives_x.append(last_x)
                stats.dives_y.append(self.best_value)

            if self.best_solution is None or solution_value < self.best_value:
                if self.debug:
                    # merge stats
                    new_x = [(x + last_x + 1) for x in solution.stats.improvements_x]
                    stats.improvements_x += new_x
                    stats.improvements_y += solution.stats.improvements_y
                    stats.final_value = solution_value

                self.best_solution = solution
                self.best_solution.stats = stats
                self.best_value = solution_value
                failed_dives = 0
            else:
                failed_dives += 1
                depth *= self.depth_multiplier  # when we are struggling, increase depth

            last_x += dive_depth
            k += 1

        print('k: {}; final depth: {}; before refinement: {:.1f}'.format(k, depth, self.best_value))

        # Refinement step at the end
        if not self._stop and self.refinement_loops > 0:
            solver = LS2OptVRPSolver(max_iters=self.refinement_loops, initial_solution=self.best_solution,
                                     debug=self.debug)
            solution = solver._solve(problem)
            solution.get_value()

        return self.best_solution


class SASolver2(SASolver):
    def __init__(self, t_min=3, t_max=100000000.0, alpha=0.999, improvement_limit=100000, debug=False,
                 refinement_loops=1000):
        super().__init__(t_min, t_max, alpha, improvement_limit, debug)
        self.refinement_loops = refinement_loops

    def __repr__(self):
        return '<{}(t_min={}, t_max={}, alpha={}, improvement_limit={}, refinement_loops={})>'.format(
            self.__class__.__name__, self.t_min, self.t_max, self.alpha, self.improvement_limit, self.refinement_loops)

    def _solve(self, problem: VRPProblem):
        super()._solve(problem)

        if not self._stop:
            # improvement step
            imp_solver = LS2OptVRPSolver(max_iters=self.refinement_loops, initial_solution=self.best_solution,
                                         debug=self.debug)
            imp_solver._solve(problem)
            self.best_solution.get_value()

        return self.best_solution


class ILSVRPBaggerSolver(VRPSolver):
    def __init__(self, max_diving_iters=10000, max_dives=10, max_gap=1.50, max_good_sols=100, debug=False):
        self.max_dives = max_dives
        self.max_diving_iters = max_diving_iters
        self._stop = False
        self.best_solution = None
        self.best_value = None
        self.worst_value = None
        self.good_solutions = []
        self.debug = debug
        self.max_gap = max_gap
        self.max_good_sols = max_good_sols

    def __repr__(self):
        return '<{}(max_diving_iters={}, max_dives={}, max_gap={}, max_good_sols={})>'.format(self.__class__.__name__,
            self.max_diving_iters, self.max_dives, self.max_gap, self.max_good_sols)

    def stop(self):
        self._stop = True
        return self.best_solution

    def _solve(self, problem: VRPProblem):
        stats = Stats()

        self.good_solutions = []
        for _ in range(self.max_good_sols):
            solution = RandomStableVRPSolver()._solve(problem)
            solution.stats = stats
            solution_value = solution.get_value()
            self.good_solutions.append((solution_value, solution))

        #self.best_value = min(v for v, _ in self.good_solutions)
        #self.worst_value = max(v for v, _ in self.good_solutions)
        #self.best_solution = [s for v, s in self.good_solutions if v == self.best_value][0]

        for k in range(self.max_dives):
            if self._stop:
                break

            # apply improvement step to all solutions
            new_good_solutions = []
            for _, solution in self.good_solutions:
                solver = LS2OptVRPSolver(max_iters=self.max_diving_iters, initial_solution=solution)
                solver._solve(problem)
                solution_value = solution.get_value()
                new_good_solutions.append((solution_value, solution))

            self.good_solutions = new_good_solutions
            self.best_value = min(v for v, _ in self.good_solutions)
            self.worst_value = max(v for v, _ in self.good_solutions)
            self.best_solution = [s for v, s in self.good_solutions if v == self.best_value][0]

            # remove solutions that have a bad gap
            new_good_solutions = []
            for solution_value, solution in self.good_solutions:
                gap = solution_value / self.best_value
                if gap < self.max_gap:
                    new_good_solutions.append((solution_value, solution))

            self.good_solutions = new_good_solutions
            self.max_gap = max(self.max_gap * 0.995, 1.03)

            #print('k = {:3d}, good sols = {:3d}, best= {:.1f}, worst = {:.1f}, mg = {:.2f}'.format(
            #    k, len(self.good_solutions), self.best_value, self.worst_value, self.max_gap))

        print('bagger; good sols = {}; best_value = {:.1f}'.format(len(self.good_solutions), self.best_value))

        return self.best_solution


class TourImprover(VRPSolver):
    def __init__(self, initial_solution=None):
        self.best_solution = initial_solution

    def _solve(self, problem: VRPProblem):
        if not self.best_solution:
            self.best_solution = RandomStableVRPSolver()._solve(problem)

        solver = LS2OptVRPSolver()
        solution_value = self.best_solution.get_value()

        big_tour = self.best_solution.big_tour
        get_c = self.best_solution.problem.get_customer
        for i in range(len(big_tour)):
            source_id = self.best_solution.tour_ids[i]
            source_c = get_c(big_tour[i])
            if is_warehouse(source_c.index):
                continue

            for j in range(i + 1, len(big_tour)):
                target_id = self.best_solution.tour_ids[j]
                target_c = get_c(big_tour[i])
                if target_id == source_id or is_warehouse(target_c.index):
                #if is_warehouse(target_c.index):
                    continue

                new_solution_value = solver.improve(self.best_solution, solution_value, i, j)
                if new_solution_value < solution_value:
                    solution_value = new_solution_value

        return self.best_solution
