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
        for small_tour in small_tours:
            self.big_tour.extend(small_tour[1:])
            i -= 1
            self.big_tour[-1] = i
        self.big_tour.append(0)

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

        if tour_count < self.problem.max_vehicles:
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
        i = 0

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

