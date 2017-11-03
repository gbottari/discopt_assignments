import math
from tools.solver_tools import Solution, Solver
from collections import namedtuple

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
    def _solve(self, problem: FLProblem) -> FLSolution:
        # pack the facilities one by one until all the customers are served
        solution = FLSolution(problem)
        solution.selections = [-1] * len(problem.customers)
        capacity_remaining = [f.capacity for f in problem.facilities]

        facility_index = 0
        for customer in problem.customers:
            if capacity_remaining[facility_index] >= customer.demand:
                solution.selections[customer.index] = facility_index
                capacity_remaining[facility_index] -= customer.demand
            else:
                facility_index += 1
                assert capacity_remaining[facility_index] >= customer.demand
                solution.selections[customer.index] = facility_index
                capacity_remaining[facility_index] -= customer.demand

        return solution
