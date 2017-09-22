import os
import logging
import logging.handlers
from typing import List

LOGS_DIR = 'logs'


def create_logger(level):
    tg_logger = logging.getLogger('solver')
    tg_logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Console handler:
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    tg_logger.addHandler(ch)

    # File handler:
    os.makedirs(LOGS_DIR, exist_ok=True)
    fh = logging.handlers.RotatingFileHandler(os.path.join(LOGS_DIR, 'solver.log'), maxBytes=20 * 1024 * 1024,
                                              backupCount=1, encoding='utf-8')
    fh.setLevel(level)
    fh.setFormatter(formatter)
    tg_logger.addHandler(fh)
    return tg_logger


class Solution:
    def __repr__(self):
        return 'Solution<\n{}\n>'.format(self.serialize())

    def is_feasible(self) -> bool:
        raise NotImplementedError()

    def serialize(self) -> str:
        raise NotImplementedError()

    def get_value(self):
        raise NotImplementedError()

    def is_optimal(self) -> bool:
        return False


class Solver:
    def __repr__(self):
        return '<{}>'.format(self.__class__.__name__)

    def _parse(self, raw_input_data: str):
        raise NotImplementedError()

    def _solve(self, input_data) -> Solution:
        raise NotImplementedError()

    def solve(self, raw_input_data: str):
        return self._solve(self._parse(raw_input_data))


class MultiSolver(Solver):
    def __init__(self, solvers: List[Solver]):
        self.solvers = solvers
        assert(len(self.solvers) > 0)

    def __repr__(self):
        return '<{}({})>'.format(self.__class__.__name__, ', '.join(repr(s) for s in self.solvers))

    def _parse(self, raw_input_data: str):
        return self.solvers[0]._parse(raw_input_data)

    def solve(self, raw_input_data: str):
        input_data = self._parse(raw_input_data)
        best_solution = None
        last_exception = None
        for solver in self.solvers:
            try:
                solution = solver._solve(input_data)
            except Exception as ex:
                logging.getLogger('solver').exception('{} encoutered an exception with {}: '.format(self, solver))
                last_exception = ex
                continue

            if best_solution is None or solution.get_value() >= best_solution.get_value():
                best_solution = solution
                if best_solution.is_optimal():
                    break

        if best_solution is None and last_exception is not None:
            raise last_exception
        return best_solution


class SolverManager:
    _shared_state = {}

    def __new__(cls):
        """
        This class uses the borg pattern. See: http://design-patterns-ebook.readthedocs.io/en/latest/creational/borg/.
        :return: a shared instance from the borg.
        """
        inst = super().__new__(cls)  # Uses the super class implementation of __new__() to create an instance.
        inst.__dict__ = cls._shared_state  # Assigns the shared state to the instance.
        if not cls._shared_state:
            # We create the instance once, when the shared state is empty.
            inst.logger = create_logger(logging.DEBUG)
            inst.logger.info('New SolverManager created.')

        return inst

    def solve(self, raw_input_data: str, solver: Solver) -> str:
        self.logger.info('Using {}'.format(solver))
        solution = solver.solve(raw_input_data)
        if not solution.is_feasible():
            raise Exception('Solution is not feasible!')
        self.logger.info('Solution value: {}'.format(solution.get_value()))
        return solution.serialize()
