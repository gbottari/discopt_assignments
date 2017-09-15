class Solution:
    def __repr__(self):
        return 'Solution<\n{}\n>'.format(self.serialize())

    def is_feasible(self) -> bool:
        raise NotImplementedError()

    def serialize(self) -> str:
        raise NotImplementedError()


class Solver:
    def _parse(self, raw_input_data: str):
        raise NotImplementedError()

    def _solve(self, input_data) -> Solution:
        raise NotImplementedError()

    def solve(self, raw_input_data: str):
        return self._solve(self._parse(raw_input_data))


def solve_and_serialize(raw_input_data: str, solver: Solver) -> str:
    solution = solver.solve(raw_input_data)
    if not solution.is_feasible():
        raise Exception('Solution is not feasible!')
    return solution.serialize()
