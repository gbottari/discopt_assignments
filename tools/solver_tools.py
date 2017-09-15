class Solution:
    def __repr__(self):
        return 'Solution<\n{}\n>'.format(self.serialize())

    def is_feasible(self) -> bool:
        raise NotImplementedError()

    def serialize(self) -> str:
        raise NotImplementedError()


class Solver:
    def _parse(self, raw_input_data):
        raise NotImplementedError()

    def _solve(self, input_data) -> Solution:
        raise NotImplementedError()

    def solve(self, raw_input_data) -> Solution:
        return self._solve(self._parse(raw_input_data))