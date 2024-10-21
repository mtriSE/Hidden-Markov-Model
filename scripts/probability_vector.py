import numpy as np
import pandas as pd

class ProbabilityVector:
    """Vector observations"""
    def __init__(self, probabilities: dict):
        states = probabilities.keys()
        probs = probabilities.values()

        assert len(states) == len(probs), \
            "Probabilities must match states."
        assert len(states) == len(set(states)), \
            "The states must be unique"
        assert abs(sum(probs) - 1.0) < 1e-12, \
            "Probabilities must sum up to 1"
        assert len(list(filter(lambda x: 0 <= x <= 1, probs))) == len(probs), \
            "Probabilities must be numbers from [0, 1] interval."

        self.states = sorted(states)
        self.values = np.array(
            list(map(lambda x: probabilities[x], self.states))).reshape(1, -1)

    @classmethod
    def initialize(cls, states: list):
        """initialize probabilities of states"""
        size = len(states)
        rand = np.random.rand(size) / (size**2) + 1 / size
        rand /= np.sum(rand, axis=0)
        return cls(dict(zip(states, rand)))

    @classmethod
    def from_numpy(cls, array: np.ndarray, state: list):
        return cls(dict(zip(state, list(array))))

    @property
    def dict(self):
        return {k: v for k, v in zip(self.states, list(self.values.flatten()))}
    
    @property
    def df(self):
        return pd.DataFrame(self.values, columns=self.states, index=['probabilities'])

    def __repr__(self):
        return "P({}) = {}.".format(self.states, self.values)

    def __eq__(self, other):
        if not isinstance(other, ProbabilityVector):
            raise NotImplementedError
        if (self.states == other.states and (self.values == other.values).all):
            return True
        return False

    def __getitem__(self, state: str) -> float:
        if state not in self.states:
            raise ValueError("Requesting unknown probaility state from vector")
        index = self.states.index(state)
        return float(self.values[0, index])

    def __mul__(self, other) -> np.ndarray:
        if isinstance(other, ProbabilityVector):
            return self.values * other.values
        elif isinstance(other, (int, float)):
            return self.values * other
        else:
            raise NotImplementedError

    def __rmul__(self, other) -> np.ndarray:
        return self.__mul__(other)

    def __matmul__(self, other) -> np.ndarray:
        if isinstance(other, ProbabilityVector):
            return self.values @ other.values

    def __truediv__(self, number) -> np.ndarray:
        if isinstance(number, (int, float)):
            raise NotImplementedError
        x = self.values
        return x / number if number != 0 else x / (number + 1e-12)
    
    def argmax(self):
        index = self.values.argmax()
        return self.states[index]
