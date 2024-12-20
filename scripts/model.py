import numpy as np
from gamma import HiddenMarkovLayer
from probability_matrix import ProbabilityMatrix
from probability_vector import ProbabilityVector


class HiddenMarkovModel:
    def __init__(self, hml: HiddenMarkovLayer):
        self.layer = hml
        self._score_init = 0
        self.score_history = []

    @classmethod
    def initialize(cls, possible_states: list, observables: list):
        layer = HiddenMarkovLayer.initialize(possible_states, observables)
        return cls(layer)

    def update(self, observations: list) -> float:
        alpha = self.layer._alphas(observations)
        beta = self.layer._betas(observations)

        # NxN
        digamma = self.layer._digammas(observations)
        score = alpha[-1].sum()
        gamma = alpha * beta / score

        T = len(alpha)
        obs_idx = [self.layer.observables.index(x) for x in observations]
        capture = np.zeros(
            (T, len(self.layer.states), len(self.layer.observables)))

        for t in range(T):
            capture[t, :, obs_idx[t]] = 1.0

        pi = gamma[0]
        T = digamma.sum(axis=0) / gamma[:-1].sum(axis=0).reshape(-1, 1)
        E = (capture * gamma[:, :, np.newaxis]
             ).sum(axis=0) / gamma.sum(axis=0).reshape(-1, 1)

        self.layer.pi = ProbabilityVector.from_numpy(pi, self.layer.states)
        self.layer.T = ProbabilityMatrix.from_numpy(
            T, self.layer.states, self.layer.states)
        self.layer.E = ProbabilityMatrix.from_numpy(
            E, self.layer.states, self.layer.observables)

        return score

    def train(self, observations: list, epochs: int, tol=None):
        self._score_init = 0
        self.score_history = (epochs + 1) * [0]
        early_stopping = isinstance(tol, (int, float))

        for epoch in range(1, epochs + 1):
            score = self.update(observations)
            print("Training... epoch = {} out of {}, score = {}".format(
                epoch, epochs, score))

            if early_stopping and abs(self._score_init - score) / score < tol:
                print("Early stopping.")
                break
            self._score_init = score
            self.score_history[epoch] = score
