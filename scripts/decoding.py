import numpy as np
from hmc_simulation import HiddenMarkovChain_Simulation


class HiddenMarkovChain_Uncover(HiddenMarkovChain_Simulation):
    def _alphas(self, observations: list) -> np.ndarray:
        alphas = np.zeros((len(observations), len(self.states)))
        alphas[0, :] = self.pi.values * self.E[observations[0]].T
        for t in range(1, len(observations)):
            alphas[t, :] = (alphas[t-1, :].reshape(1,-1) @ self.T.values) \
                * self.E[observations[t]].T
            
        return alphas

    def _betas(self, observations: list) -> np.ndarray:
        betas = np.zeros((len(observations), len(self.states)))
        betas[-1, :] = 1
        for t in range(len(observations) - 2, -1, -1):
            betas[t, :] = (self.T.values @ (self.E[observations[t+1]]
                                     * betas[t+1, :].reshape(-1, 1))).reshape(1, -1)
        return betas

    def uncover(self, observations: list) -> np.ndarray:
        alphas = self._alphas(observations)
        betas = self._betas(observations)
        argmaxs = (alphas * betas).argmax(axis=1)
        return list(map(lambda x: self.states[x], argmaxs))
