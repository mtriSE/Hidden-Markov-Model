
import numpy as np
from decoding import HiddenMarkovChain_Uncover


class HiddenMarkovLayer(HiddenMarkovChain_Uncover):
    def _digammas(self, observations: list) -> np.ndarray:
        T, N = len(observations), len(self.states)
        digammas = np.zeros((T-1, N, N))
        
        alphas = self._alphas(observations)
        betas = self._betas(observations)
        score = self.score(observations)
        for t in range(T - 1):
            # P1 will have shape NxN
            P1 = (alphas[t, :].reshape(-1,1) * self.T.values)
            # P2 will have shape NxN = ((1xN).T * 1xN) = (Nx1 * 1xN)
            P2 = self.E[observations[t+1]].T * betas[t+1].reshape(1,-1)
            digammas[t, :, :] = P1 * P2 / score
        return digammas
