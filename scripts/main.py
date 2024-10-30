import datetime
from itertools import product

from matplotlib import pyplot as plt
import pandas as pd
from decoding import HiddenMarkovChain_Uncover
from likelihood import *
from hmc_simulation import HiddenMarkovChain_Simulation
from probability_matrix import ProbabilityMatrix
from probability_vector import ProbabilityVector
import os

from gamma import HiddenMarkovLayer
from model import HiddenMarkovModel


def test_probability_vector():
    a1 = ProbabilityVector({'rain': 0.7, 'sun': 0.3})
    a2 = ProbabilityVector({'sun': 0.1, 'rain': 0.9})
    print(a1.dict)
    print(a2.df)

    print("Comparison:", a1 == a2)
    print("Element-wise multiplication:", a1 * a2)
    print("Argmax:", a1.argmax())
    print("Getitem:", a1['rain'])


def test_probability_matrix():
    a1 = ProbabilityVector({'rain': 0.7, 'sun': 0.3})
    a2 = ProbabilityVector({'rain': 0.6, 'sun': 0.4})
    A = ProbabilityMatrix({'hot': a1, 'cold': a2})

    print(A)
    print(A.df)

    b1 = ProbabilityVector({'0S': 0.1, '1M': 0.4, '2L': 0.5})
    b2 = ProbabilityVector({'0S': 0.7, '1M': 0.2, '2L': 0.1})
    B = ProbabilityMatrix({'0H': b1, '1C': b2})

    print(B)
    print(B.df)

    P = ProbabilityMatrix.initialize(list('abcd'), list('xyz'))
    print('Dot product:', a1 @ A)
    print('Initialization:', P)
    print(P.df)


def test_hidden_markov_chain():
    a1 = ProbabilityVector({'1H': 0.7, '2C': 0.3})
    a2 = ProbabilityVector({'1H': 0.4, '2C': 0.6})

    b1 = ProbabilityVector({'1S': 0.1, '2M': 0.4, '3L': 0.5})
    b2 = ProbabilityVector({'1S': 0.7, '2M': 0.2, '3L': 0.1})

    A = ProbabilityMatrix({'1H': a1, '2C': a2})
    B = ProbabilityMatrix({'1H': b1, '2C': b2})
    pi = ProbabilityVector({'1H': 0.6, '2C': 0.4})

    hmc = HiddenMarkovChain(A, B, pi)
    observations = ['1S', '2M', '3L', '2M', '1S']

    print("Score for {} is {:f}.".format(
        observations, hmc.score(observations)))

    # all_possible_observations = {'1S', '2M', '3L'}
    # chain_length = 3  # any int > 0
    # all_observation_chains = list(product(*(all_possible_observations,) * chain_length))
    # all_possible_scores = list(map(lambda obs: hmc.score(obs), all_observation_chains))
    # print("All possible scores added: {}.".format(sum(all_possible_scores)))


def test_hidden_markov_chain_with_forward_algo():
    a1 = ProbabilityVector({'1H': 0.7, '2C': 0.3})
    a2 = ProbabilityVector({'1H': 0.4, '2C': 0.6})

    b1 = ProbabilityVector({'1S': 0.1, '2M': 0.4, '3L': 0.5})
    b2 = ProbabilityVector({'1S': 0.7, '2M': 0.2, '3L': 0.1})

    A = ProbabilityMatrix({'1H': a1, '2C': a2})
    B = ProbabilityMatrix({'1H': b1, '2C': b2})
    pi = ProbabilityVector({'1H': 0.6, '2C': 0.4})

    hmc_fp = HiddenMarkovChain_FP(A, B, pi)
    observations = ['1S', '2M', '3L', '2M', '1S']

    print("Score for {} is {:f}.".format(
        observations, hmc_fp.score(observations)))


def test_simulation():
    a1 = ProbabilityVector({'1H': 0.7, '2C': 0.3})
    a2 = ProbabilityVector({'1H': 0.4, '2C': 0.6})

    b1 = ProbabilityVector({'1S': 0.1, '2M': 0.2, '3L': 0.5, '4T': 0.2})
    b2 = ProbabilityVector({'1S': 0.5, '2M': 0.2, '3L': 0.1, '4T': 0.2})

    A = ProbabilityMatrix({'1H': a1, '2C': a2})
    B = ProbabilityMatrix({'1H': b1, '2C': b2})
    pi = ProbabilityVector({'1H': 0.6, '2C': 0.4})

    LENGTH_SIMULATION = 100
    hmc_s = HiddenMarkovChain_Simulation(A, B, pi)

    observation_hist, state_hist = hmc_s.run(LENGTH_SIMULATION)
    stats = pd.DataFrame({
        'observation': observation_hist,
        'hidden_state': state_hist
    }).map(lambda x: int(x[0])).plot()

    if not os.path.exists('../figs'):
        os.mkdir('../figs')

    if not os.path.exists('../figs/forwardings'):
        os.mkdir('../figs/forwardings')

    plt.savefig("../figs/forwardings/{}_({}_steps).png".format(str(
        datetime.datetime.now().strftime("%m-%d-%Y %H-%M-%S")), str(LENGTH_SIMULATION)))
    plt.close()

def test_convergence():
    a1 = ProbabilityVector({'1H': 0.7, '2C': 0.3})
    a2 = ProbabilityVector({'1H': 0.4, '2C': 0.6})

    b1 = ProbabilityVector({'1S': 0.1, '2M': 0.2, '3L': 0.5, '4T': 0.2})
    b2 = ProbabilityVector({'1S': 0.5, '2M': 0.2, '3L': 0.1, '4T': 0.2})

    A = ProbabilityMatrix({'1H': a1, '2C': a2})
    B = ProbabilityMatrix({'1H': b1, '2C': b2})
    pi = ProbabilityVector({'1H': 0.6, '2C': 0.4})

    LOG_LENGTH_START = 1
    LOG_LENGTH_END = 5
    hmc_s = HiddenMarkovChain_Simulation(A, B, pi)

    stats = {}

    for length in np.logspace(LOG_LENGTH_START, LOG_LENGTH_END, 40, base=10).astype(int):
        observation_hist, state_hist = hmc_s.run(length)
        stats[length] = pd.DataFrame(
            {'observation': observation_hist,
             'hidden_state': state_hist}).map(lambda x: int(x[0]))

    S = np.array(list(map(lambda x: x['hidden_state'].value_counts(
    ).to_numpy() / len(x), stats.values())))

    plt.semilogx(np.logspace(LOG_LENGTH_START,
                 LOG_LENGTH_END, 40).astype(int), S)
    plt.xlabel('Chain length T')
    plt.ylabel('Probability')
    plt.title('Converging probabilities.')
    plt.legend(['1H', '2C'])
    # plt.show()

    if not os.path.exists('../figs'):
        os.mkdir('../figs')

    if not os.path.exists('../figs/convergings'):
        os.mkdir('../figs/convergings')

    plt.savefig("../figs/convergings/{}_({}-{}_steps).png".format(str(
        datetime.datetime.now().strftime("%m-%d-%Y %H-%M-%S")), str(10**LOG_LENGTH_START), str(10**LOG_LENGTH_END)))
    
    plt.close()

def test_decoding():
    np.random.seed(42)

    a1 = ProbabilityVector({'1H': 0.7, '2C': 0.3})
    a2 = ProbabilityVector({'1H': 0.4, '2C': 0.6})
    b1 = ProbabilityVector({'1S': 0.1, '2M': 0.4, '3L': 0.5}) 
    b2 = ProbabilityVector({'1S': 0.7, '2M': 0.2, '3L': 0.1})
    A  = ProbabilityMatrix({'1H': a1, '2C': a2})
    B  = ProbabilityMatrix({'1H': b1, '2C': b2})
    pi = ProbabilityVector({'1H': 0.6, '2C': 0.4})

    hmc = HiddenMarkovChain_Uncover(A, B, pi)

    observed_sequence, latent_sequence = hmc.run(5)
    optimal_state_sequence = hmc.uncover(observed_sequence)

    print("Optimal path: {}".format(optimal_state_sequence))

    all_possible_states = {'1H','2C'}
    chain_length = 6
    all_possible_states = list(product(*(all_possible_states,) * chain_length))

    df = pd.DataFrame(all_possible_states)
    dfp = pd.DataFrame()

    for i in range(chain_length):
        dfp['p' + str(i)] = df.apply(lambda x: 
            hmc.E.df.loc[x[i], observed_sequence[i]], axis=1)

    scores = dfp.sum(axis=1).sort_values(ascending=False)
    df = df.iloc[scores.index]
    df['score'] = scores
    print(df.head(10).reset_index())

def test_model():
    np.random.seed(42)

    observations = ['3L', '2M', '1S', '3L', '3L', '3L']

    states = ['1H', '2C']
    observables = ['1S', '2M', '3L']

    hml = HiddenMarkovLayer.initialize(states, observables)
    hmm = HiddenMarkovModel(hml)

    hmm.train(observations, 25)

def main():
    # test_probability_vector()
    # test_probability_matrix()
    # test_hidden_markov_chain()
    # test_hidden_markov_chain_with_forward_algo()
    # test_simulation()
    # test_convergence()
    # test_decoding()
    test_model()


if __name__ == "__main__":
    main()
