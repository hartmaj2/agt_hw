#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as so
import itertools as it


def plot_best_response_value_function(row_matrix: np.ndarray, step_size: float) -> None:
    """Plot the best response value function for the row player in a 2xN zero-sum game.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    step_size : float
        The step size for the probability of the first action of the row player
    """

    def create_plot(probs : list[float], br_vals : list[float]) -> None:
        plt.plot(probs,br_vals)
        plt.show()


    probs = np.arange(0,1+step_size,step_size,dtype=float)
    br_vals = []

    for prob in probs:
        prob_vect = np.array([prob,1-prob])
        vals = prob_vect @ row_matrix
        br_vals.append(vals.min()) # the column player chooses the response which MINIMIZES row players utility
        
    create_plot(list(probs),br_vals)


def verify_support(
    matrix: np.ndarray, row_support: np.ndarray, col_support: np.ndarray
) -> np.ndarray | None:
    """Construct a system of linear equations to check whether there
    exists a candidate for a Nash equilibrium for the given supports.

    This function is not supposed to check if actions outside of the
    given support lead to a higher payoff than the found strategy.

    The reference implementation uses `scipy.optimize.linprog`
    with the default solver -- 'highs'. You can find more information at
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html

    Parameters
    ----------
    matrix : np.ndarray
        A payoff matrix of one of the players (default is row player)
    row_support : np.ndarray
        The row player's support
    col_support : np.ndarray
        The column player's support

    Returns
    -------
    np.ndarray | None
        The opponent's strategy, if it exists, otherwise `None`
    """

    # OBS: support enumeration LP consists of two independent sub LP's
    # one which computes probs of first player such that each pure action of other player leads to same value (for the second player)
    # second is analogous 

    # So this part of the program will create variables for the actions in the column players support (so we will find column players mixed strategy)
    # we need that for each of the row players actions, the value of the convex combination of the parts of the matrix corresponding to the column players cols and this specific row is some variable v

    # Vars
    # 1. p_1,....,p_{col_support size}   (vars for the probabilities of column players actions)
    # 2. v_row (var for the value that row player gets for every pure action (the value being determined by the the weighted probs of the column player weighted by the vals at the matrix position given by the row of pure action and col of this variable)

    # Linear equations (and inequalities)
    # 1. \sum {p_i} = 1
    # 2. p_i > 0 for all i
    # 3. for every pure action i of row player SUPPORT: p_1 * m_i,n_1 + p_2 * m_i,n_2 + ... + p_{col_support_size} * m_i,n_{col_support size} = v_row
    # (n_j is index of the j_th action in the support)

    support_submatrix = matrix[np.ix_(row_support, col_support)] # cut out just the necessary rows and columns

    # add column of (-1,...,-1)^T as a last column of the matrix
    support_submatrix = np.hstack((
        support_submatrix,
        -np.ones((support_submatrix.shape[0], 1))
    ))

    distrib_vector = np.append(np.ones(len(col_support)), 0) # 1. \sum {p_i} = 1


    A = np.vstack((support_submatrix,distrib_vector))
    b = np.append(np.zeros(len(row_support)), 1) # create the right hand sides (for the last equality \sum {p_i} = 1 we need to add the 1)

    result = so.linprog(
        c=np.zeros(A.shape[1]), # we don't care what the cost function is (we can just have all zeroes)
        A_eq=A,
        b_eq=b,
        bounds=[(0, None)] * len(col_support) + [(None, None)], # the bounds are just 0 from below for each p_i and no bound for v_row
        method="highs"
    )

    if result.success:
        x = result.x
        return x[:-1]
    else:
        # print(result.message)
        return None

def support_enumeration(
    row_matrix: np.ndarray, col_matrix: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Run the Support Enumeration algorithm and return a list of all Nash equilibria

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    col_matrix : np.ndarray
        The column player's payoff matrix

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        A list of strategy profiles corresponding to found Nash equilibria
    """

    # checks whether these supports correspond to some nash equilibrium
    def verify_supports(row_matrix: np.ndarray, col_matrix: np.ndarray, row_support : np.ndarray, col_support : np.ndarray) -> tuple[np.ndarray,np.ndarray] | None:
        col_supp_distrib = verify_support(row_matrix,row_support,col_support)
        if col_supp_distrib is None:
            return None
        row_supp_distrib = verify_support(col_matrix.T,col_support,row_support)
        if row_supp_distrib is None:
            return None

        m = row_matrix.shape[0]
        n = row_matrix.shape[1]
        row_distrib = create_strategy_from_support(row_supp_distrib,row_support,m)
        col_distrib = create_strategy_from_support(col_supp_distrib,col_support,n)

        if not is_br_against(row_matrix,row_distrib,col_distrib):
            return None
        if not is_br_against(col_matrix.T,col_distrib,row_distrib):
            return None

        return row_distrib,col_distrib

    # checks if a given candidate_strategy is a best response against opponent_strategy, the matrix has values corresponding to utility of the guy with candidate strategy
    def is_br_against(candidate_matrix: np.ndarray, candidate_strategy: np.ndarray, opponent_strategy: np.ndarray) -> bool:
        candidate_action_utils = candidate_matrix @ opponent_strategy
        candidate_utility = candidate_strategy @ candidate_action_utils
        best_utility = candidate_action_utils.max()

        is_br = candidate_utility >= best_utility - 1e-9 # we need this little constant to avoid floating point arithmetic errors

        return bool(is_br)
        
    def create_strategy_from_support(support_distribution, support_indices, original_length):
        res = np.zeros(original_length)
        res[support_indices] = support_distribution
        return res

    equilibria = []

    m = row_matrix.shape[0]
    n = row_matrix.shape[1]
    for k in range(1,m+1):
        for s_1 in it.combinations(range(m),k):
            for l in range(1,n+1):
                for s_2 in it.combinations(range(n),l):
                    tup = verify_supports(row_matrix,col_matrix,np.array(s_1),np.array(s_2))
                    if tup is None:
                        continue
                    equilibria.append(tup)

    return equilibria

def main() -> None:
    pass


if __name__ == '__main__':
    main()
