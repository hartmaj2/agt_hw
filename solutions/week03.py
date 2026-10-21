#!/usr/bin/env python3

import numpy as np
import week01 as w1
import matplotlib.pyplot as plt


def compute_deltas(
    row_matrix: np.ndarray,
    col_matrix: np.ndarray,
    row_strategy: np.ndarray,
    col_strategy: np.ndarray,
) -> np.ndarray:
    """Compute players' incentives to deviate from their strategies.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    col_matrix : np.ndarray
        The column player's payoff matrix
    row_strategy : np.ndarray
        The row player's strategy
    col_strategy : np.ndarray
        The column player's strategy

    Returns
    -------
    np.ndarray
        Each player's incentive to deviate
    """


    row_br = w1.calculate_best_response_against_col(row_matrix,col_strategy)
    col_br = w1.calculate_best_response_against_row(col_matrix,row_strategy)

    row_utility, col_utility = w1.evaluate_general_sum(row_matrix,col_matrix,row_strategy,col_strategy)
    row_br_utility, _ = w1.evaluate_general_sum(row_matrix,col_matrix,row_br,col_strategy)

    _, col_br_utility = w1.evaluate_general_sum(row_matrix,col_matrix,row_strategy,col_br)


    # we use the fact, that if there is to be a best response \pi of one player to a strategy, then
    # there exists pure stategy a which yields same value as \pi

    return np.array([row_br_utility-row_utility,col_br_utility-col_utility])


def compute_nash_conv(
    row_matrix: np.ndarray,
    col_matrix: np.ndarray,
    row_strategy: np.ndarray,
    col_strategy: np.ndarray,
) -> np.float64:
    """Compute the NashConv value of a given strategy profile.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    col_matrix : np.ndarray
        The column player's payoff matrix
    row_strategy : np.ndarray
        The row player's strategy
    col_strategy : np.ndarray
        The column player's strategy

    Returns
    -------
    np.float64
        The NashConv value of the given strategy profile
    """

    return np.sum(compute_deltas(row_matrix,col_matrix,row_strategy,col_strategy))


def compute_exploitability(
    row_matrix: np.ndarray,
    col_matrix: np.ndarray,
    row_strategy: np.ndarray,
    col_strategy: np.ndarray,
) -> np.float64:
    """Compute the exploitability of a given strategy profile.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    col_matrix : np.ndarray
        The column player's payoff matrix
    row_strategy : np.ndarray
        The row player's strategy
    col_strategy : np.ndarray
        The column player's strategy

    Returns
    -------
    np.float64
        The exploitability value of the given strategy profile
    """

    return compute_nash_conv(row_matrix,col_matrix,row_strategy,col_strategy) / 2


def fictitious_play(
    row_matrix: np.ndarray, col_matrix: np.ndarray, num_iters: int, naive: bool
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Run Fictitious Play for a given number of iterations.

    Although any averaging method is valid, the reference solution updates the
    average strategy vectors using a moving average. Therefore, it is recommended
    to use the same averaging method to avoid numerical discrepancies during testing.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    col_matrix : np.ndarray
        The column player's payoff matrix
    num_iters : int
        The number of iterations to run the algorithm for
    naive : bool
        Whether to calculate the best response against the last
        opponent's strategy or the average opponent's strategy

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        The sequence of average strategy profiles produced by the algorithm
    """

    def compute_next_strategies(row_matrix,col_matrix,avg_row_strat,avg_col_strat):
        row_next = w1.calculate_best_response_against_col(row_matrix,avg_col_strat)
        col_next = w1.calculate_best_response_against_row(col_matrix,avg_row_strat)
        return row_next,col_next

    row_init_str = np.ones(row_matrix.shape[0]) / row_matrix.shape[0]
    col_init_str = np.ones(row_matrix.shape[1]) / row_matrix.shape[1]

    cur_row_str = row_init_str
    cur_col_str = col_init_str

    avg_row_str = row_init_str
    avg_col_str = col_init_str

    avg_strategies = []

    for i in range(num_iters):
        if naive:
            cur_row_str, cur_col_str = compute_next_strategies(row_matrix,col_matrix,cur_row_str,cur_col_str)
        else:
            cur_row_str, cur_col_str = compute_next_strategies(row_matrix,col_matrix,avg_row_str,avg_col_str)
        avg_row_str += (cur_row_str - avg_row_str) / (i + 1)
        avg_col_str += (cur_col_str - avg_col_str) / (i + 1)
        avg_strategies.append((avg_row_str.copy(),avg_col_str.copy()))

    return avg_strategies


def plot_exploitability(
    row_matrix: np.ndarray,
    col_matrix: np.ndarray,
    strategies: list[tuple[np.ndarray, np.ndarray]],
    label: str,
) -> list[np.float64]:
    """Compute and plot the exploitability of a sequence of strategy profiles.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    col_matrix : np.ndarray
        The column player's payoff matrix
    strategies : list[tuple[np.ndarray, np.ndarray]]
        The sequence of strategy profiles
    label : str
        The name of the algorithm that produced `strategies`

    Returns
    -------
    list[np.float64]
        A sequence of exploitability values, one for each strategy profile
    """

    exploitabilities = [compute_exploitability(row_matrix,col_matrix,row,col) for row,col in strategies]

    plt.plot(range(len(strategies)),exploitabilities)
    plt.show()

    return exploitabilities


def main() -> None:
    pass


if __name__ == '__main__':
    main()
