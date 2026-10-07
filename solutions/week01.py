#!/usr/bin/env python3

import numpy as np


def evaluate_general_sum(
    row_matrix: np.ndarray,
    col_matrix: np.ndarray,
    row_strategy: np.ndarray,
    col_strategy: np.ndarray,
) -> np.ndarray:
    """Compute the expected utility of each player in a general-sum game.

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
        A vector of expected utilities of the players
    """
    exputil_row = row_strategy @ row_matrix @ col_strategy
    exputil_col = row_strategy @ col_matrix @ col_strategy
    return np.array([exputil_row,exputil_col])


def evaluate_zero_sum(
    row_matrix: np.ndarray, row_strategy: np.ndarray, col_strategy: np.ndarray
) -> np.ndarray:
    """Compute the expected utility of each player in a zero-sum game.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    row_strategy : np.ndarray
        The row player's strategy
    col_strategy : np.ndarray
        The column player's strategy

    Returns
    -------
    np.ndarray
        A vector of expected utilities of the players
    """

    exputil_row = row_strategy @ row_matrix @ col_strategy
    return np.array([exputil_row,-exputil_row])


def calculate_best_response_against_row(
    col_matrix: np.ndarray, row_strategy: np.ndarray
) -> np.ndarray:
    """Compute a pure best response for the column player against the row player.

    Parameters
    ----------
    col_matrix : np.ndarray
        The column player's payoff matrix
    row_strategy : np.ndarray
        The row player's strategy

    Returns
    -------
    np.ndarray
        The column player's best response
    """

    exp_util_col = row_strategy @ col_matrix
    best_action_index = np.argmax(exp_util_col)
    best_strategy = np.zeros(col_matrix.shape[1])
    best_strategy[best_action_index] = 1
    return best_strategy


def calculate_best_response_against_col(
    row_matrix: np.ndarray, col_strategy: np.ndarray
) -> np.ndarray:
    """Compute a pure best response for the row player against the column player.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    col_strategy : np.ndarray
        The column player's strategy

    Returns
    -------
    np.ndarray
        The row player's best response
    """

    exp_util_col = row_matrix @ col_strategy
    best_action_index = np.argmax(exp_util_col)
    best_strategy = np.zeros(row_matrix.shape[0])
    best_strategy[best_action_index] = 1
    return best_strategy


def evaluate_row_against_best_response(
    row_matrix: np.ndarray, col_matrix: np.ndarray, row_strategy: np.ndarray
) -> np.float64:
    """Compute the utility of the row player when playing against a best response strategy.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    col_matrix : np.ndarray
        The column player's payoff matrix
    row_strategy : np.ndarray
        The row player's strategy

    Returns
    -------
    np.float64
        The expected utility of the row player
    """

    col_best_response = calculate_best_response_against_row(col_matrix,row_strategy)
    row_utility = evaluate_general_sum(row_matrix,col_matrix,row_strategy,col_best_response)[0]
    return row_utility


def evaluate_col_against_best_response(
    row_matrix: np.ndarray, col_matrix: np.ndarray, col_strategy: np.ndarray
) -> np.float64:
    """Compute the utility of the column player when playing against a best response strategy.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    col_matrix : np.ndarray
        The column player's payoff matrix
    col_strategy : np.ndarray
        The column player's strategy

    Returns
    -------
    np.float64
        The expected utility of the column player
    """

    row_best_response = calculate_best_response_against_col(row_matrix,col_strategy)
    col_utility = evaluate_general_sum(row_matrix,col_matrix,row_best_response,col_strategy)[1]
    return col_utility


def find_strictly_dominated_actions(matrix: np.ndarray) -> np.ndarray:
    """Find strictly dominated actions for the given normal-form game.

    Parameters
    ----------
    matrix : np.ndarray
        A payoff matrix of one of the players

    Returns
    -------
    np.ndarray
        Indices of strictly dominated actions
    """

    indices = []
    for i in range(matrix.shape[0]):
        util_vect_i = matrix[i,:]
        for j in range(matrix.shape[0]):
            util_vect_j = matrix[j,:]
            if np.all(util_vect_j > util_vect_i):
                indices.append(i)
                break
    return np.array(indices,dtype=np.int64)

def find_strictly_dominated_actions_ignore(matrix : np.ndarray, rows_ignore : np.ndarray, cols_ignore : np.ndarray) -> np.ndarray:
    indices = []
    cols_keep = np.setdiff1d(np.arange(matrix.shape[1]), cols_ignore)
    rows_keep = np.setdiff1d(np.arange(matrix.shape[0]), rows_ignore)
    for i in rows_keep:
        util_vect_i = matrix[i, cols_keep]
        for j in rows_keep:
            util_vect_j = matrix[j, cols_keep]
            if np.all(util_vect_j > util_vect_i):
                indices.append(i)
                break
    return np.array(indices,dtype=np.int64)

def iterated_removal_of_dominated_strategies(
    row_matrix: np.ndarray, col_matrix: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run Iterated Removal of Dominated Strategies.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    col_matrix : np.ndarray
        The column player's payoff matrix

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Four-tuple of reduced row and column payoff matrices, and remaining row and column actions
    """

    rows_ignore = np.array([],dtype=np.int64)
    cols_ignore = np.array([],dtype=np.int64)

    while True:
        row_indices = find_strictly_dominated_actions_ignore(row_matrix,rows_ignore,cols_ignore)
        col_indices = find_strictly_dominated_actions_ignore(col_matrix.T,cols_ignore,rows_ignore) # parameters changed places because of transpose

        if row_indices.size == 0 and col_indices.size == 0:
            break

        rows_ignore = np.union1d(rows_ignore,row_indices)
        cols_ignore = np.union1d(cols_ignore,col_indices)

    rows_keep = np.setdiff1d(np.arange(row_matrix.shape[0]), rows_ignore)
    cols_keep = np.setdiff1d(np.arange(col_matrix.shape[1]), cols_ignore)

    reduced_row_matrix = row_matrix[np.ix_(rows_keep, cols_keep)]
    reduced_col_matrix = col_matrix[np.ix_(rows_keep, cols_keep)]

    return reduced_row_matrix,reduced_col_matrix,rows_keep,cols_keep


def main() -> None:
    pass

if __name__ == '__main__':
    main()
