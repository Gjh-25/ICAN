import numpy as np
import tensorflow as tf


def compute_theta_c(adj_matrix):
    degrees = np.sum(adj_matrix, axis=1)
    avg_k = np.mean(degrees)
    avg_k2 = np.mean(degrees ** 2)
    theta_c = avg_k / (avg_k2 - avg_k)
    return theta_c


def simulate_one_run(adj_matrix, initial_node, theta, beta):
    n_nodes = adj_matrix.shape[0]
    status = np.zeros(n_nodes, dtype=int)
    status[initial_node] = 1

    max_steps = 1000
    for _ in range(max_steps):
        current_infected = np.where(status == 1)[0]
        if len(current_infected) == 0:
            break

        new_infections = []
        for node in current_infected:
            neighbors = np.where(adj_matrix[node] == 1)[0]
            for neighbor in neighbors:
                if status[neighbor] == 0 and np.random.rand() < theta:
                    new_infections.append(neighbor)

        new_infections = list(set(new_infections))
        for node in new_infections:
            if status[node] == 0:
                status[node] = 1

        recover = []
        for node in current_infected:
            if np.random.rand() < beta:
                status[node] = 2
                recover.append(node)

        current_infected = np.where(status == 1)[0]

    return np.sum(status != 0)


def sir_calculate_node_scores(adj_matrix, num_simulations=100):

    n_nodes = adj_matrix.shape[0]
    scores = np.zeros(n_nodes)
    theta_c = compute_theta_c(adj_matrix)
    theta = 1.5* theta_c
    beta=1

    for v in range(n_nodes):
        total = 0
        for _ in range(num_simulations):
            total += simulate_one_run(adj_matrix, v, theta, beta)
        scores[v] = total / num_simulations

    return scores
