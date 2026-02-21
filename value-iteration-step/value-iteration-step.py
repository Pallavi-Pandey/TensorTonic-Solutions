def value_iteration_step(values, transitions, rewards, gamma):
    """
    Perform one step of value iteration and return updated values.
    """
    num_states = len(values)
    new_values = [0.0] * num_states

    for s in range(num_states):
        best = float('-inf')

        for a in range(len(transitions[s])):
            probs = transitions[s][a]
            r = rewards[s][a]

            total = 0.0
            for next_s, p in enumerate(probs):
                total += p * (r + gamma * values[next_s])

            best = max(best, total)

        new_values[s] = best

    return new_values