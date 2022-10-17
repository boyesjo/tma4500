import numpy as np

np.random.seed(1337)


def generate_data(n: int = 32):

    x = np.zeros((2 * n, 4, 2))
    y = np.zeros(2 * n, dtype=int)

    for i in range(n):
        x[i, i % 4] = 1
        y[i] = 1

    for i in range(n):
        x[n + i][:, i % 2] = 1
        y[n + i] = -1

    # flatten images
    x = x.reshape(2 * n, 8)
    # rescale to (0, pi/2)
    x *= np.pi / 2

    # add gaussian noise
    x += np.random.normal(0, 0.1, x.shape)

    # shuffle data
    idx = np.arange(2 * n)
    np.random.shuffle(idx)
    x = x[idx]
    y = y[idx]

    return x, y


# x_train, y_train = generate_data(64)
# x_test, y_test = generate_data(64)
