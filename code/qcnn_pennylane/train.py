# %%
import intermediate
import pandas as pd
import simple
from gen_data import generate_data
from pennylane import numpy as np
from pennylane.optimize import AdamOptimizer

x_train, y_train = generate_data(64)
x_test, y_test = generate_data(64)


# %%
def train(x_train, y_train, circuit, param_count, max_iter=200):
    def square_loss(labels, predictions):
        loss = 0
        for label, prediction in zip(labels, predictions):
            loss = loss + (label - prediction) ** 2

        loss = loss / len(labels)
        return loss

    def cost(var, features, labels):
        preds = [circuit(var, x) for x in features]
        return square_loss(labels, preds)

    def accuracy(var, features, labels):
        preds = [np.sign(circuit(var, x)) for x in features]
        return np.mean(preds == labels)

    opt = AdamOptimizer(0.01)

    params = np.random.randn(param_count)

    history = []

    for i in range(max_iter):
        (params, _, _), loss = opt.step_and_cost(
            cost, params, x_train, y_train
        )
        test_loss = cost(params, x_test, y_test)
        history.append(
            {
                "Iteration": i,
                "Loss": float(loss),
                "Test loss": float(test_loss),
                "Accuracy": float(accuracy(params, x_train, y_train)),
                "Test accuracy": float(accuracy(params, x_test, y_test)),
            }
        )
        print(history[-1])

    return history


# %%
for circuit, param_count, name in [
    (simple.circuit, simple.TOTAL_PARAMS, "simple"),
    (intermediate.circuit, intermediate.TOTAL_PARAMS, "intermediate"),
]:
    history = train(x_train, y_train, circuit, param_count)
    pd.DataFrame(history).to_csv(f"results_{name}.csv", index=False)

# %%
