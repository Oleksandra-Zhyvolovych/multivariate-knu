from matplotlib import pyplot as plt
import numpy as np
import enum

gb = (1.25665, 0.0133085)


def ObjectiveFunction(x: np.array) -> float:
    return (
        abs(-4 * x[0] + 2 * x[1] + 5)
        + abs(2 * x[0] + 3 * x[1] - 2)
        + 0.1 * (x[0] - 3 * x[1]) ** 6
    )


def SubGradient(x: np.array) -> np.array:
    res = np.array([6 * (x[0] - 3 * x[1]) ** 5, -18 * (x[0] - 3 * x[1]) ** 5])
    if 4 * x[0] < 2 * x[1] + 5:
        res[0] += -4
        res[1] += 2
    elif 4 * x[0] == 2 * x[1] + 5:
        res[0] += np.random.uniform(-4, 4, 1)
        res[1] += np.random.uniform(-2, 2, 1)
    else:
        res[0] += 4
        res[1] += -2

    if 2 * x[0] + 3 * x[1] > 2:
        res[0] += 2
        res[1] += 3
    elif 2 * x[0] + 3 * x[1] == 2:
        res[0] += np.random.uniform(-2, 2, 1)
        res[1] += np.random.uniform(-3, 3, 1)
    else:
        res[0] += -2
        res[1] += -3

    return res


def plotObjectiveFunction():
    x = np.arange(0.5, 1.5, 0.1)
    y = np.arange(-0.5, 0.5, 0.1)
    X, Y = np.meshgrid(x, y)
    Z = ObjectiveFunction(np.array([X, Y]))
    ax = plt.axes(projection="3d")
    ax.plot_surface(X, Y, Z, cmap="plasma", edgecolor="none")
    plt.savefig("objective_function.png")
    plt.clf()


def plotSubGradient():
    x = np.arange(-5, 5, 0.1)
    y = np.arange(-5, 5, 0.1)
    X, Y = np.meshgrid(x, y)
    Z = [
        SubGradient(np.array([X[i, j], Y[i, j]]))
        for i in range(X.shape[0])
        for j in range(X.shape[1])
    ]
    a = np.array([x for (x, _) in Z])
    b = np.array([y for (_, y) in Z])
    # normalize the vectors
    mod = np.sqrt(a ** 2 + b ** 2)
    a /= mod
    b /= mod

    plt.quiver(X, Y, -a, -b)
    plt.plot(x, (4 * x - 5) / 2, "r", label="$f_1(x) = \|-4x + 2y + 5\|$")
    plt.plot(x, (2 - 2 * x) / 3, "b", label="$f_2(x) = \|2x + 3y - 2\|$")
    plt.plot(*gb, "go", label="$\mathbf{\min}_{global}$")
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.legend()
    plt.savefig("subgradient.png")
    plt.clf()


class Strategy(enum.Enum):
    consant_size = 1
    constant_length = 2
    square_summable = 3
    diminishing = 4


def optimize(
    f,
    g,
    initial_point,
    strategy=Strategy.consant_size,
    h=0.1,
    a=1.0,
    b=1.0,
    max_iterations=10000,
    epsilon=1e-6,
):
    points = []
    vals = []
    best_point = initial_point
    best_value = f(best_point)
    points.append(best_point)
    vals.append(best_value)

    step_size = 0.0
    for iter in range(1, max_iterations):
        if strategy == Strategy.consant_size:
            step_size = h
        elif strategy == Strategy.constant_length:
            step_size = h / np.linalg.norm(g(points[-1]))
        elif strategy == Strategy.square_summable:
            step_size = a / (iter + b)
        elif strategy == Strategy.diminishing:
            step_size = a / np.sqrt(iter)

        new_point = points[-1] - step_size * g(points[-1])
        new_value = f(new_point)
        points.append(new_point)
        vals.append(new_value)

        if new_value < best_value:
            best_point = new_point
            best_value = new_value
        if (
            np.linalg.norm(points[-1] - points[-2]) < epsilon
            and abs(vals[-1] - vals[-2]) < epsilon
        ):
            break

    return points, vals, best_point, best_value


if __name__ == "__main__":
    plotObjectiveFunction()
    plotSubGradient()

    global_min = ObjectiveFunction(gb)
    plt.xlabel("Ітерація")
    plt.ylabel("$|f(x) - f(x^{\\ast})|$")
    # region const_size
    const_size = {}
    h = 0.001
    const_size[0.001] = optimize(
        ObjectiveFunction,
        SubGradient,
        (0, 0),
        strategy=Strategy.consant_size,
        h=h,
        max_iterations=10 ** 4,
    )

    h = 0.0001
    const_size[0.0001] = optimize(
        ObjectiveFunction,
        SubGradient,
        (0, 0),
        strategy=Strategy.consant_size,
        h=h,
        max_iterations=10 ** 4,
    )
    h = 0.00001
    const_size[0.00001] = optimize(
        ObjectiveFunction,
        SubGradient,
        (0, 0),
        strategy=Strategy.consant_size,
        h=h,
        max_iterations=10 ** 4,
    )

    plt.plot(
        range(len(const_size[0.001][1])),
        list(map(lambda x: abs(global_min - x), const_size[0.001][1])),
        label="$h = 0.001$",
    )
    plt.plot(
        range(len(const_size[0.0001][1])),
        list(map(lambda x: abs(global_min - x), const_size[0.0001][1])),
        label="$h = 0.0001$",
    )
    plt.plot(
        range(len(const_size[0.00001][1])),
        list(map(lambda x: abs(global_min - x), const_size[0.00001][1])),
        label="$h = 0.00001$",
    )
    plt.legend()
    plt.savefig("const_size.png")
    plt.clf()
    const_size.clear()
    # endregion
    plt.xlabel("Ітерація")
    plt.ylabel("$|f(x) - f(x^{\\ast})|$")
    # region const_len
    const_len = {}
    h = 0.001
    const_len[0.001] = optimize(
        ObjectiveFunction,
        SubGradient,
        (0, 0),
        strategy=Strategy.constant_length,
        h=h,
        max_iterations=10 ** 4,
    )

    h = 0.0001
    const_len[0.0001] = optimize(
        ObjectiveFunction,
        SubGradient,
        (0, 0),
        strategy=Strategy.constant_length,
        h=h,
        max_iterations=10 ** 4,
    )
    h = 0.00001
    const_len[0.00001] = optimize(
        ObjectiveFunction,
        SubGradient,
        (0, 0),
        strategy=Strategy.constant_length,
        h=h,
        max_iterations=10 ** 4,
    )

    plt.plot(
        range(len(const_len[0.001][1])),
        list(map(lambda x: abs(global_min - x), const_len[0.001][1])),
        label="$h = 0.001$",
    )
    plt.plot(
        range(len(const_len[0.0001][1])),
        list(map(lambda x: abs(global_min - x), const_len[0.0001][1])),
        label="$h = 0.0001$",
    )
    plt.plot(
        range(len(const_len[0.00001][1])),
        list(map(lambda x: abs(global_min - x), const_len[0.00001][1])),
        label="$h = 0.00001$",
    )
    plt.legend()
    plt.savefig("const_len.png")
    plt.clf()
    const_len.clear()
    # endregion
    plt.xlabel("Ітерація")
    plt.ylabel("$|f(x) - f(x^{\\ast})|$")
    # region square_summable
    square_summable = {}
    a = 0.1
    b = 1.0
    square_summable[(0.1, 1.0)] = optimize(
        ObjectiveFunction,
        SubGradient,
        (0, 0),
        strategy=Strategy.square_summable,
        a=a,
        b=b,
        max_iterations=10 ** 4,
    )

    a = 0.01
    b = 1
    square_summable[(0.01, 1)] = optimize(
        ObjectiveFunction,
        SubGradient,
        (0, 0),
        strategy=Strategy.square_summable,
        a=a,
        b=b,
        max_iterations=10 ** 4,
    )

    a = 0.01
    b = 10
    square_summable[(0.01, 10)] = optimize(
        ObjectiveFunction,
        SubGradient,
        (0, 0),
        strategy=Strategy.square_summable,
        a=a,
        b=b,
        max_iterations=10 ** 4,
    )

    plt.plot(
        range(len(square_summable[(0.1, 1.0)][1])),
        list(map(lambda x: abs(global_min - x), square_summable[(0.1, 1.0)][1])),
        label="$a = 0.1, b = 1.0$",
    )

    plt.plot(
        range(len(square_summable[(0.01, 1)][1])),
        list(map(lambda x: abs(global_min - x), square_summable[(0.01, 1)][1])),
        label="$a = 0.01, b = 1$",
    )

    plt.plot(
        range(len(square_summable[(0.01, 10)][1])),
        list(map(lambda x: abs(global_min - x), square_summable[(0.01, 10)][1])),
        label="$a = 0.01, b = 10$",
    )

    plt.legend()
    plt.savefig("square_summable.png")
    plt.clf()
    square_summable.clear()
    # endregion
    plt.xlabel("Ітерація")
    plt.ylabel("$|f(x) - f(x^{\\ast})|$")
    # region diminishing
    diminishing = {}
    a = 0.01

    diminishing[0.1] = optimize(
        ObjectiveFunction,
        SubGradient,
        (0, 0),
        strategy=Strategy.diminishing,
        a=a,
        max_iterations=10 ** 4,
    )

    a = 0.001
    diminishing[0.01] = optimize(
        ObjectiveFunction,
        SubGradient,
        (0, 0),
        strategy=Strategy.diminishing,
        a=a,
        max_iterations=10 ** 4,
    )

    a = 0.0001
    diminishing[0.001] = optimize(
        ObjectiveFunction,
        SubGradient,
        (0, 0),
        strategy=Strategy.diminishing,
        a=a,
        max_iterations=10 ** 4,
    )

    plt.plot(
        range(len(diminishing[0.1][1])),
        list(map(lambda x: abs(global_min - x), diminishing[0.1][1])),
        label="$a = 0.1$",
    )

    plt.plot(
        range(len(diminishing[0.01][1])),
        list(map(lambda x: abs(global_min - x), diminishing[0.01][1])),
        label="$a = 0.01$",
    )

    plt.plot(
        range(len(diminishing[0.001][1])),
        list(map(lambda x: abs(global_min - x), diminishing[0.001][1])),
        label="$a = 0.001$",
    )

    plt.legend()
    plt.savefig("diminishing.png")
    plt.clf()
    diminishing.clear()
    # endregion
