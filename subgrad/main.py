import enum
import os

import matplotlib.pyplot as plt
import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go
import math

gb = (1.25665, 0.0133085)


def ObjectiveFunction(x: np.array) -> float:
    return (
        abs(-4 * x[0] + 2 * x[1] + 5)
        + abs(2 * x[0] + 3 * x[1] - 2)
        + 0.1 * (x[0] - 3 * x[1]) ** 6
    )


def SubGradient(x: np.array) -> np.array:
    res = np.array(
        [0.1 * 6 * (x[0] - 3 * x[1]) ** 5, -3 * 0.1 * 6 * (x[0] - 3 * x[1]) ** 5],
        dtype=np.float64,
    )

    if math.isclose(4 * x[0], 2 * x[1] + 5, abs_tol=1e-6):
        res[0] += np.random.uniform(-4, 4, 1)
        res[1] += np.random.uniform(-2, 2, 1)
    elif 4 * x[0] < 2 * x[1] + 5:
        res[0] += -4
        res[1] += 2
    else:
        res[0] += 4
        res[1] += -2

    if math.isclose(2 * x[0] + 3 * x[1], 2, abs_tol=1e-6):
        res[0] += np.random.uniform(-2, 2, 1)
        res[1] += np.random.uniform(-3, 3, 1)
    elif 2 * x[0] + 3 * x[1] > 2:
        res[0] += 2
        res[1] += 3
    else:
        res[0] += -2
        res[1] += -3

    return res


def SubDifferential(x: np.array) -> np.array:
    g_1 = np.array([0.6 * (x[0] - 3 * x[1]) ** 5, -1.8 * (x[0] - 3 * x[1]) ** 5])
    g_2 = None
    g_3 = None

    if math.isclose(4 * x[0], 2 * x[1] + 5, abs_tol=1e-5):
        g_2 = np.array([(-4, 2), (4, -2)])
    elif 2 * x[1] + 5 > -4 * x[0]:
        g_2 = np.array([-4, 2])
    else:
        g_2 = np.array([4, -2])

    if math.isclose(2 * x[0] + 3 * x[1], 2, abs_tol=1e-5):
        g_3 = np.array([(2, 3), (-2, -3)])
    elif 2 * x[0] + 3 * x[1] > 2:
        g_3 = np.array([2, 3])
    else:
        g_3 = np.array([-2, -3])

    return g_1 + g_2 + g_3


def createNonSmoothQuiverVectors(points):
    x1, x2 = points

    x, y = [], []
    u, v = [], []

    for i in range(x1.shape[0]):
        subd = SubDifferential((x1[i], x2[i]))
        if subd.shape == (2, 2):
            u_in_dot = np.linspace(*subd[0], 2)
            v_in_dot = np.linspace(*subd[1], 2)
            u_in_dot, v_in_dot = np.meshgrid(u_in_dot, v_in_dot)
            u_in_dot = u_in_dot.ravel()
            v_in_dot = v_in_dot.ravel()

            u.append(u_in_dot)
            v.append(v_in_dot)

            x = x + len(u_in_dot) * [x1[i]]
            y = y + len(v_in_dot) * [x2[i]]

        else:
            u.append(subd[0])
            v.append(subd[1])
            x = x + [x1[i]]
            y = y + [x2[i]]

    x = np.array(x)
    y = np.array(y)
    u = np.array(u).ravel()
    v = np.array(v).ravel()

    return x, y, u, v


def plotObjective():
    x = np.linspace(-10, 10, 500)
    y = np.linspace(-10, 10, 500)
    Z = ObjectiveFunction(np.meshgrid(x, y))

    surf = go.Surface(x=x, y=y, z=Z, name="f(x,y)")
    fig = go.Figure(data=[surf])
    fig.show()

    fig.write_image("objective.png")


def plotSubDiff():
    # Plot the smooth part
    x, y = np.linspace(-4, 4, 40), np.linspace(-4, 4, 40)
    X, Y = np.meshgrid(x, y)
    u, v = np.zeros(X.shape), np.zeros(Y.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            u[i, j] = SubDifferential(np.array([X[i, j], Y[i, j]]))[0]
            v[i, j] = SubDifferential(np.array([X[i, j], Y[i, j]]))[1]

    # Normalize the vectors, so they are direction vectors
    w = u ** 2 + v ** 2
    u = u / np.sqrt(w)
    v = v / np.sqrt(w)
    fig = ff.create_quiver(X, Y, -u, -v)

    # Plot the non-smooth part |-4x_1+2x_2+5|
    f_1x = np.linspace(-4, 4, 40)
    f_1y = (4 * f_1x - 4) / 2
    x_1, y_1, u_1, v_1 = createNonSmoothQuiverVectors(np.array([f_1x, f_1y]))

    # Normalize the vectors, so they are direction vectors
    w = u_1 ** 2 + v_1 ** 2
    u_1 = u_1 / np.sqrt(w)
    v_1 = v_1 / np.sqrt(w)

    quiver1 = ff.create_quiver(x_1, y_1, -u_1, -v_1, line=dict(color="red"))
    fig.add_traces(data=quiver1.data)

    f_1plot = go.Scatter(x=x_1, y=y_1, line=dict(color="red"))
    fig.add_trace(f_1plot)

    # Plot the non-smooth part |2x_1+3x_2-2|
    f_2x = np.linspace(-4, 4, 40)
    f_2y = (2 - 2 * f_2x) / 3
    x_2, y_2, u_2, v_2 = createNonSmoothQuiverVectors(np.array([f_2x, f_2y]))
    # Normalize the vectors, so they are direction vectors
    w = u_2 ** 2 + v_2 ** 2
    u_2 = u_2 / np.sqrt(w)
    v_2 = v_2 / np.sqrt(w)
    quiver2 = ff.create_quiver(x_2, y_2, -u_2, -v_2, line=dict(color="green"))

    fig.add_traces(data=quiver2.data)

    f_2plot = go.Scatter(x=x_2, y=y_2, line=dict(color="green"))
    fig.add_trace(f_2plot)

    fig.update_layout(
        title="Графік субдиференціалу",
        yaxis_range=[-4, 4],
        xaxis_range=[-4, 4],
    )

    fig.write_image("subdifferential.png")
    fig.show()


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
    tolerance=1e-6,
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
            abs(vals[-1] - vals[-2]) < tolerance
            and np.linalg.norm(points[-1] - points[-2]) < tolerance
        ):
            break

    return points, vals, best_point, best_value


if __name__ == "__main__":
    plotObjective()
    plotSubDiff()

    global_min = ObjectiveFunction(gb)
    plt.xlabel("Ітерація")
    plt.ylabel("$|f(x) - f(x^{\\ast})|$")
    # region const_size
    const_size = {}
    params = [0.01, 0.001, 0.0001]
    for h in params:
        const_size[f"h={h}"] = optimize(
            ObjectiveFunction,
            SubGradient,
            (10, 10),
            strategy=Strategy.consant_size,
            h=h,
            max_iterations=10 ** 3,
        )

    for key in const_size.keys():
        plt.plot(
            range(len(const_size[key][1])),
            list(map(lambda x: abs(global_min - x), const_size[key][1])),
            label=f"${key}$",
        )
        print(
            f"{key}:\t{const_size[key][2]}\t{const_size[key][3]}\t{abs(const_size[key][3] - global_min)}"
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
    params = [0.01, 0.001, 0.0001]
    for h in params:
        const_len[f"h={h}"] = optimize(
            ObjectiveFunction,
            SubGradient,
            (0, 0),
            strategy=Strategy.constant_length,
            h=h,
            max_iterations=10 ** 3,
        )
    for key in const_len.keys():
        plt.plot(
            range(len(const_len[key][1])),
            list(map(lambda x: abs(global_min - x), const_len[key][1])),
            label=f"{key}$",
        )
        print(
            f"{key}:\t{const_len[key][2]}\t{const_len[key][3]}\t{abs(const_len[key][3] - global_min)}"
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
    params = [(0.1, 1.0), (0.05, 1.0), (0.05, 10.0)]
    for (a, b) in params:
        square_summable[f"a={a}, b={b}"] = optimize(
            ObjectiveFunction,
            SubGradient,
            (0, 0),
            strategy=Strategy.square_summable,
            a=a,
            b=b,
            max_iterations=10 ** 3,
        )
    for key in square_summable.keys():
        plt.plot(
            range(len(square_summable[key][1])),
            list(map(lambda x: abs(global_min - x), square_summable[key][1])),
            label=f"${key}$",
        )
        print(
            f"{key}:\t{square_summable[key][2]}\t{square_summable[key][3]}\t{abs(square_summable[key][3] - global_min)}"
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
    params = [0.1, 0.05, 0.01]
    for a in params:
        diminishing[f"a={a}"] = optimize(
            ObjectiveFunction,
            SubGradient,
            (0, 0),
            strategy=Strategy.diminishing,
            a=a,
            max_iterations=10 ** 3,
        )
    for key in diminishing.keys():
        plt.plot(
            range(len(diminishing[key][1])),
            list(map(lambda x: abs(global_min - x), diminishing[key][1])),
            label=f"${key}$",
        )
        print(
            f"{key}:\t{diminishing[key][2]}\t{diminishing[key][3]}\t{abs(diminishing[key][3] - global_min)}"
        )

    plt.legend()
    plt.savefig("diminishing.png")
    plt.clf()
    diminishing.clear()
    # endregion
