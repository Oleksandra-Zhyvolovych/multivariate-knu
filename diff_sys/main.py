import numpy as np
import plotly.graph_objects as go

import os

if not os.path.exists("images"):
    os.mkdir("images")

# Вихідні дані задачі
dim = 3
t0 = 0
T = 10
C1 = C2 = 2
R1 = R2 = 1
L = 1
time_step = 0.1
initial = np.ones(dim)

print("Вихідні дані:")
print(f"Часовий проміжок: [{t0, T}]")
print(f"Часовий крок: {time_step}")
print(f"Параметри моделі:")
print(f"R_1 = R_2 = {R1}")
print(f"C_1 = C_2 = {C1}")
print(f"L = {L}")
print(f"x0 = {initial}")


def v(t):
    return 10 * np.sin(t)


def system_f(t):
    return np.array([1 / (C1 * R1) * v(t), 1 / (C1 * R1) * v(t), 0])


def build_system_matrix():
    return np.array(
        [
            [-1 / C1 * (1 / R1 + 1 / R2), 1 / (C1 * R2), 0],
            [-1 / C1 * (1 / R1 + 1 / R2), -1 / C1 * (C1 * R2 / L - 1 / R2), R2 / L],
            [1 / (R2 * C2), -1 / (R2 * C2), 0],
        ]
    )


def RK4(x_0, t_0, T, rhs, time_step):
    values = [x_0]
    cur_t = t_0
    while cur_t < T:
        k1 = rhs(values[-1], cur_t)
        k2 = rhs(values[-1] + k1 * time_step / 2, cur_t + time_step / 2)
        k3 = rhs(values[-1] + k2 * time_step / 2, cur_t + time_step / 2)
        k4 = rhs(values[-1] + k3 * time_step, cur_t + time_step)
        values.append(values[-1] + time_step / 6 * (k1 + 2 * k2 + 2 * k3 + k4))
        cur_t += time_step
    return np.array(values)


def build_ellipsoid(p, B, quality: int = 15) -> np.ndarray:
    phi = np.linspace(0, 2 * np.pi, quality)
    theta = np.linspace(0, np.pi, quality)
    phi, theta = np.meshgrid(phi, theta)

    eigenvalues, eigenvectors = np.linalg.eig(B)

    x = np.sqrt(eigenvalues[0]) * np.sin(theta) * np.cos(phi)
    y = np.sqrt(eigenvalues[1]) * np.sin(theta) * np.sin(phi)
    z = np.sqrt(eigenvalues[2]) * np.cos(theta)

    x = x.flatten()
    y = y.flatten()
    z = z.flatten()

    values = np.vstack((x, y, z))

    real_values = eigenvectors.dot(values) + p[:, np.newaxis]
    return real_values[0], real_values[1], real_values[2]


if __name__ == "__main__":
    A = build_system_matrix()

    def diff_equation_rhs(x, t):
        return system_f(t) + np.dot(A, x)

    ellipsoid_center_series = RK4(initial, t0, T, diff_equation_rhs, time_step)
    print(f"Кінцевий стан системи на момент T=10 : {ellipsoid_center_series[-1]}")
    # region plot center
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=[p[0] for p in ellipsoid_center_series],
                y=[p[1] for p in ellipsoid_center_series],
                z=[p[2] for p in ellipsoid_center_series],
                marker=dict(size=2),
                line=dict(color="green", width=2),
                name="Траєкторія",
            ),
            go.Scatter3d(
                x=[initial[0]],
                y=[initial[1]],
                z=[initial[2]],
                marker=dict(size=5),
                line=dict(color="blue", width=2),
                name="Початкова точка",
            ),
        ]
    )

    fig.update_layout(
        legend=dict(orientation="h", x=0, y=0),
        scene=dict(
            xaxis=dict(title=f"x"),
            yaxis=dict(title=f"y"),
            zaxis=dict(title=f"z"),
        ),
        width=800,
        height=600,
    )
    fig.show()
    fig.write_image("images/trajectory_center.png")

    # endregion
    # region plot ellipsoid
    ellipsoids_series = []

    Q0 = np.diag(np.ones(3)) * 0.5

    fundamental_matrix_equation_rhs = lambda Q, t: A @ Q + Q @ A.T

    Q_series = RK4(Q0, t0, T, fundamental_matrix_equation_rhs, time_step)

    for i, cur_Q in enumerate(Q_series):
        ellipsoids_series.append(build_ellipsoid(ellipsoid_center_series[i], cur_Q))

    # Подивимось наскільки близько збігаються розв'язки задачі для кожної початкової  точки з розв'язком для центральної точки
    max_distances_from_center = (
        max(abs(ellipsoids_series[-1][0] - ellipsoid_center_series[-1][0])),
        max(abs(ellipsoids_series[-1][1] - ellipsoid_center_series[-1][1])),
        max(abs(ellipsoids_series[-1][2] - ellipsoid_center_series[-1][2])),
    )
    print(
        f"Максимальні відхилення по осям від центральної точки :{max_distances_from_center}"
    )

    fig = go.Figure(
        data=go.Mesh3d(
            {
                "x": ellipsoids_series[0][0],
                "y": ellipsoids_series[0][1],
                "z": ellipsoids_series[0][2],
                "alphahull": 0,
            }
        ),
        frames=[
            go.Frame(
                data=go.Mesh3d(
                    {
                        "x": ellipsoids_series[i][0],
                        "y": ellipsoids_series[i][1],
                        "z": ellipsoids_series[i][2],
                        "alphahull": 0,
                    }
                ),
                name=str(i),
            )
            for i in range(len(ellipsoids_series))
        ],
    )

    def frame_args(duration):
        return {
            "frame": {"duration": duration},
            "mode": "immediate",
            "fromcurrent": True,
            "transition": {"duration": duration, "easing": "linear"},
        }

    sliders = [
        {
            "pad": {"b": 10, "t": 60},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [
                {
                    "args": [[f.name], frame_args(0)],
                    "label": str(k),
                    "method": "animate",
                }
                for k, f in enumerate(fig.frames)
            ],
        }
    ]

    fig.update_layout(
        scene=dict(
            xaxis=dict(
                range=[-ellipsoid_center_series.max(), ellipsoid_center_series.max()],
                title=f"x{1}",
            ),
            yaxis=dict(
                range=[-ellipsoid_center_series.max(), ellipsoid_center_series.max()],
                title=f"x{2}",
            ),
            zaxis=dict(
                range=[-ellipsoid_center_series.max(), ellipsoid_center_series.max()],
                title=f"x{3}",
            ),
        ),
        width=800,
        height=800,
        margin=dict(r=10, l=10, b=10, t=10),
        scene_aspectmode="manual",
        scene_aspectratio=dict(x=1, y=1, z=1),
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [None, frame_args(50)],
                        "label": "&#9654;",  # play symbol
                        "method": "animate",
                    },
                    {
                        "args": [[None], frame_args(0)],
                        "label": "&#9724;",  # pause symbol
                        "method": "animate",
                    },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 70},
                "type": "buttons",
                "x": 0.1,
                "y": 0,
            }
        ],
        sliders=sliders,
    )

    fig.show()
    # endregion