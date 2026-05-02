import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def objective(x, y):
    valley1 = -3.0 * np.exp(-((x - 2) ** 2 + (y + 3) ** 2) / 2.0)
    valley2 = -2.5 * np.exp(-((x + 3) ** 2 + (y - 1) ** 2) / 1.5)
    base = 0.05 * (x ** 2 + y ** 2)
    return base + valley1 + valley2


def load_paths(csv_paths):
    """
    CSVファイルを読み込み，探索経路の情報を辞書のリストとして返す
    """
    paths = []

    for csv_path_str in csv_paths:
        csv_path = Path(csv_path_str)

        if not csv_path.exists():
            raise FileNotFoundError(f"CSVファイルが見つかりません: {csv_path}")

        df = pd.read_csv(csv_path)

        # 必要な列があるか確認
        required_columns = ["iteration", "x", "y", "value", "moved"]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"{csv_path} に列 {col} がありません")

        paths.append({
            "name": csv_path.stem,
            "df": df,
        })

    return paths


def draw_paths_2d(plt_obj, paths):
    """
    2Dグラフに探索経路を重ね描きする
    """
    for path_info in paths:
        df = path_info["df"]
        name = path_info["name"]

        line = plt_obj.plot(
            df["x"],
            df["y"],
            linewidth=2,
            marker="o",
            markersize=3,
            label=name
        )[0]

        color = line.get_color()

        # 開始点
        plt_obj.scatter(
            df["x"].iloc[0],
            df["y"].iloc[0],
            marker="s",
            s=80,
            color=color
        )

        # 終了点
        plt_obj.scatter(
            df["x"].iloc[-1],
            df["y"].iloc[-1],
            marker="*",
            s=150,
            color=color
        )


def draw_paths_3d(ax, paths):
    """
    3Dグラフに探索経路を重ね描きする
    """
    for path_info in paths:
        df = path_info["df"]
        name = path_info["name"]

        line = ax.plot(
            df["x"],
            df["y"],
            df["value"],
            linewidth=2,
            marker="o",
            markersize=3,
            label=name
        )[0]

        color = line.get_color()

        # 開始点
        ax.scatter(
            df["x"].iloc[0],
            df["y"].iloc[0],
            df["value"].iloc[0],
            marker="s",
            s=80,
            color=color
        )

        # 終了点
        ax.scatter(
            df["x"].iloc[-1],
            df["y"].iloc[-1],
            df["value"].iloc[-1],
            marker="*",
            s=150,
            color=color
        )


def main():
    parser = argparse.ArgumentParser(
        description="目的関数の景観と探索経路を可視化する"
    )
    parser.add_argument(
        "csv_paths",
        nargs="*",
        help="探索結果CSVファイルのパス（0個以上指定可能）"
    )
    args = parser.parse_args()

    # 結果保存用フォルダ
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # CSV読み込み（0個でも可）
    paths = load_paths(args.csv_paths)

    # 表示範囲
    x_min, x_max = -6.0, 6.0
    y_min, y_max = -8.0, 4.0

    # 格子点を作成
    x = np.linspace(x_min, x_max, 200)
    y = np.linspace(y_min, y_max, 200)
    X, Y = np.meshgrid(x, y)

    # 各点で目的関数値を計算
    Z = objective(X, Y)

    # 代表的な谷の位置
    valley1_x = 2.0
    valley1_y = -3.0
    valley1_z = objective(valley1_x, valley1_y)

    valley2_x = -3.0
    valley2_y = 1.0
    valley2_z = objective(valley2_x, valley2_y)

    # 出力ファイル名
    if len(paths) == 0:
        output_2d = results_dir / "fitness_landscape_2d.png"
        output_3d = results_dir / "fitness_landscape_3d.png"
        output_gif = results_dir / "fitness_landscape_3d_rotation.gif"
    else:
        output_2d = results_dir / "fitness_landscape_with_paths_2d.png"
        output_3d = results_dir / "fitness_landscape_with_paths_3d.png"
        output_gif = results_dir / "fitness_landscape_with_paths_3d_rotation.gif"

    # =========================
    # 1. 2Dカラーマップ
    # =========================
    plt.figure(figsize=(7, 6))

    contour = plt.contourf(X, Y, Z, levels=30, cmap="viridis")
    plt.colorbar(contour, label="Objective value")

    # 谷の位置を表示
    plt.scatter(valley1_x, valley1_y, marker="X", s=150, label="Valley 1")
    plt.scatter(valley2_x, valley2_y, marker="X", s=150, label="Valley 2")

    # 探索経路を表示（引数がある場合のみ）
    if len(paths) > 0:
        draw_paths_2d(plt, paths)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Fitness Landscape: 2D Color Map")
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_2d, dpi=300)

    # =========================
    # 2. 3Dサーフェス
    # =========================
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    surface = ax.plot_surface(
        X,
        Y,
        Z,
        cmap="viridis",
        edgecolor="none",
        alpha=0.85
    )

    # 谷の位置を表示
    ax.scatter(valley1_x, valley1_y, valley1_z, s=80, label="Valley 1")
    ax.scatter(valley2_x, valley2_y, valley2_z, s=80, label="Valley 2")

    # 探索経路を表示（引数がある場合のみ）
    if len(paths) > 0:
        draw_paths_3d(ax, paths)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("Objective value")
    ax.set_title("Fitness Landscape: 3D Surface")

    fig.colorbar(surface, ax=ax, shrink=0.6, label="Objective value")
    ax.legend()
    plt.tight_layout()

    plt.savefig(output_3d, dpi=300)

    # =========================
    # 3. 3D回転アニメーション
    # =========================
    fig_anim = plt.figure(figsize=(8, 6))
    ax_anim = fig_anim.add_subplot(111, projection="3d")

    surface_anim = ax_anim.plot_surface(
        X,
        Y,
        Z,
        cmap="viridis",
        edgecolor="none",
        alpha=0.85
    )

    ax_anim.scatter(valley1_x, valley1_y, valley1_z, s=80, label="Valley 1")
    ax_anim.scatter(valley2_x, valley2_y, valley2_z, s=80, label="Valley 2")

    if len(paths) > 0:
        draw_paths_3d(ax_anim, paths)

    ax_anim.set_xlabel("x")
    ax_anim.set_ylabel("y")
    ax_anim.set_zlabel("Objective value")
    ax_anim.set_title("Fitness Landscape: 3D Surface")

    fig_anim.colorbar(surface_anim, ax=ax_anim, shrink=0.6, label="Objective value")
    ax_anim.legend()

    def update(frame):
        ax_anim.view_init(elev=30, azim=frame)
        return fig_anim,

    animation = FuncAnimation(
        fig_anim,
        update,
        frames=np.arange(0, 360, 5),
        interval=100
    )

    animation.save(output_gif, writer=PillowWriter(fps=10))

    print(f"2D画像を保存しました: {output_2d}")
    print(f"3D画像を保存しました: {output_3d}")
    print(f"3D回転GIFを保存しました: {output_gif}")


if __name__ == "__main__":
    main()