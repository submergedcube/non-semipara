from .nonparametric_modules import (
    nadaraya_watson_estimator,
    Kernel,
)
import numpy as np
from scipy.optimize import minimize


def calculate_robinson_estimator(
    X: np.array, y: np.array, Z: np.array, kernel: Kernel, band_width: float
) -> np.array:
    """部分線形モデルに対するRobinsonの推定量(p.75)のβを求める
    新規の予測をするときは、新規のZからNadaraya-WatsonでX,yを予測(eta_x,eta_y)し、[1,(X-eta_x)]@beta+eta_yを計算する

    Args:
        X (np.array): 線形で使う説明変数。shape=(n, 1)のみ受け付ける。
        y (np.array): 目的変数。shape=(n,)
        Z (np.array): ノンパラメトリック回帰で影響を考える説明変数。shape=(n,)
        kernel (Kernel): 内部のNadaraya-Watson推定を使うときに使うカーネル
        band_width (float): 内部のNadaraya-Watson推定を使うときに使うバンド幅

    Returns:
        _type_: _description_
    """
    eta_x_hat = np.array(
        [nadaraya_watson_estimator(Z, X, kernel, i, band_width) for i in Z]
    )
    eta_y_hat = np.array(
        [nadaraya_watson_estimator(Z, y, kernel, i, band_width) for i in Z]
    )
    new_X = (X - eta_x_hat).reshape(-1, 1)
    new_X = np.hstack([np.array([1] * X.shape[0]).reshape(-1, 1), new_X])
    new_Y = y - eta_y_hat

    beta = np.linalg.inv(
        np.array(
            [
                new_X[i].reshape(-1, 1) @ new_X[i].reshape(1, -1)
                for i in range(new_X.shape[0])
            ]
        ).sum(axis=0)
    ) @ (new_X * new_Y.reshape(-1, 1)).sum(axis=0)

    return beta


def calculate_ichimura_estimator(
    X1: np.array, y1: np.array, band_width: float, kernel: Kernel, verbose=False
) -> np.array:
    """市村推定量(p.78)を計算する。説明変数が少しでも大きくなると途端に遅くなる。

    Args:
        X1 (np.array): 説明変数
        y1 (np.array): 目的変数
        band_width (float): NW推定量に使うバンド幅
        kernel (Kernel): NW推定量に使うカーネル
        verbose (bool, optional): デバッグ用のフラグ. Defaults to False.


    Returns:
        np.array: 説明変数と内積を取るβの第二成分以降。δハット。最適化の制約上βの第一成分は1に固定されている。内積をカーネルの引数としてNWを取ればよい。
    """

    def print_iter(xk):
        print_iter.count += 1
        print(
            f"iteration {print_iter.count}: x = {np.array2string(xk, precision=2, separator=', ')}"
        )

    def LOO_NW(
        x: np.array,
        y: np.array,
        s: np.array,
        b: np.array,
        band_width: float,
        kernel: Kernel,
        index: int,
    ):
        """ただの一個抜きNadaraya-Watson推定量。"""
        if index > x.shape[0]:
            raise ValueError
        x = np.vstack([x[:index], x[index + 1 :]])
        y = np.hstack([y[:index], y[index + 1 :]])

        result = nadaraya_watson_estimator(x @ b, y, kernel, s @ b, band_width)
        return result

    def calculate_error(
        b: np.array, xx: np.array, yy: np.array, band_width: float, kernel: Kernel
    ):
        """最適化用の評価関数。"""
        sum = 0
        for i in range(xx.shape[0]):
            estimated_y = LOO_NW(
                xx,
                yy,
                xx[i, :],
                np.vstack([np.array([1]), b.reshape(-1, 1)]),
                band_width,
                kernel,
                i,
            )
            if np.isnan(estimated_y):
                estimated_y == 1e12
            sum += (yy[i] - estimated_y) ** 2
        return sum

    # カウンタを初期化
    print_iter.count = 0
    if verbose:
        result = minimize(
            calculate_error,
            np.array([0.1] * (X1.shape[1] - 1)),
            args=(X1, y1, band_width, kernel),
            method="Nelder-Mead",  # Powellも使えるらしいがなぜかnanが帰ってきがち。
            callback=print_iter,
        )
    else:
        result = minimize(
            calculate_error,
            np.array([0.1] * (X1.shape[1] - 1)),
            args=(X1, y1, band_width, kernel),
            method="Nelder-Mead",
        )
    return result["x"]
