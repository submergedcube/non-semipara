import numpy as np
from abc import ABC, abstractmethod
import pandas as pd
from scipy.optimize import minimize


class Kernel(ABC):
    def __init__(self, mean: float = 0.0):
        """
        基底カーネルクラス

        Args:
            mean (float, optional):
                カーネル中心
            variance (float, optional):
                カーネル分散
            k (float, optional):
                二乗積分値
        """
        self.mean = mean
        self.variance = self.THEORETICAL_VARIANCE
        self.std = np.sqrt(self.variance)
        self.k = self.THEORETICAL_K

    @abstractmethod
    def __call__(self, x: float) -> float:
        """カーネル関数を計算"""
        pass

    def get_params(self) -> dict:
        """カーネルのパラメータを取得"""
        return {"mean": self.mean, "variance": self.variance, "k": self.k}

    def get_kappa_star(self) -> float:
        """p57,(3.13)の値を返す。κ２スター"""
        kappa_star = (self.v2**2 - self.v1 * self.v3) / (self.v0 * self.v2 - self.v1**2)
        return kappa_star

    @abstractmethod
    def get_R_star(self) -> float:
        """手計算したR*(k*) P.66を返す"""
        pass


class GaussianKernel(Kernel):
    THEORETICAL_K: float = 1 / np.sqrt(2 * np.pi)
    THEORETICAL_VARIANCE: float = 1.0
    v0: float = 0.5
    v1: float = 1 / np.sqrt(2 * np.pi)
    v2: float = 0.5
    v3: float = 2 / np.sqrt(2 * np.pi)
    p: float = v2 / (v0 * v2 - v1**2)
    q: float = v1 / (v0 * v2 - v1**2)

    def __call__(self, x: float) -> float:
        return np.exp(-1 * (x - self.mean) ** 2 / 2) / np.sqrt(2 * np.pi)

    def get_R_star(self):
        return (
            self.p**2 / 4 / np.sqrt(np.pi)
            - self.p * self.q / 2 / np.pi
            + self.q**2 * self.THEORETICAL_K / 2
        )


class TriangularKernel(Kernel):
    THEORETICAL_K: float = 2 / 3
    THEORETICAL_VARIANCE: float = 1 / 6
    v0: float = 0.5
    v1: float = 1 / 6
    v2: float = 1 / 12
    v3: float = 1 / 20
    p: float = v2 / (v0 * v2 - v1**2)
    q: float = v1 / (v0 * v2 - v1**2)

    def __call__(self, x: float) -> float:
        if np.abs(x) <= 1:
            return float(1 - np.abs(x))
        else:
            return 0

    def get_R_star(self):
        return self.p**2 / 3 - self.p * self.q / 6 + self.q**2 * self.THEORETICAL_K / 2


class RectangularKernel(Kernel):
    THEORETICAL_K: float = 0.5
    THEORETICAL_VARIANCE: float = 1 / 3
    v0: float = 0.5
    v1: float = 1 / 4
    v2: float = 1 / 6
    v3: float = 1 / 8
    p: float = v2 / (v0 * v2 - v1**2)
    q: float = v1 / (v0 * v2 - v1**2)

    def __call__(self, x: float) -> float:
        if np.abs(x) <= 1:
            return 1 / 2
        else:
            return 0

    def get_R_star(self):
        return self.p**2 / 4 - self.p * self.q / 4 + self.q**2 * self.THEORETICAL_K / 2


class EpanechnikovKernel(Kernel):
    THEORETICAL_K: float = 0.6
    THEORETICAL_VARIANCE: float = 0.2
    v0: float = 0.5
    v1: float = 3 / 16
    v2: float = 1 / 10
    v3: float = 1 / 16
    p: float = v2 / (v0 * v2 - v1**2)
    q: float = v1 / (v0 * v2 - v1**2)

    def __call__(self, x: float) -> float:
        if np.abs(x) <= 1:
            return 3 * (1 - x**2) / 4
        else:
            return 0

    def get_R_star(self):
        return (
            3 * self.p**2 / 10
            - 3 * self.p * self.q / 4
            + self.q**2 * self.THEORETICAL_K / 2
        )


class BiweightKernel(Kernel):
    THEORETICAL_K: float = 5 / 7
    THEORETICAL_VARIANCE: float = 1 / 7
    v0: float = 0.5
    v1: float = 5 / 32
    v2: float = 1 / 14
    v3: float = 5 / 128
    p: float = v2 / (v0 * v2 - v1**2)
    q: float = v1 / (v0 * v2 - v1**2)

    def __call__(self, x: float) -> float:
        if np.abs(x) <= 1:
            return 15 * (1 - x**2) ** 2 / 16
        else:
            return 0

    def get_R_star(self):
        return (
            5 * self.p**2 / 14
            - self.p * self.q * 45 / 256
            + self.q**2 * self.THEORETICAL_K / 2
        )


def nadaraya_watson_estimator(
    x: np.array, y: np.array, kernel: Kernel, s: float, band_width: float
) -> float:
    """Nadaraya Watson estimator

    Args:
        x (np.array): 共変量or説明変数。一次元。
        y (np.array): 目的変数
        kernel (Kernel): カーネル
        s (float): 新たな共変量
        band_width (float): バンド幅

    Returns:
        float: 予測値
    """
    tmp = [(i - s) / band_width for i in x]
    kernel_series = np.array([kernel(i) for i in tmp])

    return (kernel_series * y).sum() / kernel_series.sum()


def local_polinomial_estimator(
    x: np.array, y: np.array, kernel: Kernel, s: float, band_width: float, q: int
) -> np.array:
    """Local polinomial estimator

    Args:
         x (np.array): 共変量or説明変数。一次元。
        y (np.array): 目的変数
        kernel (Kernel): カーネル
        s (float): 新たな共変量
        band_width (float): バンド幅
        q (int): 次数。1なら局所線形推定になる。

    Returns:
        np.array: 予測値を含むβ。予測値は0成分
    """
    # カーネルの値関連
    tmp = [(i - s) / band_width for i in x]
    kernel_series = np.array([kernel(i) for i in tmp])

    # Zを作る
    z1 = np.array([x[i] - s for i in range(len(x))])
    Z = np.array([z1**i for i in range(q + 1)])
    ZZ = np.sum(
        [
            kernel_series[i] * np.array([Z[:, i]]).T @ np.array([Z[:, i]])
            for i in range(Z.shape[1])
        ],
        axis=0,
    )
    # 本当は全部exceptの方でいい
    try:
        ZZ_inv = np.linalg.inv(ZZ)
    except np.linalg.LinAlgError:
        ZZ_inv = np.linalg.pinv(ZZ)
    beta = np.sum(
        [kernel_series[i] * y[i] * np.array([Z[:, i]]).T for i in range(Z.shape[1])],
        axis=0,
    )

    beta = ZZ_inv @ beta

    return beta


def convert_and_square(data):
    # リストをNumPy配列に変換
    arr = np.array(data, dtype=float)

    # NaNを0に置換
    arr = np.nan_to_num(arr, nan=0.0)

    # 0でない要素を2乗
    arr = np.where(arr != 0, arr**2, arr)

    return arr


# ここではp53 h_optの式に基づき最適なバンド幅を計算する。
def calculate_optimal_bandwidth(
    X_train: np.array, y_train: np.array, band_width: float, kernel: Kernel, p: int = 2
):
    n = X_train.shape[0]
    min_x = pd.Series(X_train).quantile(0.02)
    max_x = pd.Series(X_train).quantile(0.98)
    x = min_x
    mudds = []
    fxs = []
    sigmas = []

    step = 50
    for i in range(step):
        beta = local_polinomial_estimator(X_train, y_train, kernel, x, band_width, p)
        mudds.append(beta[2][0])
        left = x
        right = x + (max_x - min_x) / step
        # 確率関数を雑に算出する。
        fxs.append(((X_train >= left) & (X_train < right)).sum() / n)
        # σを雑に計算する。
        sigmas.append(y_train[(X_train >= left) & (X_train < right)].std())
        x = right

    integral_step = (max_x - min_x) / step
    numerator_integral = (convert_and_square(sigmas) * integral_step).sum()
    denominator_integral = (convert_and_square(mudds) * np.array(fxs)).sum()
    h_opt = (
        kernel.get_params()["k"]
        * numerator_integral
        / (kernel.get_params()["variance"] ** 2 * denominator_integral * n)
    ) ** (0.2)  # ここ教科書と違うけどこっちが正解。

    return h_opt


# シリーズ法用にデータを整形する関数
def convert_for_series_method_data(s: np.array, r: int = 3, K: int = 10, knots=[]):
    """_summary_

    Args:
        s (np.array): _description_
        r (int, optional): 最大次数. Defaults to 3.
        K (int, optional): ノット数(区間を分割する個数. Defaults to 10.
        knots (list, optional): _description_. Defaults to [].

    Returns:
        _type_: _description_
    """
    if len(knots) == 0:
        knot_h = (s.max() - s.min()) / K
        knots = np.array([s.min() + (i + 1) * knot_h for i in range(K - 1)])
    df = pd.concat([pd.Series(s) ** i for i in range(r + 1)], axis=1)
    for j in range(1, K, 1):
        df[f"l{j}"] = [max(i - knots[j - 1], 0) ** r for i in df[1]]
    p = df.to_numpy().T
    return p, knots


def calculate_CV_LOO(q, y, beta):
    QQ_inv = np.linalg.inv(
        np.sum(
            [
                q[:, i].reshape(-1, 1) @ q[:, i].reshape(1, -1)
                for i in range(q.shape[1])
            ],
            axis=0,
        )
    )
    sum = 0
    for i in range(q.shape[1]):
        sum += (
            (y[i] - (q[:, i] @ beta)[0])
            / (1 - q[:, i] @ QQ_inv @ (q[:, i].reshape(-1, 1)))
        ) ** 2
    return sum / q.shape[1]


def calculate_optimal_K(s: np.array, y: np.array, max_K: int = 50):
    CVs = []
    try_K = range(1, max_K)
    for k in try_K:
        q, _ = convert_for_series_method_data(s, K=k)
        beta = np.linalg.inv(q @ q.T) @ q @ y
        CVs.append(calculate_CV_LOO(q, y, beta))

    best_k = try_K[np.argmin(CVs)]

    return best_k


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
            method="Nelder-Mead",  # Powellも使えるらしいがなぜかnanが返ってきがち。
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
