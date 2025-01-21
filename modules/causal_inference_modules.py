from matplotlib.pylab import beta
from sklearn.linear_model import LogisticRegression
import numpy as np
from .nonparametric_modules import (
    convert_for_series_method_data,
)


def calculate_ATE_by_regression_adjustment(
    df_X_0: np.array, df_y_0: np.array, df_X_1: np.array, df_y_1: np.array
) -> float:
    """回帰調整法による介入効果の推定

    Args:
        df_X_0 (np.array): 非介入群の特徴量
        df_y_0 (np.array): 非介入群の目的変数
        df_X_1 (np.array): 介入群の特徴量
        df_y_1 (np.array): 介入群の目的変数

    Returns:
        float: 介入効果の推定値
    """
    df_X_0 = np.hstack([np.ones(df_X_0.shape[0]).reshape(-1, 1), df_X_0])
    df_X_1 = np.hstack([np.ones(df_X_1.shape[0]).reshape(-1, 1), df_X_1])

    beta_0 = np.linalg.inv(df_X_0.T @ df_X_0) @ df_X_0.T @ df_y_0
    beta_1 = np.linalg.inv(df_X_1.T @ df_X_1) @ df_X_1.T @ df_y_1

    All_X = np.vstack([df_X_0, df_X_1])

    mu0 = All_X @ beta_0
    mu1 = All_X @ beta_1

    tau_hat_ate_RA = np.mean(mu1 - mu0)
    return tau_hat_ate_RA


def calculate_ATE_by_IWP(X: np.array, Z: np.array, y: np.array) -> float:
    """介入効果を逆確率重み付け法で推定する。(p.80)

    Args:
        X (np.array): 説明変数。1次元配列でなくてもよい。
        Z (np.array): 介入変数。1次元配列であること。
        y (np.array): 目的変数。1次元配列であること。

    Returns:
        float: 介入効果の推定値。
    """
    lr = LogisticRegression()
    lr.fit(X, Z)
    p = 1 / (1 + np.exp(-(X @ np.array(lr.coef_[0]) + lr.intercept_[0])))

    dyp = Z * y / p
    second_dyp = (1 - Z) * y / (1 - p)
    return (dyp - second_dyp).mean()


def calculate_ATE_by_hahn_method(X: np.array, Z: np.array, y: np.array) -> list:
    """平均介入効果をHahnの方法とHiranoの方法で推定する(p.80)

    Args:
        X (np.array): 説明変数。1次元配列であること。
        Z (np.array): 介入変数。1次元配列であること。
        y (np.array): 目的変数。1次元配列であること。

    Returns:
        list: 介入効果の推定値。第一成分がHahnの方法の推定量,Hiranoの方法の推定量。
    """
    X = X.reshape(1, -1)[0]
    Z = Z.reshape(-1, 1)
    y = y.reshape(-1, 1)
    q, knots = convert_for_series_method_data(X)

    beta1 = np.linalg.inv(q @ q.T) @ q @ (y * Z)
    eta_hat_d = q.T @ beta1

    beta2 = np.linalg.inv(q @ q.T) @ q @ (y * (1 - Z))
    eta_hat_1d = q.T @ beta2

    beta3 = np.linalg.inv(q @ q.T) @ q @ Z
    p = q.T @ beta3

    eta_hat_hahn = (eta_hat_d / p - eta_hat_1d / (1 - p)).sum() / y.shape[0]
    eta_hat_hirano = (y * Z / p - y * (1 - Z) / (1 - p)).sum() / y.shape[0]
    return eta_hat_hahn, eta_hat_hirano
