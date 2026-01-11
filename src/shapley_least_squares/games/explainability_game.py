from abc import abstractmethod
from typing import override

import click
import numpy as np
from sklearn.datasets import fetch_california_housing, load_diabetes, load_wine
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    mean_absolute_percentage_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline

from shapley_least_squares.games.utils.base_game import BaseGame


class _FeatureCoalitionGame(BaseGame):
    def __init__(self, model, dataset_loading_func) -> None:
        X, y = dataset_loading_func(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        rng = np.random.default_rng(seed=42)
        self._datapoint_to_be_explained = X_test[rng.integers(0, X_test.shape[0])]
        self._feature_means = X_train.mean(axis=0)
        self._n = X_train.shape[1]

        self._model = model
        self._model.fit(X_train, y_train)
        self._eval_model(X_test, y_test)

        self._empty_set_pred = self._predict(self._feature_means)

    @override
    def v(self, S: list[int] | np.ndarray) -> float:
        S = np.array(S, dtype=int)

        if len(S) == 0:
            return 0.0

        datapoint_to_be_explained = self._datapoint_to_be_explained.copy()
        mask = np.ones(self._n, dtype=bool)
        mask[S] = False
        datapoint_to_be_explained[mask] = self._feature_means[mask]

        return self._predict(datapoint_to_be_explained) - self._empty_set_pred

    @abstractmethod
    def _predict(self) -> float:
        raise NotImplementedError()

    @abstractmethod
    def _eval_model(self, X_test: np.ndarray, y_test: np.ndarray) -> None:
        raise NotImplementedError()


class RegressionFeatureCoalitionGame(_FeatureCoalitionGame):
    def __init__(self, model, dataset_loading_func) -> None:
        super().__init__(model, dataset_loading_func)

    @override
    def _predict(self, x: np.ndarray) -> float:
        return self._model.predict(x.reshape(1, -1))[0]

    @override
    def _eval_model(self, X_test: np.ndarray, y_test: np.ndarray) -> None:
        y_pred = self._model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_percentage_error(y_test, y_pred)
        rmse = np.sqrt(root_mean_squared_error(y_test, y_pred))

        click.echo("\nModel Evaluation:")
        click.echo(f"R² Score: {r2:.4f}")
        click.echo(f"MAPE:     {mae:.4f}")
        click.echo(f"RMSE:     {rmse:.4f}")


class ClassificationFeatureCoalitionGame(_FeatureCoalitionGame):
    def __init__(self, model, dataset_loading_func, class_idx: int) -> None:
        self._class_idx = class_idx
        super().__init__(model, dataset_loading_func)

    @override
    def _predict(self, x: np.ndarray) -> float:
        return self._model.predict_proba(x.reshape(1, -1))[0, self._class_idx]

    @override
    def _eval_model(self, X_test: np.ndarray, y_test: np.ndarray) -> None:
        y_pred = self._model.predict(X_test)

        click.echo(classification_report(y_test, y_pred))


class DiabetesGBRGame(RegressionFeatureCoalitionGame):
    def __init__(self) -> None:
        model = GradientBoostingRegressor(
            n_estimators=50, learning_rate=0.05, max_depth=3, random_state=42
        )
        super().__init__(model, dataset_loading_func=load_diabetes)


class HousingMLPGame(RegressionFeatureCoalitionGame):
    def __init__(self) -> None:
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "mlp",
                    MLPRegressor(
                        hidden_layer_sizes=(64, 32),
                        activation="relu",
                        solver="adam",
                        max_iter=500,
                        random_state=42,
                    ),
                ),
            ]
        )
        super().__init__(model, dataset_loading_func=fetch_california_housing)


class WineRFGame(ClassificationFeatureCoalitionGame):
    def __init__(self) -> None:
        model = RandomForestClassifier(n_estimators=4, random_state=42)
        super().__init__(model, dataset_loading_func=load_wine, class_idx=0)
