# Copyright (c) 2022-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#####################################################################################

import logging
from unittest.mock import Mock, patch

import numpy as np
from pytest import fixture, mark, raises
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from groufi.group_permutation import (
    compute_scores_shuffled,
    create_correlated_groups,
    get_score_importances,
    group_permutation_importance,
    iter_shuffled,
    show_correlated_groups,
)


@fixture(scope="module")
def estimator() -> LinearRegression:
    estimator = LinearRegression()
    estimator.fit(np.random.randn(10, 4), np.random.randn(10, 1))
    return estimator


##################################################
#     Tests for group_permutation_importance     #
##################################################


def test_group_permutation_importance(estimator: BaseEstimator):
    df = group_permutation_importance(
        estimator=estimator,
        features=np.random.randn(10, 4),
        feature_names=[f"feat{i}" for i in range(4)],
    )
    assert set(df.index) == {f"feat{i}" for i in range(4)}
    assert set(df.columns) == {"Feature Importance", "group"}
    assert df.shape == (4, 2)


@mark.parametrize("threshold", (0.2, 0.5))
def test_group_permutation_importance_threshold(estimator: BaseEstimator, threshold: float):
    group_mock = Mock(return_value=((0, 2), (1,), (0, 2), (3,)))
    with patch("groufi.group_permutation.create_correlated_groups", group_mock):
        group_permutation_importance(
            estimator=estimator,
            features=np.random.randn(10, 4),
            feature_names=[f"feat{i}" for i in range(4)],
            threshold=threshold,
        )
        assert group_mock.call_args[0][0].shape == (4, 4)  # correlation matrix
        assert group_mock.call_args[1] == {"threshold": threshold}


@mark.parametrize("n_iter", (2, 3))
def test_group_permutation_importance_n_iter(estimator: BaseEstimator, n_iter: int):
    score_mock = Mock(return_value=(1.2, (np.ones(4), np.ones(4))))
    with patch("groufi.group_permutation.get_score_importances", score_mock):
        group_permutation_importance(
            estimator=estimator,
            features=np.random.randn(10, 4),
            feature_names=[f"feat{i}" for i in range(4)],
            n_iter=n_iter,
        )
        assert score_mock.call_args[1]["n_iter"] == n_iter


def test_group_permutation_importance_same_random_state(estimator: BaseEstimator):
    features = np.random.randn(10, 4)
    feature_names = [f"feat{i}" for i in range(4)]
    assert group_permutation_importance(
        estimator=estimator,
        features=features,
        feature_names=feature_names,
        random_state=1,
    ).equals(
        group_permutation_importance(
            estimator=estimator,
            features=features,
            feature_names=feature_names,
            random_state=1,
        )
    )


def test_group_permutation_importance_different_random_states(estimator: BaseEstimator):
    features = np.random.randn(10, 4)
    feature_names = [f"feat{i}" for i in range(4)]
    assert not group_permutation_importance(
        estimator=estimator,
        features=features,
        feature_names=feature_names,
        random_state=1,
    ).equals(
        group_permutation_importance(
            estimator=estimator,
            features=features,
            feature_names=feature_names,
            random_state=2,
        )
    )


##############################################
#     Tests for create_correlated_groups     #
##############################################


def test_create_correlated_groups():
    assert create_correlated_groups(correlation=np.array([[1, 0.3, 0.8], [0.1, 1, 0.1], [0.8, 0.3, 1]])) == (
        (0, 2),
        (1,),
        (0, 2),
    )


def test_create_correlated_groups_independent():
    assert create_correlated_groups(correlation=np.eye(4)) == ((0,), (1,), (2,), (3,))


def test_create_correlated_groups_with_nan():
    assert create_correlated_groups(correlation=np.array([[1, 0.3, np.nan], [0.1, 1, 0.1], [0.8, 0.3, np.nan]])) == (
        (0,),
        (1,),
        (0,),
    )


def test_create_correlated_groups_incorrect_ndim():
    with raises(ValueError):
        create_correlated_groups(correlation=np.zeros(10))


def test_create_correlated_groups_incorrect_shape():
    with raises(ValueError):
        create_correlated_groups(correlation=np.zeros((10, 8)))


############################################
#     Tests for show_correlated_groups     #
############################################


def test_show_correlated_groups(caplog):
    with caplog.at_level(logging.DEBUG):
        show_correlated_groups(((0, 2), (1,), (0, 2), tuple()), ["feature1", "feature2", "feature3", "feature4"])
        assert len(caplog.messages) == 4


def test_show_correlated_groups_incorrect_length():
    with raises(ValueError):
        show_correlated_groups(((0, 2), (1,), (0, 2)), ["feature1", "feature2"])


###################################
#     Tests for iter_shuffled     #
###################################


def test_iter_shuffled():
    arrays = [array.copy() for array in iter_shuffled(np.arange(15).reshape((3, 5)), ((0, 2), (1,)))]
    assert all(array.shape == (3, 5) for array in arrays)


def test_iter_shuffled_mock():
    def shuffle_mock(x: np.array):
        x[:, :] = np.array([[5, 6, 7, 8, 9], [10, 11, 12, 13, 14], [0, 1, 2, 3, 4]])

    rng_mock = Mock(shuffle=shuffle_mock)
    with patch("groufi.group_permutation.check_random_state", Mock(return_value=rng_mock)):
        arrays = [array.copy() for array in iter_shuffled(np.arange(15).reshape((3, 5)), ((0, 2), (1,)))]
        assert np.array_equal(
            arrays[0],
            np.array(
                [
                    [5, 1, 7, 3, 4],
                    [10, 6, 12, 8, 9],
                    [0, 11, 2, 13, 14],
                ]
            ),
        )
        assert np.array_equal(
            arrays[1],
            np.array(
                [
                    [0, 6, 2, 3, 4],
                    [5, 11, 7, 8, 9],
                    [10, 1, 12, 13, 14],
                ]
            ),
        )


def test_iter_shuffled_same_random_state():
    features = np.random.randn(4, 6)
    arrays1 = [array.copy() for array in iter_shuffled(features, ((0, 2), (1,)), random_state=1)]
    arrays2 = [array.copy() for array in iter_shuffled(features, ((0, 2), (1,)), random_state=1)]
    assert all(np.array_equal(arr1, arr2) for arr1, arr2 in zip(arrays1, arrays2))


def test_iter_shuffled_different_random_states():
    features = np.random.randn(4, 6)
    arrays1 = [array.copy() for array in iter_shuffled(features, ((0, 2), (1,)), random_state=1)]
    arrays2 = [array.copy() for array in iter_shuffled(features, ((0, 2), (1,)), random_state=2)]
    assert not all(np.array_equal(arr1, arr2) for arr1, arr2 in zip(arrays1, arrays2))


#############################################
#     Tests for compute_scores_shuffled     #
#############################################


@mark.parametrize("num_groups", (1, 2, 6, 10))
def test_compute_scores_shuffled_num_groups(num_groups: int):
    def score_func(features, y_true):
        return r2_score(y_true, features)

    scores = compute_scores_shuffled(
        score_func=score_func,
        features=np.random.randn(4, 6),
        targets=np.random.randn(4, 6),
        groups=tuple((i % 6,) for i in range(6)),
    )
    assert scores.shape == (6,)


def test_compute_scores_shuffled_same_random_state():
    def score_func(features, y_true):
        return r2_score(y_true, features)

    features = np.random.randn(4, 6)
    targets = np.random.randn(4, 6)
    groups = ((0, 2), (1,))
    assert np.array_equal(
        compute_scores_shuffled(
            score_func=score_func,
            features=features,
            targets=targets,
            groups=groups,
            random_state=1,
        ),
        compute_scores_shuffled(
            score_func=score_func,
            features=features,
            targets=targets,
            groups=groups,
            random_state=1,
        ),
    )


def test_compute_scores_shuffled_different_random_states():
    def score_func(features, y_true):
        return r2_score(y_true, features)

    features = np.random.randn(4, 6)
    targets = np.random.randn(4, 6)
    groups = ((0, 2), (1,))
    assert not np.array_equal(
        compute_scores_shuffled(
            score_func=score_func,
            features=features,
            targets=targets,
            groups=groups,
            random_state=1,
        ),
        compute_scores_shuffled(
            score_func=score_func,
            features=features,
            targets=targets,
            groups=groups,
            random_state=2,
        ),
    )


###########################################
#     Tests for get_score_importances     #
###########################################


@mark.parametrize("n_iter", (1, 2))
def test_get_score_importances_n_iter(n_iter: int):
    def score_func(features, y_true):
        return r2_score(y_true, features)

    base_score, score_decreases = get_score_importances(
        score_func=score_func,
        features=np.random.randn(4, 6),
        targets=np.random.randn(4, 6),
        groups=((0, 2), (1,)),
        n_iter=n_iter,
    )
    assert isinstance(base_score, float)
    assert len(score_decreases) == n_iter
    assert all(array.shape == (2,) for array in score_decreases)


def test_get_score_importances_same_random_state():
    def score_func(features, y_true):
        return r2_score(y_true, features)

    features = np.random.randn(4, 6)
    targets = np.random.randn(4, 6)
    groups = ((0, 2), (1,))
    base_score1, score_decreases1 = get_score_importances(
        score_func=score_func,
        features=features,
        targets=targets,
        groups=groups,
        random_state=1,
    )
    base_score2, score_decreases2 = get_score_importances(
        score_func=score_func,
        features=features,
        targets=targets,
        groups=groups,
        random_state=1,
    )
    assert base_score1 == base_score2
    assert np.array_equal(np.mean(score_decreases1, axis=0), np.mean(score_decreases2, axis=0))


def test_get_score_importances_different_random_states():
    def score_func(features, y_true):
        return r2_score(y_true, features)

    features = np.random.randn(4, 6)
    targets = np.random.randn(4, 6)
    groups = ((0, 2), (1,))
    base_score1, score_decreases1 = get_score_importances(
        score_func=score_func,
        features=features,
        targets=targets,
        groups=groups,
        random_state=1,
    )
    base_score2, score_decreases2 = get_score_importances(
        score_func=score_func,
        features=features,
        targets=targets,
        groups=groups,
        random_state=2,
    )
    assert base_score1 == base_score2
    assert not np.array_equal(np.mean(score_decreases1, axis=0), np.mean(score_decreases2, axis=0))
