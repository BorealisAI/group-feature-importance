# Copyright (c) 2022-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#####################################################################################
# Part of this code is based on the eli5 permutation implementation implementation
# from https://github.com/TeamHG-Memex/eli5
#####################################################################################

__all__ = ["group_permutation_importance"]

import logging
from typing import Callable, Iterable, List, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from numpy.random import RandomState
from sklearn.metrics import r2_score
from sklearn.utils import check_random_state

logger = logging.getLogger(__name__)


def group_permutation_importance(
    estimator,
    features: np.ndarray,
    feature_names: Sequence[str],
    threshold: float = 0.75,
    n_iter: int = 20,
    random_state: Union[None, int, RandomState] = None,
) -> pd.DataFrame:
    r"""Computes the permutation feature importance by using group of correlated features.

    Args:
        estimator: Specifies a sklearn estimator with a ``predict`` method.
        features (``numpy.ndarray`` of shape ``(num_examples, feature_size)``): Specifies the matrix of features.
        feature_names: Specifies the names of each feature to make result interpretation easier.
        threshold (float, optional): Specifies the threshold used to create the groups.
            Two features are considered correlated if the correlation value is higher
            than the threshold. Default: ``0.75``
        n_iter (int, optional): Specifies the number of iterations of the basic algorithm.
            Each iteration starting from a different random seed to create the feature permutation.
            Default: ``20``
        random_state (``None``, int or ``RandomState``, optional): see ``sklearn.utils.check_random_state``
            documentation. Default: ``None``

    Returns:
        ``pandas.DataFrame``: A DataFrame with the feature importance of each feature.
            The index of the DataFrame is the feature names sorted by decreasing order of feature importance value.
            The DataFrame has two columns: ``'Feature Importance'`` and ``'group'``.
            ``'Feature Importance'`` contains the estimated feature importance whereas ``'group'``
            contains the correlated features to the current feature.

    Example:

    .. code-block:: python

        >>> import numpy as np
        >>> from groufi import group_permutation_importance
        >>> X_features: np.ndarray = ...  # A matrix of features
        >>> names: Sequence[str] = [...]  # The name of each feature
        >>> model = ...  # a sklearn estimator with a ``predict`` method
        >>> df_gfi = group_permutation_importance(
        ...     estimator=model,
        ...     features=X_features,
        ...     feature_names=names,
        ...     threshold=0.75,
        ...     random_state=42,
        ... )
    """
    logger.debug("Computing the correlation matrix...")
    correlation = np.corrcoef(features, rowvar=False)
    logger.debug(f"Creating the groups of correlated features (threshold={threshold})...")
    groups = create_correlated_groups(correlation, threshold=threshold)
    show_correlated_groups(groups, feature_names)

    def score_func(features: np.ndarray, y_true: np.ndarray):
        return r2_score(y_true, estimator.predict(features))

    base_score, score_decreases = get_score_importances(
        score_func=score_func,
        features=features,
        targets=estimator.predict(features),  # Use the model prediction as ground-truth
        groups=groups,
        n_iter=n_iter,
        random_state=random_state,
    )
    feature_importance = np.mean(score_decreases, axis=0)
    return pd.DataFrame(
        data={"Feature Importance": feature_importance, "group": groups},
        index=feature_names,
    ).sort_values(by=["Feature Importance"], ascending=False)


def create_correlated_groups(correlation: np.ndarray, threshold: float = 0.75) -> Tuple[Tuple[int, ...], ...]:
    r"""Creates the groups of correlated features.

    Note: NaN is interpreted as no correlation between the two variables.

    Args:
        correlation (``numpy.ndarray``): Specifies a correlation matrix.
        threshold (float, optional): Specifies the threshold used to create the groups.
            Two rows (or columns) are considered correlated if the correlation value is higher
            than the threshold. Default: ``0.75``

    Returns:
        tuple: The group of correlated features for each feature. It is represented as a tuple of tuples.
            The outer tuple indicates each feature, and the inner tuples indicates the correlated groups.
            If the output is the variable ``output``, ``output[i]`` indicates the group of features
            correlated  to the ``i``-th feature.``output[i][j]`` is the ``j``-th correlated feature
            to the ``i``-th feature. Note that the current feature is always included in the group
            of correlated features.

    Raises:
        ValueError if ``correlation`` is not a squared matrix.

    Example:

    .. code-block:: python

        >>> from groufi.group_permutation import create_correlated_groups
        >>> correlation = np.array([[1, 0.3, 0.8], [0.1, 1, 0.1], [0.8, 0.3, 1]])
        >>> create_correlated_groups(correlation)
        ((0, 2), (1,), (0, 2))
    """
    if correlation.ndim != 2:
        raise ValueError(f"`correlation` has to be 2 dimensional array (received: {correlation.ndim})")
    if correlation.shape[0] != correlation.shape[1]:
        raise ValueError(f"Incorrect shape. `correlation` has to be a squared matrix (received: {correlation.shape})")
    indices = []
    for i in range(correlation.shape[0]):
        indices.append(tuple(np.flatnonzero(correlation[i] >= threshold).tolist()))
    return tuple(indices)


def show_correlated_groups(groups: Sequence[Sequence[int]], names: Sequence[str]) -> None:
    r"""Shows the correlated groups.

    Args:
        groups: Specifies the groups of correlated features. See the output of ``create_correlated_groups``
            for more information about the structure.
        names: Specifies the names of each feature.

    Raises:
        ValueError if the input lengths do not match.

    Example:

    .. code-block:: python

        >>> show_correlated_groups(((0, 2), (1,), (0, 2)), ['feat1', 'feat2', 'feat3'])
        Group (00) feat1:
            (00) feat1
            (02) feat3
        Group (01) feat2:
            (01) feat2
        Group (02) feat3:
            (00) feat1
            (02) feat3
    """
    if len(groups) != len(names):
        raise ValueError(f"`groups` ({len(groups)}) and `names` ({len(names)}) should have the same length")
    for i, group in enumerate(groups):
        corr_names = "\n".join([f"\t({j:02d}) {names[j]}" for j in group])
        logger.debug(f"Group ({i:02d}) {names[i]}:\n{corr_names}")


def iter_shuffled(
    features: np.ndarray,
    groups: Sequence[Sequence[int]],
    random_state: Union[None, int, RandomState] = None,
) -> Iterable[np.ndarray]:
    """Creates an iterable of matrices which have one or more columns shuffled.

    The columns to shuffle are controlled by the ``groups`` variable.
    After each iteration yielded matrix is mutated inplace, so
    if you want to use multiple of them at the same time, make copies.

    Args:
        features (``numpy.ndarray`` of shape ``(num_examples, feature_size)``): Specifies the matrix of features
            to shuffle.
        groups: Specifies the groups of correlated features. See the output of ``create_correlated_groups``
            for more information about the structure.
        random_state (``None``, int or ``RandomState``, optional): see ``sklearn.utils.check_random_state``
            documentation. Default: ``None``

    Returns:
        Each item in the iterable is a ``numpy.ndarray`` of shape ``(num_examples, feature_size)``.

    Based on ``eli5.permutation_importance.iter_shuffled``
    https://github.com/TeamHG-Memex/eli5/blob/master/eli5/sklearn/permutation_importance.py
    """
    random_state = check_random_state(random_state)
    features_shuffled = features.copy()
    random_state.shuffle(features_shuffled)
    output = features.copy()
    for group in groups:
        output[:, group] = features_shuffled[:, group]
        yield output
        output[:, group] = features[:, group]


def compute_scores_shuffled(
    score_func: Callable[[np.ndarray, np.ndarray], float],
    features: np.ndarray,
    targets: np.ndarray,
    groups: Sequence[Sequence[int]],
    random_state: Union[None, int, RandomState] = None,
) -> np.ndarray:
    r"""Computes the scores associated where the features are shuffled by group.

    Args:
        score_func (callable): Specifies the callable used to compute the score of the predictions done
            with shuffled features.
        features (``numpy.array`` of shape ``(num_examples, feature_size)``): Specifies the matrix of features.
        targets (``numpy.array`` of shape ``(num_examples, prediction_size)``): Specifies the matrix of
            targets.
        groups: Specifies the groups of correlated features. See the output of ``create_correlated_groups``
            for more information about the structure.
        random_state (``None``, int or ``RandomState``, optional): see ``sklearn.utils.check_random_state``
            documentation. Default: ``None``

    Returns:
        ``numpy.array`` of shape ``(feature_size,)``: The score associated to each feature when its
            associated group is shuffled.
    """
    iter_shuffled_features = iter_shuffled(features=features, groups=groups, random_state=random_state)
    return np.array([score_func(shuffled_features, targets) for shuffled_features in iter_shuffled_features])


def get_score_importances(
    score_func: Callable[[np.ndarray, np.ndarray], float],
    features: np.ndarray,
    targets: np.ndarray,
    groups: Sequence[Sequence[int]],
    n_iter: int = 20,
    random_state: Union[None, int, RandomState] = None,
) -> Tuple[float, List[np.ndarray]]:
    """Computes the score importance.

    Args:
        score_func (callable): Specifies the callable used to compute the score of the predictions done
            with shuffled features.
        features (``numpy.array`` of shape ``(num_examples, feature_size)``): Specifies the matrix of features.
        targets (``numpy.array`` of shape ``(num_examples, prediction_size)``): Specifies the matrix of
            targets.
        groups: Specifies the groups of correlated features. See the output of ``create_correlated_groups``
            for more information about the structure.
        n_iter (int, optional): Specifies the number of iterations of the basic algorithm.
            Each iteration starting from a different random seed. Default: ``20``
        random_state (``None``, int or ``RandomState``, optional): see ``sklearn.utils.check_random_state``
            documentation. Default: ``None``

    Returns:
        ``(base_score, score_decreases)`` tuple with the base score and score decreases
            when a feature is not available. ``base_score`` is ``score_func(features, targets)``;
            ``score_decreases`` is a list of length ``n_iter`` with feature importance arrays
            (each array is of shape ``(feature_size,)``); feature importances are computed
            as score decrease when a feature is not available.

    Based on ``eli5.permutation_importance.get_score_importances``
    https://github.com/TeamHG-Memex/eli5/blob/master/eli5/sklearn/permutation_importance.py

    Example:

    .. code-block:: python

        >>> import numpy as np
        >>> from groufi.group_permutation import get_score_importances
        >>> base_score, score_decreases = get_score_importances(score_func, features, targets, groups)
        # If you just want feature importances, you can take a mean of the result:
        >>> feature_importances = np.mean(score_decreases, axis=0)
    """
    random_state = check_random_state(random_state)
    base_score = score_func(features, targets)
    scores_decreases = []
    for _ in range(n_iter):
        scores_shuffled = compute_scores_shuffled(
            score_func=score_func,
            features=features,
            targets=targets,
            groups=groups,
            random_state=random_state,
        )
        scores_decreases.append(-scores_shuffled + base_score)
    return base_score, scores_decreases
