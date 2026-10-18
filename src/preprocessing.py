
"""Preprocessing pipelines for the METABRIC survival project.

Every step (imputation, scaling, encoding, rare-mutation filter) is fitted
on training data only, because it lives inside a scikit-learn pipeline.
"""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import VarianceThreshold


# Remove mutation genes that are too rare in the training data
MIN_MUT_RATE = 0.01
MUT_VARIANCE = MIN_MUT_RATE * (1 - MIN_MUT_RATE)

# Categories with fewer than 10 training examples can be grouped together
MIN_CATEGORY_COUNT = 10


def _numeric():
    """Fill missing numbers and scale numerical features."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])


def _numeric_with_flag():
    """Fill missing numbers and add an indicator for missing values."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])


def _categorical():
    """Fill missing categories and convert them into numerical columns."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=MIN_CATEGORY_COUNT,
            sparse_output=False
        )),
    ])


def _mutation():
    """Remove mutation columns with very low variance."""
    return Pipeline([
        ("rare_filter", VarianceThreshold(threshold=MUT_VARIANCE)),
    ])


def build_preprocessor(feature_set, spec):
    """
    Build a preprocessing pipeline for clinical, molecular or combined data.
    The feature groups come from feature_spec.json.
    """
    parts = []

    if feature_set in ("clinical", "combined"):
        parts.append(("clin_num", _numeric(), spec["clin_numeric"]))
        parts.append((
            "clin_num_flag",
            _numeric_with_flag(),
            spec["clin_numeric_flag"]
        ))
        parts.append((
            "clin_cat",
            _categorical(),
            spec["clin_categorical"]
        ))

    if feature_set in ("molecular", "combined"):
        parts.append(("mol_num", _numeric(), spec["mol_numeric"]))
        parts.append((
            "mol_cat",
            _categorical(),
            spec["mol_categorical"]
        ))
        parts.append(("mut", _mutation(), spec["mut"]))

    if not parts:
        raise ValueError(
            "feature_set must be clinical, molecular or combined"
        )

    preprocessor = ColumnTransformer(
        parts,
        remainder="drop",
        verbose_feature_names_out=True
    )

    preprocessor.set_output(transform="pandas")

    return preprocessor
