"""
Generate plain-language LLM explanations for claims predicted as fraud.

Each flagged claim's data and top SHAP factors are combined with dataset
statistics and feature interaction context (CSVs produced by
dataset_knowledge_extractor_with_feature_interactions.py), rendered into
a prompt template, and sent to LLM. Non-fraud claims
are never sent.

Entry points:
    ModelInterpreter.explain_with_llm   explain one prediction
    ShapLLMExplainer.explain            explain one claim from its SHAP values

Model feature names (e.g. ``num__Age``, one-hot ``cat__Make_Honda``) are
mapped back to dataset feature names automatically, or explicitly via
``feature_name_mapping``.

Requires OPENAI_API_KEY; OPENAI_BASE_URL is an optional override.
"""

from __future__ import annotations

import logging
import os
import re
import string
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterable, List, Mapping, Optional, Tuple, Union

import pandas as pd


logger = logging.getLogger(__name__)


# ============================================================
# Default configs - Global constants
# ============================================================


####################### replace openAI config with internal config #######################
MODEL = "gpt-5.6-luna"

API_KEY_ENV_VAR = "OPENAI_API_KEY"

############################################################################################

FRAUD_LABEL = 1

# the placeholder in the prompt saved in .txt file requires these values
REQUIRED_PROMPT_PLACEHOLDERS: FrozenSet[str] = frozenset(
    {
        "dataset_knowledge",
        "feature_interactions",
        "claim_data",
        "prediction",
        "shap_explanation",
        "top_n_shap",
    }
)

REQUIRED_STATS_COLUMNS: FrozenSet[str] = frozenset(
    {
        "Feature",
        "Feature_Type",
        "Correlation_With_Target",
    }
)

COLLINEAR_FEATURE_COLUMN_PATTERN = re.compile(r"^Collinear_Feature_(\d+)$")

# required because the names returned from ColumnTransformer does not match the features names in
# the csv
TRANSFORMER_PREFIX_SEPARATOR = "__"


def read_csv(path: Path, required_columns: FrozenSet[str], name: str) -> pd.DataFrame:
    """Read a CSV file and check that it contains the required columns.

    Args:
        path: Path to the CSV file.
        required_columns: Column names that must be present in the file.
        name: Human-readable name of the file, used in error messages.

    Returns:
        The CSV contents as a DataFrame.

    Raises:
        FileNotFoundError: If ``path`` does not exist or is not a file.
        ValueError: If any of ``required_columns`` is missing.
    """

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{name} file not found: {path}")

    df = pd.read_csv(path)

    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"{name} file {path} is missing columns: {sorted(missing)}")

    return df


def load_prompt_template(path: Path) -> str:
    """Read the prompt template and validate its placeholders.

    Checks that the template's placeholders are exactly the ones this
    module fills in, so a broken template fails at start-up rather than
    on the first API call.

    Args:
        path: Path to the plain-text prompt template, using
            ``str.format`` placeholders.

    Returns:
        The raw template text, ready for ``str.format``.

    Raises:
        FileNotFoundError: If ``path`` does not exist or is not a file.
        ValueError: If the template has unbalanced braces, is missing a
            required placeholder or contains an unknown placeholder.
    """

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Prompt template file not found: {path}")

    template = path.read_text(encoding="utf-8")

    try:
        placeholders = {
            field_name
            for _, field_name, _, _ in string.Formatter().parse(template)
            if field_name is not None
        }
    except ValueError as e:
        raise ValueError(
            f"Prompt template {path} is malformed ({e}). Escape literal "
            f"braces as '{{{{' and '}}}}'."
        ) from e

    missing = REQUIRED_PROMPT_PLACEHOLDERS - placeholders
    unknown = placeholders - REQUIRED_PROMPT_PLACEHOLDERS

    if missing:
        raise ValueError(
            f"Prompt template {path} is missing placeholders: {sorted(missing)}"
        )
    if unknown:
        raise ValueError(
            f"Prompt template {path} has unknown placeholders: {sorted(unknown)}"
        )

    return template


def create_openai_client(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: float = 60.0,
    max_retries: int = 2,
) -> Any:
    """
    replace with internal llm client setup
    """

    from openai import OpenAI

    api_key = api_key or os.getenv(API_KEY_ENV_VAR)
    if not api_key:
        raise ValueError(f"{API_KEY_ENV_VAR} environment variable is not set.")

    return OpenAI(
        api_key=api_key, base_url=base_url, timeout=timeout, max_retries=max_retries
    )



def format_dataset_knowledge_row(row: pd.Series) -> str:
    """Describe one feature's dataset statistics. required to convert
    csv into a text format to insert into the prompt

    Args:
        row: One row of the dataset statistics table. Numeric features
            use the mean/median/spread columns; categorical features use
            the most common and highest/lowest fraud-rate category
            columns.

    Returns:
        A single-line, human-readable summary of the feature.
    """

    if row["Feature_Type"] == "numeric":
        return (
            f"{row['Feature']} (numeric): "
            f"mean={row['Mean']:.2f}, median={row['Median']:.2f}, "
            f"std={row['Std']:.2f}, variance={row['Variance']:.2f}, "
            f"range=[{row['Min']:.2f}, {row['Max']:.2f}]; "
            f"average value is {row['Mean_When_Fraud']:.2f} for "
            f"fraudulent claims vs {row['Mean_When_NonFraud']:.2f} for "
            f"non-fraudulent claims; "
            f"correlation with fraud "
            f"{row['Correlation_With_Target']:.3f}"
        )

    return (
        f"{row['Feature']} (categorical, {int(row['Num_Categories'])} "
        f"categories): most common value is "
        f"'{row['Most_Common_Value']}' "
        f"({row['Most_Common_Frequency']:.1%} of claims); "
        f"category most associated with fraud is "
        f"'{row['Highest_Fraud_Category']}' "
        f"(fraud rate {row['Highest_Fraud_Category_Rate']:.1%}); "
        f"category least associated with fraud is "
        f"'{row['Lowest_Fraud_Category']}' "
        f"(fraud rate {row['Lowest_Fraud_Category_Rate']:.1%}); "
        f"association with fraud "
        f"{row['Correlation_With_Target']:.3f}"
    )


def build_dataset_knowledge_text(dataset_stats_df: pd.DataFrame) -> str:
    """Build the dataset knowledge block of the prompt.

    Args:
        dataset_stats_df: Per-feature dataset statistics, one row per
            feature, in the order the lines should appear.

    Returns:
        One line per feature (see ``format_dataset_knowledge_row``),
        joined by newlines.
    """

    return "\n".join(
        format_dataset_knowledge_row(row) for _, row in dataset_stats_df.iterrows()
    )



def collinear_ranks(feature_interactions_df: pd.DataFrame) -> List[int]:
    """Find the partner ranks present in the feature interactions table.

    Args:
        feature_interactions_df: Feature interactions table with
            ``Collinear_Feature_{rank}`` columns.

    Returns:
        The ranks found in the column names, sorted ascending
        (e.g. ``[1, 2, 3, 4, 5]``).

    Examples:
        >>> collinear_ranks(pd.DataFrame(columns=["Feature", "Collinear_Feature_2", "Collinear_Feature_1"]))
        [1, 2]
    """

    return sorted(
        int(match.group(1))
        for match in (
            COLLINEAR_FEATURE_COLUMN_PATTERN.match(col)
            for col in feature_interactions_df.columns
        )
        if match
    )


def format_interaction_partners(row: pd.Series, ranks: List[int]) -> List[str]:
    """Describe each collinear partner of one feature.

    Args:
        row: One row of the feature interactions table.
        ranks: Partner ranks to describe, in order (see
            ``collinear_ranks``).

    Returns:
        One description per partner, with its association strength and,
        when available, its interaction effect. Ranks with no partner
        are skipped.
    """

    partner_descriptions: List[str] = []

    for rank in ranks:
        partner = row[f"Collinear_Feature_{rank}"]

        if pd.isna(partner):
            continue

        description = (
            f"'{partner}' (association strength "
            f"{row[f'Collinearity_Strength_{rank}']:.3f}"
        )

        interaction_effect = row[f"Interaction_Effect_With_Collinear_Feature_{rank}"]

        if pd.notna(interaction_effect):
            description += (
                f", interaction effect when both elevated = {interaction_effect:+.3f}"
            )

        description += ")"

        partner_descriptions.append(description)

    return partner_descriptions


def build_feature_interactions_text(
    feature_interactions_df: pd.DataFrame, feature_order: pd.Series
) -> str:
    """Build the feature interactions block of the prompt.

    Args:
        feature_interactions_df: Per-feature collinear partners and
            interaction effects, one row per feature.
        feature_order: Feature names in the order the lines should
            appear, so the block lines up with the dataset knowledge
            block.

    Returns:
        One line per feature that has at least one partner, joined by
        newlines.
    """

    ranks = collinear_ranks(feature_interactions_df)

    # Keep the same feature order as the dataset knowledge block, so
    # the two sections are easy to cross-reference.
    ordered_df = (
        feature_interactions_df.set_index("Feature")
        .reindex(feature_order)
        .reset_index()
    )

    lines: List[str] = []

    for _, row in ordered_df.iterrows():
        partner_descriptions = format_interaction_partners(row, ranks)

        if partner_descriptions:
            lines.append(
                f"{row['Feature']}: most closely associated with "
                + ", ".join(partner_descriptions)
            )

    return "\n".join(lines)



def normalise_shap_values(shap_values: Any) -> List[Tuple[str, float]]:
    """Convert SHAP values to a list of ``(feature, shap_value)`` pairs.

    Args:
        shap_values: SHAP values in any of these shapes:

            - ``ModelInterpreter.transform`` output for any
              ``return_type`` ('name_value_dicts', 'dicts', 'tuples',
              'single_dict'), with or without feature values,
            - a ``{feature: shap_value}`` mapping or ``pd.Series``.

    Returns:
        ``(feature, shap_value)`` pairs, in the input order.

    Raises:
        TypeError: If an element has an unsupported shape.

    Examples:
        >>> normalise_shap_values([{"Name": "Age", "Value": 0.2, "FeatureValue": 41}])
        [('Age', 0.2)]
        >>> normalise_shap_values({"Age": (41, 0.2), "Make": -0.1})
        [('Age', 0.2), ('Make', -0.1)]
        >>> normalise_shap_values([("Age", 41, 0.2), ("Make", -0.1)])
        [('Age', 0.2), ('Make', -0.1)]
        >>> normalise_shap_values([{"Age": 0.2}, {"Make": (41, -0.1)}])
        [('Age', 0.2), ('Make', -0.1)]
    """

    pairs: List[Tuple[Any, Any]] = []

    if isinstance(shap_values, (Mapping, pd.Series)):
        pairs.extend(shap_values.items())
    else:
        for item in shap_values:
            if isinstance(item, Mapping) and {"Name", "Value"} <= set(item):
                pairs.append((item["Name"], item["Value"]))
            elif isinstance(item, Mapping):
                pairs.extend(item.items())
            elif isinstance(item, (tuple, list)) and len(item) in (2, 3):
                pairs.append((item[0], item[-1]))
            else:
                raise TypeError(f"Unsupported SHAP value entry: {item!r}")

    # (feature value, shap value) tuples come from return_feature_values=True
    return [
        (str(name), float(value[-1] if isinstance(value, tuple) else value))
        for name, value in pairs
    ]


def resolve_feature_name(
    name: str,
    known_features: Iterable[str],
    feature_name_mapping: Optional[Mapping[str, str]] = None,
) -> Optional[str]:
    """Match a model feature name to a dataset feature name.

    Preferred it as it reads better in the output but not required as you could 
    follow the ColumnTransformer name till the end

    Tried in order:

    1. ``feature_name_mapping[name]``,
    2. an exact match,
    3. the name without a ColumnTransformer prefix (``num__Age`` -> ``Age``),
    4. a one hot encoded column ``<feature>_<category>`` of a known
       feature (``cat__Make_Honda`` -> ``Make``), preferring the longest
       matching feature.

    Args:
        name: Feature name as used by the model.
        known_features: Feature names in the dataset statistics and
            feature interactions CSVs.
        feature_name_mapping: Explicit ``{model_name: dataset_name}``
            mapping, checked first.

    Returns:
        The dataset feature name, or None if no match is found.

    Examples:
        >>> known = ["Age", "Make", "MonthClaimed", "WeekOfMonthClaimed"]
        >>> resolve_feature_name("num__Age", known)
        'Age'
        >>> resolve_feature_name("cat__Make_Honda", known)
        'Make'
        >>> resolve_feature_name("num__WeekOfMonthClaimed", known)
        'WeekOfMonthClaimed'
        >>> resolve_feature_name("Unknown", known) is None
        True
    """

    known = set(known_features)

    if feature_name_mapping and name in feature_name_mapping:
        return feature_name_mapping[name]

    if name in known:
        return name

    _, separator, unprefixed = name.partition(TRANSFORMER_PREFIX_SEPARATOR)
    candidate = unprefixed if separator else name

    if candidate in known:
        return candidate

    owners = [feature for feature in known if candidate.startswith(f"{feature}_")]
    if owners:
        return max(owners, key=len)

    return None


def format_claim_data(claim_features: Any) -> str:
    """Format the claim's features as text.

    Args:
        claim_features: The claim's own fields as a mapping, a
            ``pd.Series`` or a single-row ``pd.DataFrame``, in the order
            they should appear.

    Returns:
        One ``name: value`` line per field, joined by newlines.

    Raises:
        ValueError: If a DataFrame with more than one row is given.

    Examples:
        >>> print(format_claim_data({"Make": "Honda", "Age": 41}))
        Make: Honda
        Age: 41
    """

    if isinstance(claim_features, pd.DataFrame):
        if len(claim_features) != 1:
            raise ValueError(
                f"claim_features must be a single row, got {len(claim_features)} rows"
            )
        claim_features = claim_features.iloc[0]

    return "\n".join(f"{col}: {value}" for col, value in claim_features.items())


def fraud_label(predicted_fraud: Any) -> int:
    """Validate a predicted fraud label.

    Args:
        predicted_fraud: The model's predicted class for the claim,
            ``1`` (fraud) or ``0`` (not fraud), e.g. from
            ``model.predict``.

    Returns:
        The label as an int.

    Raises:
        ValueError: If ``predicted_fraud`` is not 0 or 1 (e.g. a
            probability was passed instead of a class).

    Examples:
        >>> fraud_label(True)
        1
        >>> fraud_label(0.83)
        Traceback (most recent call last):
        ...
        ValueError: predicted_fraud must be the predicted class 0 or 1, got 0.83
    """

    try:
        label = int(predicted_fraud)
    except (TypeError, ValueError):
        label = None

    if label not in (0, FRAUD_LABEL) or label != predicted_fraud:
        raise ValueError(
            f"predicted_fraud must be the predicted class 0 or 1, got {predicted_fraud}"
        )

    return label


# ============================================================
# EXPLAINER
# ============================================================


class ShapLLMExplainer:
    """
    Explain claims predicted as fraud in plain language with an LLM.

    The prompt template, dataset statistics and feature interactions are
    loaded and validated once on construction, so one explainer can be
    created at service start-up and reused for every claim.

    Parameters:
    ----------
    dataset_stats_path: str or Path
        CSV of per-feature dataset statistics.

    feature_interactions_path: str or Path
        CSV of per-feature collinear partners and interaction effects.

    prompt_template_path: str or Path
        Plain-text prompt template with str.format placeholders.

    client: object, default = None
        LLM client exposing ``client.responses.create(model=..., input=...)``
        that returns an object with an ``output_text`` attribute, e.g.
        ``openai.OpenAI`` or ``openai.AzureOpenAI``. If None, a client is
        created with ``create_openai_client`` from environment variables.

    model: str, default = None
        Model name. If None, ``MODEL`` is used.

    top_n_shap: int, default = 10
        Number of SHAP factors included per claim, by absolute value.

    feature_name_mapping: dict, default = None
        Explicit ``{model_feature_name: dataset_feature_name}`` mapping for
        SHAP feature names that are not matched automatically (see
        ``resolve_feature_name``). Several model features mapped to one
        dataset feature have their SHAP values summed.
    """

    def __init__(
        self,
        dataset_stats_path: Union[str, Path],
        feature_interactions_path: Union[str, Path],
        prompt_template_path: Union[str, Path],
        client=None,
        model: Optional[str] = None,
        top_n_shap: int = 10,
        feature_name_mapping: Optional[Mapping[str, str]] = None,
    ):
        if not isinstance(top_n_shap, int) or top_n_shap < 1:
            raise ValueError(f"top_n_shap must be an int >= 1, got {top_n_shap}")

        self.prompt_template = load_prompt_template(Path(prompt_template_path))

        dataset_stats_df = read_csv(
            Path(dataset_stats_path), REQUIRED_STATS_COLUMNS, "Dataset statistics"
        )
        feature_interactions_df = read_csv(
            Path(feature_interactions_path),
            frozenset({"Feature"}),
            "Feature interactions",
        )

        dataset_stats_df = dataset_stats_df.sort_values(
            "Correlation_With_Target", ascending=False
        )

        self.context: Dict[str, Any] = {
            "dataset_knowledge": build_dataset_knowledge_text(dataset_stats_df),
            "feature_interactions": build_feature_interactions_text(
                feature_interactions_df, dataset_stats_df["Feature"]
            ),
            "top_n_shap": top_n_shap,
        }

        self.known_features: FrozenSet[str] = frozenset(
            dataset_stats_df["Feature"]
        ) | frozenset(feature_interactions_df["Feature"])
        self.feature_name_mapping = dict(feature_name_mapping or {})
        self.top_n_shap = top_n_shap
        self.client = client if client is not None else create_openai_client()
        self.model = model or MODEL
        self._warned_features: set = set()

    def dataset_shap_values(self, shap_values: Any) -> List[Tuple[str, float]]:
        """Report SHAP values under the dataset feature names.

        Args:
            shap_values: SHAP values in any shape accepted by
                ``normalise_shap_values``.

        Returns:
            ``(dataset_feature, shap_value)`` pairs in first-seen order,
            with SHAP values of model features that resolve to the same
            dataset feature (e.g. one hot encoded columns) summed. Names
            that cannot be resolved are kept unchanged and logged once.
        """

        aggregated: Dict[str, float] = {}

        for name, value in normalise_shap_values(shap_values):
            feature = resolve_feature_name(
                name, self.known_features, self.feature_name_mapping
            )
            if feature is None:
                if name not in self._warned_features:
                    logger.warning(
                        "SHAP feature %r is not in the dataset statistics or "
                        "feature interactions; it is reported unchanged.",
                        name,
                    )
                    self._warned_features.add(name)
                feature = name

            aggregated[feature] = aggregated.get(feature, 0.0) + value

        return list(aggregated.items())

    def format_claim_texts(
        self, claim_features: Any, shap_values: Any, predicted_fraud: Any
    ) -> Dict[str, str]:
        """Format one claim's data, prediction and top SHAP factors as text.

        Args:
            claim_features: See ``explain``.
            shap_values: See ``explain``.
            predicted_fraud: See ``explain``.

        Returns:
            A dict with the prompt placeholders ``claim_data``,
            ``prediction`` and ``shap_explanation`` mapped to their text.
        """

        prediction_text = f"predicted_fraud: {fraud_label(predicted_fraud)}"

        # Sort by absolute SHAP contribution so the LLM receives the most
        # important factors first, and keep only the top N to focus the
        # LLM on the important factors.
        shap_data: List[Tuple[str, float]] = sorted(
            self.dataset_shap_values(shap_values),
            key=lambda item: abs(item[1]),
            reverse=True,
        )[: self.top_n_shap]

        shap_text = "\n".join(
            f"{feature}: {value:+.6f}" for feature, value in shap_data
        )

        return {
            "claim_data": format_claim_data(claim_features),
            "prediction": prediction_text,
            "shap_explanation": shap_text,
        }

    def build_prompt(
        self, claim_features: Any, shap_values: Any, predicted_fraud: Any
    ) -> str:
        """Render the prompt for one claim.

        Args:
            check explain's docstring for details about the input args 

        Returns:
            The fully rendered prompt.
        """

        return self.prompt_template.format(
            **self.context,
            **self.format_claim_texts(claim_features, shap_values, predicted_fraud),
        )

    def explain(
        self, claim_features: Any, shap_values: Any, predicted_fraud: Any
    ) -> Optional[str]:
        """Generate a plain-language explanation for a claim flagged as fraud.

        Args:
            claim_features: The claim's own fields (e.g. the raw,
                unencoded claim row) as a mapping, ``pd.Series`` or
                single-row ``pd.DataFrame``. All fields are sent to the LLM.
            shap_values: The claim's SHAP values in any shape accepted by
                ``normalise_shap_values``, e.g. ``ModelInterpreter.transform``
                output.
            predicted_fraud: The model's predicted class, 1 (fraud) or 0.

        Returns:
            The LLM's text response, or None if the claim is not
            predicted as fraud, in which case the LLM is not called.

        Raises:
            ValueError: If ``predicted_fraud`` is not 0 or 1.
            Exception: Any error raised by the LLM client (e.g.
                ``openai.OpenAIError``) is propagated to the caller.
        """

        if fraud_label(predicted_fraud) != FRAUD_LABEL:
            return None

        prompt = self.build_prompt(claim_features, shap_values, predicted_fraud)

        response = self.client.responses.create(model=self.model, input=prompt)

        return response.output_text
