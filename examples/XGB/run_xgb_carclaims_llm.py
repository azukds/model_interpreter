"""
End-to-end Integration test: xgb_carclaims model -> ModelInterpreter -> LLM fraud explanations.

Trains the same XGBoost fraud model on the same data during research placement,
fits a ModelInterpreter to it and, for the first N test claims predicted
as fraud, gets the SHAP contributions with ModelInterpreter.transform and
the LLM explanation with ModelInterpreter.explain_with_llm. The results
are saved to a CSV.
 
"""

import argparse
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from model_interpreter.interpreter import ModelInterpreter  # noqa: E402
from model_interpreter.shap_llm import ShapLLMExplainer  # noqa: E402

logger = logging.getLogger(__name__)

# CSV file storing the predictions and later on used to create a html
# to review results qualitatively.
OUTPUT_FILE_NAME = "carclaims_llm_explanations_e2e_v4.csv"

# threhsold for detecting fraud
THRESHOLD = 0.5


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--data",
        required=True,
        help="Fraud claims CSV (carclaims.csv).",
    )
    parser.add_argument(
        "--dataset-stats",
        required=True,
        help="CSV of per-feature dataset statistics.",
    )
    parser.add_argument(
        "--feature-interactions",
        required=True,
        help="CSV of per-feature collinear partners and interaction effects.",
    )
    parser.add_argument(
        "--prompt-template",
        required=True,
        help="Plain-text prompt template with str.format placeholders.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Number of claims predicted as fraud to explain. If None, all samples predicted as fraud are explained",
    )
    parser.add_argument(
        "--top-n-shap",
        type=int,
        default=10,
        help="Top SHAP factors included per claim (default: %(default)s).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="LLM model name (default: shap_llm.MODEL).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not call the LLM; save the prompts that would be sent instead.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=f"CSV the explanations are saved to (default: {OUTPUT_FILE_NAME} next to --data).",
    )
    return parser.parse_args()


def dry_run_client():
    """Client that returns the prompt instead of calling the LLM."""
    return SimpleNamespace(
        responses=SimpleNamespace(
            create=lambda model, input: SimpleNamespace(
                output_text=f"[dry run: no LLM call, prompt for {model} below]\n{input}"
            )
        )
    )


def main() -> int:
    args = parse_args()

    logging.basicConfig(
        level="INFO",
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # set up data and process it

    carclaims_df = pd.read_csv(args.data)

    carclaims_df["FraudFound"] = carclaims_df["FraudFound"].map({"No": 0, "Yes": 1})

    X = carclaims_df.drop(
        columns=[
            "FraudFound", "PolicyNumber", "RepNumber",
            "Deductible", "Month", "WeekOfMonth", "DayOfWeek",
        ]
    )  # fmt: skip
    y = carclaims_df["FraudFound"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    categorical_cols = X_train.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    numerical_cols = X_train.select_dtypes(include=["number"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_cols,
            ),
            ("num", "passthrough", numerical_cols),
        ]
    )

    X_train_encoded = preprocessor.fit_transform(X_train)
    X_test_encoded = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()

    X_train_encoded_df = pd.DataFrame(
        X_train_encoded, columns=feature_names, index=X_train.index
    )
    X_test_encoded_df = pd.DataFrame(
        X_test_encoded, columns=feature_names, index=X_test.index
    )

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = XGBClassifier(
        objective="binary:logistic",
        n_estimators=500,
        learning_rate=0.03,
        max_depth=4,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        X_train_encoded_df,
        y_train,
        eval_set=[(X_test_encoded_df, y_test)],
        verbose=False,
    )

    y_test_proba = pd.Series(
        model.predict_proba(X_test_encoded_df)[:, 1], index=X_test.index
    )
    y_test_pred = (y_test_proba >= THRESHOLD).astype(int)

    print(
        f"\nTest claims: {len(X_test):,} | predicted as fraud: {y_test_pred.sum():,}"
    )

    # fit interpreter to model

    model_interpreter = ModelInterpreter(list(feature_names))
    model_interpreter.fit(model)

    # init LLM explainer
    # only need to create once.

    llm_explainer = ShapLLMExplainer(
        dataset_stats_path=args.dataset_stats,
        feature_interactions_path=args.feature_interactions,
        prompt_template_path=args.prompt_template,
        client=dry_run_client() if args.dry_run else None,
        model=args.model,
        top_n_shap=args.top_n_shap,
    )

    # Explain the first N claims predicted as fraud

    fraud_ids = y_test_pred[y_test_pred == 1].index[: args.n_samples]
    output_rows = []
    n_failed = 0

    for position, claim_id in enumerate(fraud_ids, start=1):
        single_row = X_test_encoded_df.loc[[claim_id]]  # model input (encoded)
        claim_features = X_test.loc[claim_id]  # raw claim fields
        predicted_fraud = int(y_test_pred[claim_id])

        # return feature contribution importances for a single row
        contribution_list = model_interpreter.transform(
            single_row, return_type="name_value_dicts"
        )

        print("\n" + "=" * 78)
        print(
            f"[{position}/{len(fraud_ids)}] claim_id={claim_id} | "
            f"predicted_fraud={predicted_fraud} | "
            f"fraud_probability={y_test_proba[claim_id]:.3f} | "
            f"actual FraudFound={y_test[claim_id]}"
        )
        print("Top 3 contributions:", contribution_list[:3])

        # explain the prediction with the LLM
        try:
            explanation = model_interpreter.explain_with_llm(
                contribution_list,
                claim_features=claim_features,
                predicted_fraud=predicted_fraud,
                llm_explainer=llm_explainer,
            )
        except Exception as e:
            logger.exception("LLM explanation failed for claim %s", claim_id)
            explanation = f"ERROR: {e}"
            n_failed += 1

        print("\nLLM explanation:\n")
        print(explanation)

        output_rows.append(
            {
                "claim_id": claim_id,
                **claim_features,
                "Predicted_FraudFound": predicted_fraud,
                "Fraud_Probability": y_test_proba[claim_id],
                "LLM_Explanation": explanation,
            }
        )


    output_path = args.output or Path(args.data).parent / OUTPUT_FILE_NAME
    pd.DataFrame(output_rows).to_csv(output_path, index=False)

    print("\n" + "=" * 78)
    print(f"Saved {len(output_rows)} explanation(s) to {output_path}")

    if n_failed:
        print(f"{n_failed} of {len(output_rows)} LLM call(s) failed")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
