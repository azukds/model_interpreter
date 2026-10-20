import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

from model_interpreter import shap_llm
from model_interpreter.interpreter import ModelInterpreter
from model_interpreter.shap_llm import (
    ShapLLMExplainer,
    build_dataset_knowledge_text,
    build_feature_interactions_text,
    load_prompt_template,
    normalise_shap_values,
    resolve_feature_name,
)

PROMPT_TEMPLATE = (
    "STATS:\n{dataset_knowledge}\n"
    "INTERACTIONS:\n{feature_interactions}\n"
    "DATA:\n{claim_data}\n"
    "PREDICTION:\n{prediction}\n"
    "TOP {top_n_shap}:\n{shap_explanation}"
)


class FakeResponses:
    """Records Responses API calls and returns a canned answer."""

    def __init__(self):
        self.calls = []

    def create(self, model, input):
        self.calls.append({"model": model, "input": input})
        return SimpleNamespace(output_text=f"explanation {len(self.calls)}")


def shap_section(prompt: str) -> list:
    """Return the SHAP lines of a prompt rendered with PROMPT_TEMPLATE."""
    return prompt.split(":\n")[-1].split("\n")


@pytest.fixture
def fake_client():
    return SimpleNamespace(responses=FakeResponses())


@pytest.fixture
def dataset_stats():
    return pd.DataFrame(
        {
            "Feature": ["Age", "Make", "MonthClaimed", "WeekOfMonthClaimed"],
            "Feature_Type": ["numeric", "categorical", "categorical", "numeric"],
            "Mean": [40.0, np.nan, np.nan, 2.5],
            "Median": [38.0, np.nan, np.nan, 3.0],
            "Std": [12.0, np.nan, np.nan, 1.0],
            "Variance": [144.0, np.nan, np.nan, 1.0],
            "Min": [18.0, np.nan, np.nan, 1.0],
            "Max": [80.0, np.nan, np.nan, 5.0],
            "Mean_When_Fraud": [35.0, np.nan, np.nan, 2.0],
            "Mean_When_NonFraud": [41.0, np.nan, np.nan, 3.0],
            "Num_Categories": [np.nan, 3.0, 2.0, np.nan],
            "Most_Common_Value": [np.nan, "Honda", "Jan", np.nan],
            "Most_Common_Frequency": [np.nan, 0.5, 0.6, np.nan],
            "Highest_Fraud_Category": [np.nan, "BMW", "Feb", np.nan],
            "Highest_Fraud_Category_Rate": [np.nan, 0.2, 0.1, np.nan],
            "Lowest_Fraud_Category": [np.nan, "Ford", "Jan", np.nan],
            "Lowest_Fraud_Category_Rate": [np.nan, 0.01, 0.05, np.nan],
            "Correlation_With_Target": [-0.1, 0.3, 0.05, 0.01],
        }
    )


@pytest.fixture
def feature_interactions():
    return pd.DataFrame(
        {
            "Feature": ["Age", "Make"],
            "Collinear_Feature_1": ["Make", "Age"],
            "Collinearity_Strength_1": [0.4, 0.4],
            "Interaction_Effect_With_Collinear_Feature_1": [0.05, np.nan],
            "Collinear_Feature_2": [np.nan, np.nan],
            "Collinearity_Strength_2": [np.nan, np.nan],
            "Interaction_Effect_With_Collinear_Feature_2": [np.nan, np.nan],
        }
    )


@pytest.fixture
def input_files(tmp_path: Path, dataset_stats, feature_interactions):
    """Dataset statistics, feature interactions and prompt template files."""
    paths = {
        "dataset_stats_path": tmp_path / "stats.csv",
        "feature_interactions_path": tmp_path / "interactions.csv",
        "prompt_template_path": tmp_path / "prompt.txt",
    }
    dataset_stats.to_csv(paths["dataset_stats_path"], index=False)
    feature_interactions.to_csv(paths["feature_interactions_path"], index=False)
    paths["prompt_template_path"].write_text(PROMPT_TEMPLATE)
    return paths


@pytest.fixture
def explainer(input_files, fake_client):
    return ShapLLMExplainer(**input_files, client=fake_client, model="test-model")


@pytest.fixture
def claim_features():
    return pd.Series({"Make": "BMW", "Age": 41, "MonthClaimed": "Feb"})


@pytest.fixture
def shap_values():
    return {"Age": -0.1, "Make": 0.6, "MonthClaimed": 0.02}


class TestPromptTemplate(object):
    """Tests for loading the prompt template file."""

    def test_missing_template_file(self, tmp_path: Path) -> None:
        """Test FileNotFoundError for a missing template file"""
        with pytest.raises(FileNotFoundError, match="Prompt template file not found"):
            load_prompt_template(tmp_path / "missing.txt")

    @pytest.mark.parametrize(
        "template, match",
        [
            ("{claim_data} {prediction}", "missing placeholders"),
            (PROMPT_TEMPLATE + " {unknown}", "unknown placeholders"),
            (PROMPT_TEMPLATE + " {", "malformed"),
        ],
    )
    def test_invalid_template(self, tmp_path: Path, template: str, match: str) -> None:
        """Test ValueError for invalid templates"""
        path = tmp_path / "prompt.txt"
        path.write_text(template)
        with pytest.raises(ValueError, match=match):
            load_prompt_template(path)

    def test_template_read_unchanged(self, tmp_path: Path) -> None:
        """Test the template file is used exactly as written"""
        path = tmp_path / "prompt.txt"
        path.write_text('PROMPT_TEMPLATE = """\n' + PROMPT_TEMPLATE)
        assert load_prompt_template(path) == 'PROMPT_TEMPLATE = """\n' + PROMPT_TEMPLATE


class TestContextText(object):
    """Tests for the dataset knowledge and feature interactions blocks."""

    def test_dataset_knowledge_text(self, dataset_stats) -> None:
        """Test numeric and categorical features use the original wording"""
        age, make = build_dataset_knowledge_text(dataset_stats.head(2)).split("\n")
        assert age == (
            "Age (numeric): mean=40.00, median=38.00, std=12.00, "
            "variance=144.00, range=[18.00, 80.00]; average value is 35.00 "
            "for fraudulent claims vs 41.00 for non-fraudulent claims; "
            "correlation with fraud -0.100"
        )
        assert make == (
            "Make (categorical, 3 categories): most common value is 'Honda' "
            "(50.0% of claims); category most associated with fraud is 'BMW' "
            "(fraud rate 20.0%); category least associated with fraud is "
            "'Ford' (fraud rate 1.0%); association with fraud 0.300"
        )

    def test_feature_interactions_text(self, feature_interactions) -> None:
        """Test partners are described, empty ranks skipped and order kept"""
        text = build_feature_interactions_text(
            feature_interactions, pd.Series(["Make", "Age"], name="Feature")
        )
        assert text.split("\n") == [
            "Make: most closely associated with 'Age' (association strength 0.400)",
            "Age: most closely associated with 'Make' (association strength 0.400, "
            "interaction effect when both elevated = +0.050)",
        ]


class TestFeatureNames(object):
    """Tests for matching SHAP feature names to dataset feature names."""

    KNOWN = ["Age", "Make", "MonthClaimed", "WeekOfMonthClaimed", "Days:Policy-Claim"]

    @pytest.mark.parametrize(
        "name, expected",
        [
            ("Age", "Age"),
            ("num__Age", "Age"),
            ("cat__Make", "Make"),
            ("cat__Make_Honda", "Make"),
            ("Make_Honda", "Make"),
            ("cat__MonthClaimed_Jan", "MonthClaimed"),
            ("num__WeekOfMonthClaimed", "WeekOfMonthClaimed"),
            ("cat__Days:Policy-Claim_more than 30", "Days:Policy-Claim"),
            ("Unknown", None),
        ],
    )
    def test_resolve_feature_name(self, name, expected) -> None:
        """Test prefixed, one hot encoded and unknown names"""
        assert resolve_feature_name(name, self.KNOWN) == expected

    def test_mapping_takes_precedence(self) -> None:
        """Test an explicit mapping is used before automatic matching"""
        assert resolve_feature_name("Age", self.KNOWN, {"Age": "Make"}) == "Make"
        assert resolve_feature_name("f0", self.KNOWN, {"f0": "Age"}) == "Age"

    def test_one_hot_shap_values_summed(self, explainer) -> None:
        """Test one hot encoded SHAP values are summed per dataset feature"""
        result = explainer.dataset_shap_values(
            {
                "cat__Make_BMW": 0.5,
                "cat__Make_Ford": 0.25,
                "num__Age": -0.1,
                "num__WeekOfMonthClaimed": 0.125,
            }
        )
        assert result == [("Make", 0.75), ("Age", -0.1), ("WeekOfMonthClaimed", 0.125)]

    def test_unknown_feature_kept_and_logged_once(self, explainer, caplog) -> None:
        """Test unresolved names are reported unchanged and warned about once"""
        explainer.dataset_shap_values({"Mystery": 0.1})
        result = explainer.dataset_shap_values({"Mystery": 0.1})
        assert result == [("Mystery", 0.1)]
        assert caplog.text.count("'Mystery' is not in the dataset statistics") == 1

    def test_feature_name_mapping_parameter(self, input_files, fake_client) -> None:
        """Test feature_name_mapping maps and aggregates custom names"""
        explainer = ShapLLMExplainer(
            **input_files,
            client=fake_client,
            feature_name_mapping={"f0": "Make", "f1": "Make", "f2": "Age"},
        )
        assert explainer.dataset_shap_values({"f0": 0.5, "f1": 0.25, "f2": 1.0}) == [
            ("Make", 0.75),
            ("Age", 1.0),
        ]


class TestShapValueShapes(object):
    """Tests for accepting every ModelInterpreter.transform return type."""

    @pytest.mark.parametrize(
        "shap_values",
        [
            {"Age": 0.2, "Make": -0.1},
            {"Age": (41, 0.2), "Make": ("BMW", -0.1)},
            pd.Series({"Age": 0.2, "Make": -0.1}),
            [{"Name": "Age", "Value": 0.2}, {"Name": "Make", "Value": -0.1}],
            [
                {"Name": "Age", "Value": 0.2, "FeatureValue": 41},
                {"Name": "Make", "Value": -0.1, "FeatureValue": "BMW"},
            ],
            [{"Age": 0.2}, {"Make": -0.1}],
            [{"Age": (41, 0.2)}, {"Make": ("BMW", -0.1)}],
            [("Age", 0.2), ("Make", -0.1)],
            [("Age", 41, 0.2), ("Make", "BMW", -0.1)],
        ],
    )
    def test_normalise_shap_values(self, shap_values) -> None:
        """Test every supported shape gives the same pairs"""
        assert normalise_shap_values(shap_values) == [("Age", 0.2), ("Make", -0.1)]

    def test_unsupported_entry(self) -> None:
        """Test TypeError for an unsupported entry"""
        with pytest.raises(TypeError, match="Unsupported SHAP value entry"):
            normalise_shap_values([0.2])


class TestShapLLMExplainer(object):
    """Tests for ShapLLMExplainer."""

    def test_explain_fraud_calls_client(
        self, explainer, fake_client, claim_features, shap_values
    ) -> None:
        """Test the rendered prompt and model are sent to the client for fraud"""
        explainer.top_n_shap = 2

        result = explainer.explain(claim_features, shap_values, predicted_fraud=1)

        assert result == "explanation 1"
        (call,) = fake_client.responses.calls
        assert call["model"] == "test-model"
        prompt = call["input"]
        assert prompt.startswith("STATS:\nMake (categorical, 3 categories)")
        assert "INTERACTIONS:\nMake: most closely associated with 'Age'" in prompt
        assert "DATA:\nMake: BMW\nAge: 41\nMonthClaimed: Feb\n" in prompt
        assert "PREDICTION:\npredicted_fraud: 1\n" in prompt
        assert "TOP 10:\n" in prompt
        assert shap_section(prompt) == ["Make: +0.600000", "Age: -0.100000"]

    @pytest.mark.parametrize("predicted_fraud", [0, np.int64(0), False, 0.0])
    def test_explain_not_fraud_skips_client(
        self, explainer, fake_client, claim_features, shap_values, predicted_fraud
    ) -> None:
        """Test claims not predicted as fraud are never sent to the LLM"""
        assert explainer.explain(claim_features, shap_values, predicted_fraud) is None
        assert fake_client.responses.calls == []

    @pytest.mark.parametrize("predicted_fraud", [np.int64(1), True, 1.0])
    def test_explain_fraud_label_types(
        self, explainer, fake_client, claim_features, shap_values, predicted_fraud
    ) -> None:
        """Test numpy, bool and float fraud labels are accepted"""
        assert explainer.explain(claim_features, shap_values, predicted_fraud)
        assert "predicted_fraud: 1" in fake_client.responses.calls[0]["input"]

    @pytest.mark.parametrize("predicted_fraud", [0.83, 2, -1, "fraud", None])
    def test_explain_invalid_label(
        self, explainer, fake_client, claim_features, shap_values, predicted_fraud
    ) -> None:
        """Test ValueError for a probability or other non 0/1 prediction"""
        with pytest.raises(ValueError, match="predicted class 0 or 1"):
            explainer.explain(claim_features, shap_values, predicted_fraud)
        assert fake_client.responses.calls == []

    def test_claim_features_dataframe(self, explainer, fake_client) -> None:
        """Test a single-row DataFrame is accepted and multiple rows rejected"""
        row = pd.DataFrame([{"Make": "BMW", "Age": 41}])
        explainer.explain(row, {"Age": 0.1}, 1)
        assert "DATA:\nMake: BMW\nAge: 41\n" in fake_client.responses.calls[0]["input"]

        with pytest.raises(ValueError, match="single row"):
            explainer.explain(pd.concat([row, row]), {"Age": 0.1}, 1)

    @pytest.mark.parametrize(
        "missing",
        ["dataset_stats_path", "feature_interactions_path", "prompt_template_path"],
    )
    def test_missing_input_file(self, input_files, fake_client, missing) -> None:
        """Test FileNotFoundError when a required input file is missing"""
        input_files[missing] = input_files[missing].with_name("missing")
        with pytest.raises(FileNotFoundError, match="not found"):
            ShapLLMExplainer(**input_files, client=fake_client)

    def test_missing_stats_columns(
        self, input_files, fake_client, dataset_stats
    ) -> None:
        """Test ValueError when dataset statistics columns are missing"""
        dataset_stats.drop(columns="Correlation_With_Target").to_csv(
            input_files["dataset_stats_path"], index=False
        )
        with pytest.raises(ValueError, match="missing columns"):
            ShapLLMExplainer(**input_files, client=fake_client)

    @pytest.mark.parametrize("top_n_shap", [0, -1, 1.5])
    def test_invalid_top_n_shap(self, input_files, fake_client, top_n_shap) -> None:
        """Test ValueError for invalid top_n_shap"""
        with pytest.raises(ValueError, match="top_n_shap must be an int >= 1"):
            ShapLLMExplainer(**input_files, client=fake_client, top_n_shap=top_n_shap)

    def test_missing_api_key(self, input_files, monkeypatch) -> None:
        """Test ValueError when no API key is available"""
        pytest.importorskip("openai")
        monkeypatch.delenv(shap_llm.API_KEY_ENV_VAR, raising=False)
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            ShapLLMExplainer(**input_files)


class TestModelInterpreterExplainWithLLM(object):
    """Tests for ModelInterpreter.explain_with_llm()."""

    @pytest.fixture
    def fitted(self):
        """One hot encoded model with ColumnTransformer feature names."""
        rng = np.random.default_rng(0)
        raw = pd.DataFrame(
            {
                "Make": rng.choice(["BMW", "Ford", "Honda"], size=300),
                "MonthClaimed": rng.choice(["Jan", "Feb"], size=300),
                "Age": rng.integers(18, 80, size=300),
                "WeekOfMonthClaimed": rng.integers(1, 6, size=300),
            }
        )
        y = ((raw["Make"] == "BMW") & (raw["Age"] < 50)).astype(int)
        preprocessor = ColumnTransformer(
            [
                ("cat", OneHotEncoder(sparse_output=False), ["Make", "MonthClaimed"]),
                ("num", "passthrough", ["Age", "WeekOfMonthClaimed"]),
            ]
        )
        encoded = pd.DataFrame(
            preprocessor.fit_transform(raw),
            columns=preprocessor.get_feature_names_out(),
        )
        model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=0)
        model.fit(encoded, y)
        mi = ModelInterpreter(feature_names=list(encoded.columns))
        mi.fit(model)
        return mi, model, raw, encoded

    def first_row(self, model, encoded, predicted):
        idx = int(np.flatnonzero(model.predict(encoded) == predicted)[0])
        return idx, encoded.iloc[[idx]]

    def test_fraud_prediction(self, fitted, explainer, fake_client) -> None:
        """Test transform output reaches the LLM under dataset feature names"""
        mi, model, raw, encoded = fitted
        idx, single_row = self.first_row(model, encoded, predicted=1)
        contribution_list = mi.transform(single_row, return_type="name_value_dicts")

        explanation = mi.explain_with_llm(
            contribution_list,
            claim_features=raw.iloc[idx],
            predicted_fraud=model.predict(single_row)[0],
            llm_explainer=explainer,
        )

        assert explanation == "explanation 1"

        expected = {}
        for item in contribution_list:
            feature = item["Name"].split("__")[1].split("_")[0]
            expected[feature] = expected.get(feature, 0.0) + item["Value"]

        prompt = fake_client.responses.calls[0]["input"]
        reported = dict(line.split(": ") for line in shap_section(prompt))
        assert set(reported) == {"Make", "MonthClaimed", "Age", "WeekOfMonthClaimed"}
        for feature, value in expected.items():
            assert float(reported[feature]) == pytest.approx(value, abs=1e-6)
        assert f"DATA:\nMake: {raw.iloc[idx]['Make']}\n" in prompt

    def test_same_as_shap_llm(self, fitted, explainer, fake_client) -> None:
        """Test the method gives the same prompt as calling shap_llm directly"""
        mi, model, raw, encoded = fitted
        idx, single_row = self.first_row(model, encoded, predicted=1)
        contribution_list = mi.transform(single_row)

        mi.explain_with_llm(contribution_list, raw.iloc[idx], 1, explainer)
        explainer.explain(raw.iloc[idx], contribution_list, 1)

        first, second = (call["input"] for call in fake_client.responses.calls)
        assert first == second

    def test_non_fraud_prediction(self, fitted, explainer, fake_client) -> None:
        """Test non-fraud predictions return None without calling the LLM"""
        mi, model, raw, encoded = fitted
        idx, single_row = self.first_row(model, encoded, predicted=0)

        explanation = mi.explain_with_llm(
            mi.transform(single_row),
            claim_features=raw.iloc[idx],
            predicted_fraud=model.predict(single_row)[0],
            llm_explainer=explainer,
        )

        assert explanation is None
        assert fake_client.responses.calls == []

    @pytest.mark.parametrize(
        "return_type", ["name_value_dicts", "dicts", "tuples", "single_dict"]
    )
    @pytest.mark.parametrize("return_feature_values", [False, True])
    def test_all_transform_outputs(
        self, fitted, explainer, fake_client, return_type, return_feature_values
    ) -> None:
        """Test every transform return type gives the same prompt"""
        mi, model, raw, encoded = fitted
        idx, single_row = self.first_row(model, encoded, predicted=1)

        mi.explain_with_llm(mi.transform(single_row), raw.iloc[idx], 1, explainer)
        mi.explain_with_llm(
            mi.transform(
                single_row,
                return_type=return_type,
                return_feature_values=return_feature_values,
            ),
            raw.iloc[idx],
            1,
            explainer,
        )

        first, second = (call["input"] for call in fake_client.responses.calls)
        assert first == second

    @pytest.mark.parametrize("llm_explainer", [None, "explainer", object()])
    def test_invalid_llm_explainer(self, fitted, llm_explainer) -> None:
        """Test TypeError if llm_explainer is not a ShapLLMExplainer"""
        mi, model, raw, encoded = fitted
        with pytest.raises(TypeError, match="llm_explainer should be a"):
            mi.explain_with_llm({"num__Age": 0.1}, raw.iloc[0], 1, llm_explainer)

    def test_interpreter_stays_picklable(self, fitted, explainer) -> None:
        """Test the explainer is not stored on the interpreter"""
        mi, model, raw, encoded = fitted
        idx, single_row = self.first_row(model, encoded, predicted=1)
        mi.explain_with_llm(mi.transform(single_row), raw.iloc[idx], 1, explainer)

        assert explainer not in vars(mi).values()
        # round trip of an object created in this test, not untrusted data
        assert pickle.loads(pickle.dumps(mi)).feature_names == mi.feature_names  # noqa: S301
