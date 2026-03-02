import pytest  # type: ignore
import sys
from model_interpreter.interpreter import ModelInterpreter  # type: ignore

sys.path.append("../")


class TestFit(object):
    """Tests for ModelInterpreter.fit()."""

    @pytest.mark.parametrize(
        "Model",
        [
            "lin_model",
            "k_model",
            "DBS_model",
            "logic_model",
            "balanced_bagging_classifier",
            "balanced_bagging_regressor",
            "linearSVC_model",
        ],
    )
    def test_exception_model_is_tree_based(
        self, MI: ModelInterpreter, Model, request
    ) -> None:
        """Test to check if .fit() throws and exception if model_type is not tree based"""
        model_type = request.getfixturevalue(Model)
        with pytest.raises(Exception):
            MI.fit(model_type)

    def test_error_X_train_is_required_for_lin(
        self, lin_model, MI: ModelInterpreter
    ) -> None:
        """Test to check ValueError if X_train is not passed for a linear/logistic model"""
        with pytest.raises(
            ValueError,
            match="X-train input required to fit linear or kernel explainer",
        ):
            MI.fit(lin_model)

    @pytest.mark.parametrize(
        "invalid_x_train_type",
        ["String", 1, [0, 1], 0.1, {"a": 1, "b": 2, "c": 1964}],
    )
    def test_error_X_train_type_for_lin(
        self, MI: ModelInterpreter, lin_model, invalid_x_train_type
    ) -> None:
        """Test to check TypeError if X_train is not a  DataFrame or ndarray"""
        with pytest.raises(
            TypeError,
            match=f"X_train should be of type numpy ndarray or pandas DataFrame, instead received type: {type(invalid_x_train_type)}",
        ):
            MI.fit(lin_model, invalid_x_train_type)

    @pytest.mark.parametrize(
        "Model",
        [
            "balanced_bagging_classifier",
            "balanced_bagging_regressor",
        ],
    )
    def test_error_X_train_is_required_for_kernel(
        self, Model, MI: ModelInterpreter
    ) -> None:
        """Test to check ValueError if X_train is not passed for a non standard model"""
        with pytest.raises(
            ValueError,
            match="X-train input required to fit linear or kernel explainer",
        ):
            MI.fit(Model)

    @pytest.mark.parametrize(
        "invalid_x_train_type",
        ["String", 1, [0, 1], 0.1, {"a": 1, "b": 2, "c": 1964}],
    )
    def test_error_X_train_type_for_kernel(
        self, MI: ModelInterpreter, balanced_bagging_classifier, invalid_x_train_type
    ) -> None:
        """Test to check TypeError if X_train is not a  DataFrame or ndarray"""
        with pytest.raises(
            TypeError,
            match=f"X_train should be of type numpy ndarray or pandas DataFrame, instead received type: {type(invalid_x_train_type)}",
        ):
            MI.fit(balanced_bagging_classifier, invalid_x_train_type)

    def test_error_classification_is_required_for_kernel(
        self,
        DBS_model,
        MI: ModelInterpreter,
        kernel_X_train,
    ) -> None:
        """Test to check if ValueError if is_classification: int is not passed for a non-standard model"""
        with pytest.raises(
            ValueError,
            match="is_classification input required to fit kernel explainer. True for a classification model, False for a regression model, received None",
        ):
            MI.fit(DBS_model, X_train=kernel_X_train)

    def test_error_classification_model_type_not_supported(
        self, MI: ModelInterpreter, DBS_model, kernel_X_train
    ) -> None:
        """Test to check ValueError if model is not tree-based or linear/logistic or has predict_proba/predict method"""
        with pytest.raises(
            ValueError,
            match="classification model must have a predict_proba method",
        ):
            MI.fit(DBS_model, kernel_X_train, is_classification=True)
