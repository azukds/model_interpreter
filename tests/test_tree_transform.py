import pytest  # type: ignore
from model_interpreter.interpreter import ModelInterpreter  # type: ignore
from hypothesis import given, strategies as st, settings, HealthCheck
import pandas as pd  # type: ignore


class TestTreeTransform(object):
    """Tests for ModelInterpreter.transform() when the model is tree based."""

    def test_tree_model_expected_output_name_value_dicts(
        self,
        MI: ModelInterpreter,
        tree_model,
        single_row: pd.DataFrame,
    ) -> None:
        """Test that transform returns the expected output when return_type = "name_value_dicts" """

        MI.fit(tree_model)
        df_transformed = MI.transform(single_row, return_type="name_value_dicts")
        keys_to_check = {"Name", "Value"}
        check_flag = all(
            isinstance(item, dict) and keys_to_check == set(item.keys())
            for item in df_transformed
        )
        expected = True
        actual = check_flag

        assert expected == actual

    def test_tree_model_expected_output_name_value_dicts_contents(
        self,
        MI: ModelInterpreter,
        tree_model,
        single_row: pd.DataFrame,
    ) -> None:
        """Test that the dictionaries in the output when transform return_type is "name_value_dicts" contain the expected outputs"""

        MI.fit(tree_model)
        df_transformed = MI.transform(single_row, return_type="name_value_dicts")
        check_flag = all(
            isinstance(item["Name"], str) and isinstance(item["Value"], float)
            for item in df_transformed
        )
        expected = True
        actual = check_flag

        assert expected == actual

    def test_tree_model_expected_output_rounded_decimals(
        self,
        MI: ModelInterpreter,
        tree_model,
        single_row: pd.DataFrame,
    ) -> None:
        """Test that the shap value contributions are rounded to the correct precision"""

        MI.fit(tree_model)
        df_transformed = MI.transform(
            single_row, return_type="name_value_dicts", return_precision=10
        )
        check_flag = all(
            round(item["Value"], 10) == item["Value"] for item in df_transformed
        )
        expected = True
        actual = check_flag

        assert expected == actual

    def test_tree_model_expected_output_tuples(
        self,
        MI: ModelInterpreter,
        tree_model,
        single_row: pd.DataFrame,
    ) -> None:
        """Test that transform returns the expected output when return_type = "tuples" """

        MI.fit(tree_model)
        df_transformed = MI.transform(single_row, return_type="tuples")
        check_flag = all(isinstance(item, tuple) for item in df_transformed)
        expected = True
        actual = check_flag

        assert expected == actual

    def test_tree_model_expected_output_dicts_(
        self,
        MI: ModelInterpreter,
        tree_model,
        single_row: pd.DataFrame,
    ) -> None:
        """Test that transform returns the expected output when return_type = "dicts" """
        MI.fit(tree_model)
        df_transformed = MI.transform(single_row, return_type="dicts")
        check_flag = all(isinstance(item, dict) for item in df_transformed)
        expected = True
        actual = check_flag

        assert expected == actual

    def test_tree_model_expected_output_single_dict(
        self,
        MI: ModelInterpreter,
        tree_model,
        single_row: pd.DataFrame,
    ) -> None:
        """Test that transform returns the expected output when return_type = "single_dict" """
        MI.fit(tree_model)
        df_transformed = MI.transform(single_row, return_type="single_dict")
        check_flag = isinstance(df_transformed, dict)
        expected = True
        actual = check_flag

        assert expected == actual

    # sorting types
    def test_tree_model_expected_output_abs(
        self,
        MI: ModelInterpreter,
        tree_model,
        single_row: pd.DataFrame,
    ) -> None:
        """Test that transform returns the expected output when sorting = "abs" """
        MI.fit(tree_model)
        df_transformed = MI.transform(single_row, sorting="abs", return_type="dicts")
        newlist = sorted(
            df_transformed,
            key=lambda df_transformed: list(map(abs, df_transformed.values())),
            reverse=True,
        )
        expected = df_transformed
        actual = newlist

        assert expected == actual

    def test_tree_model_expected_output_label_1(
        self,
        MI: ModelInterpreter,
        tree_model,
        single_row: pd.DataFrame,
    ) -> None:
        """Test that transform returns the expected output when sorting = "label" and
        pred_label = 1
        """
        MI.fit(tree_model)
        df_transformed = MI.transform(
            single_row, sorting="label", pred_label=1, return_type="dicts"
        )

        newlist = sorted(
            df_transformed,
            key=lambda df_transformed: list(df_transformed.values()),
            reverse=True,
        )
        expected = df_transformed
        actual = newlist

        assert expected == actual

    def test_tree_model_expected_output_label_0(
        self,
        MI: ModelInterpreter,
        tree_model,
        single_row: pd.DataFrame,
    ) -> None:
        """Test that transform returns the expected output when sorting = "label" and
        pred_label = 0
        """
        MI.fit(tree_model)
        df_transformed = MI.transform(
            single_row, sorting="label", pred_label=0, return_type="dicts"
        )

        newlist = sorted(
            df_transformed,
            key=lambda df_transformed: list(df_transformed.values()),
            reverse=False,
        )

        expected = df_transformed
        actual = newlist

        assert expected == actual

    # n_return
    @given(s=st.integers().filter(lambda x: x > 0))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_tree_model_expected_output_n_return(
        self,
        MI: ModelInterpreter,
        tree_model,
        features: list,
        s: int,
        single_row: pd.DataFrame,
    ) -> None:
        """Test that transform returns the expected output when n_return = any integer>0"""
        MI.fit(tree_model)
        df_transformed = MI.transform(single_row, n_return=s, return_type="dicts")

        if 0 < s <= len(features):
            expected = len(df_transformed)
            actual = s
        else:
            expected = len(df_transformed)
            actual = len(features)

        assert expected == actual

    @given(s=st.booleans())
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_tree_model_expected_output_return_feature_values(
        self,
        MI: ModelInterpreter,
        tree_model,
        s: int,
        single_row: pd.DataFrame,
    ) -> None:
        """Test that transform returns the expected output when return_feature_values = True or False"""
        MI.fit(tree_model)
        df_transformed = MI.transform(
            single_row, return_feature_values=s, return_type="dicts"
        )

        if not s:
            count = len(df_transformed[1].values())
            expected = count
            actual = 1
        else:
            count = len(list(df_transformed[1].values())[0])
            expected = count
            actual = 2

        assert expected == actual

    # feature_mappings
    def test_tree_model_expected_output_return_feature_mappings(
        self,
        MI: ModelInterpreter,
        tree_model,
        single_row: pd.DataFrame,
    ) -> None:
        """Test that transform returns the expected output when feature_mappings = mapping_dict"""
        mapping_dict = {
            "MedInc": "MedInc feature",
            "HouseAge": "Age of house",
            "AveRooms": "Average number of rooms",
            "AveBedrms": "Average number of bedrooms",
            "Population": "Population feature",
            "AveOccup": "Average occupation",
        }
        MI.fit(tree_model)
        df_transformed = MI.transform(
            single_row, feature_mappings=mapping_dict, return_type="dicts"
        )

        all_keys = set().union(
            *(df_transformed.keys() for df_transformed in df_transformed)
        )

        check_flag = set(list(all_keys)) == set(list(mapping_dict.values()))
        expected = True
        actual = check_flag

        assert expected == actual

    def test_tree_model_expected_output_return_group_mappings(
        self,
        MI: ModelInterpreter,
        tree_model,
        single_row: pd.DataFrame,
    ) -> None:
        """Test that transform returns the expected output when feature_mappings = grouping_dict"""
        grouping_dict = {
            "MedInc": "MedInc feature",
            "HouseAge": "Age of house",
            "AveRooms": "Number of rooms",
            "AveBedrms": "Number of rooms",
            "Population": "Population feature",
            "AveOccup": "Average occupation",
        }
        MI.fit(tree_model)
        df_transformed = MI.transform(
            single_row, feature_mappings=grouping_dict, return_type="dicts"
        )

        all_keys = set().union(
            *(df_transformed.keys() for df_transformed in df_transformed)
        )

        check_flag = set(list(all_keys)) == set(list(grouping_dict.values()))
        expected = True
        actual = check_flag

        assert expected == actual

    def test_tree_model_name_value_dicts_include_feature_values_when_requested(
        self,
        MI: ModelInterpreter,
        tree_model,
        single_row: pd.DataFrame,
    ) -> None:
        """Test that feature values are included in name_value_dicts output when requested."""

        MI.fit(tree_model)
        df_transformed = MI.transform(
            single_row,
            return_type="name_value_dicts",
            return_feature_values=True,
        )

        med_inc_item = next(item for item in df_transformed if item["Name"] == "MedInc")
        assert med_inc_item["FeatureValue"] == single_row.iloc[0]["MedInc"]

    def test_tree_model_name_value_dicts_default_output_omits_feature_values(
        self,
        MI: ModelInterpreter,
        tree_model,
        single_row: pd.DataFrame,
    ) -> None:
        """Test that default name_value_dicts output remains unchanged unless requested."""

        MI.fit(tree_model)
        df_transformed = MI.transform(single_row, return_type="name_value_dicts")
        assert all("FeatureValue" not in item for item in df_transformed)

    def test_tree_model_name_value_dicts_include_aggregated_one_hot_feature_values(
        self,
        tree_model_one_hot,
        one_hot_tree_X_test: pd.Series,
        one_hot_features: list,
    ) -> None:
        """Test that one-hot encoded features expose the aggregated human-readable value when requested."""

        x = ModelInterpreter(feature_names=one_hot_features, one_hot_cols=["Over25"])
        x.fit(tree_model_one_hot)
        single_row = one_hot_tree_X_test[one_hot_features].head(1)
        df_transformed = x.transform(
            single_row,
            return_feature_values=True,
            return_type="name_value_dicts",
        )

        over25_item = next(item for item in df_transformed if item["Name"] == "Over25")
        expected_value = (
            "Over25"
            if single_row.iloc[0]["Over25_Over25"] == 1
            else "Under25"
            if single_row.iloc[0]["Over25_Under25"] == 1
            else None
        )
        assert over25_item["FeatureValue"] == expected_value

    def test_tree_model_error_sorting_input_not_supported(
        self,
        MI: ModelInterpreter,
        tree_model,
        single_row: pd.DataFrame,
    ) -> None:
        """Test to check ValueError if x.transform() sorting method is not supported"""
        with pytest.raises(
            ValueError,
            match="sorting method not supported",
        ):
            MI.fit(tree_model)
            MI.transform(single_row, sorting="XYZ", return_type="dicts")

    def test_tree_model_error_return_type_not_supported(
        self,
        MI: ModelInterpreter,
        tree_model,
        single_row: pd.DataFrame,
    ) -> None:
        """Test to check ValueError if x.transform() return_type method is not supported"""
        with pytest.raises(
            ValueError,
            match="return_type not supported",
        ):
            MI.fit(tree_model)
            MI.transform(single_row, return_type="XYZ")

    @pytest.mark.parametrize(
        "invalid_enable_categorical_type",
        ["String", 123, [0, 1], 0.1, {"a": 1, "b": 2, "c": 1964}],
    )
    def test_error_enable_categorical_type(
        self, MI: ModelInterpreter, tree_model, invalid_enable_categorical_type
    ) -> None:
        """Test to check TypeError if enable_categorical is not bool"""
        with pytest.raises(
            TypeError,
            match="enable_categorical must be boolean",
        ):
            MI.transform(tree_model, enable_categorical=invalid_enable_categorical_type)

    @pytest.mark.parametrize(
        "invalid_predict_class",
        ["String", [0, 1], (1, 2, 4), 4, -1, {"a": 1, "b": 2, "c": 1964}],
    )
    def test_error_predict_class(
        self,
        MI: ModelInterpreter,
        tree_model_RF,
        invalid_predict_class,
        single_row: pd.DataFrame,
    ) -> None:
        """Test to check TypeError if predict_class is not bool"""
        with pytest.raises(
            ValueError,
            match="predicted class not a valid int",
        ):
            MI.fit(tree_model_RF)
            MI.transform(single_row, predict_class=invalid_predict_class)

    def test_tree_model_zero_class_non_list(
        self,
        MI: ModelInterpreter,
        tree_model,
        single_row: pd.DataFrame,
    ) -> None:
        """Test to check if binary classification for 0 and 1 are opposite"""
        MI.fit(tree_model)
        df_transformed_0 = MI.transform(
            single_row, predict_class=0, return_type="dicts"
        )
        df_transformed_1 = MI.transform(
            single_row, predict_class=1, return_type="dicts"
        )

        excepted = df_transformed_0
        actual = df_transformed_1

        for key in actual[0]:
            assert round(actual[0][key] * -1, 5) == round(excepted[0][key], 5)

    def test_tree_model_zero_class_list(
        self,
        MI: ModelInterpreter,
        tree_model_RF_binary,
        single_row: pd.DataFrame,
    ) -> None:
        """Test to check if binary classification for 0 and 1 are opposite"""
        MI.fit(tree_model_RF_binary)
        df_transformed_0 = MI.transform(
            single_row, predict_class=0, return_type="dicts"
        )
        df_transformed_1 = MI.transform(
            single_row, predict_class=1, return_type="dicts"
        )

        excepted = df_transformed_0
        actual = df_transformed_1

        for key in actual[0]:
            assert round(actual[0][key] * -1, 5) == round(excepted[0][key], 5)
