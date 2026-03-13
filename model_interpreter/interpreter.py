import shap  # type: ignore
import logging
import numpy as np  # type: ignore
import pandas as pd
from shap.utils._exceptions import InvalidModelError

from model_interpreter._version import __version__


class ModelInterpreter:
    """
    ModelInterpreter

    Model interpreter, including:
    grouping, conversion and sorting for single prediction contribution

    Parameters:
    ----------
    feature_names: list/tuple
        feature_names of used feature in the model in the same order as when
        passed into the model build

    one_hot_cols: list/tuple, default = None
        identifies fields that have had one hot encoding applied to generate
        model features, if applicable. The one hot encoded feature names are
        automatically derived based on this list, otherwise they are treated
        normally. eg: passing ["colour"] will aggregate the contributions
        of fields with a name beginning with "colour_" and then return the
        aggregated contributions for "colour"

    Examples:
    ---------
    >>> features = ['feat_a', 'feat_b', 'feat_c', 'feat_d']
    >>> mi = ModelInterpreter(features)
    >>> mi.feature_names
    ['feat_a', 'feat_b', 'feat_c', 'feat_d']

    >>> mi = ModelInterpreter(features, one_hot_cols=['feat_a'])
    >>> mi.one_hot_cols
    ['feat_a']
    """

    def __init__(
        self,
        feature_names,
        one_hot_cols=None,
    ):
        self.feature_names = feature_names
        self.one_hot_cols = one_hot_cols
        self.version_ = __version__

    def _tree_explainer_setup(self, model):
        """
        Try and fit a shap.TreeExplainer to a given model

        Parameters:
        -----------
        model: model to be explained

        Returns:
        -----------
        boolean for whether the explainer has been fit successfully or not
        """
        try:
            self.explainer = shap.TreeExplainer(model)
            return True
        except InvalidModelError as e:
            logging.info(
                f"Model is not supported by shap tree explainer, shap.TreeExplainer raised: {e}. Will try to fit linear explainer instead"
            )
            return False

    def _linear_explainer_setup(self, model, X_train):
        """
        Try and fit a shap.LinearExplainer to a given model

        model: model to be explained

        X_train: np.ndarray or pd.DataFrame
            training data required to fit shap.Linear explainer for linear and
            logistic regression models and shap.KernelExplainer for non standard models

        Returns:
        -----------
        boolean for whether the explainer has been fit successfully or not
        """

        if X_train is None:
            raise ValueError("X-train input required to fit linear or kernel explainer")
        if not isinstance(X_train, (np.ndarray, pd.DataFrame)):
            raise TypeError(
                f"X_train should be of type numpy ndarray or pandas DataFrame, instead received type: {type(X_train)}"
            )
        try:
            self.explainer = shap.LinearExplainer(model, X_train)
            return True
        except InvalidModelError as e:
            logging.info(
                f"Model is not supported by shap linear explainer, shap.LinearExplainer raised: {e}. Will try to fit kernel explainer with data sample"
            )

            return False

    def _kernel_explainer_setup(self, model, X_train, is_classification, n_samples):
        """
        Try and fit a shap.KernelExplainer to a given model

        model: model to be explained

        X_train: np.ndarray or pd.DataFrame
            training data required to fit shap.Linear explainer for linear and
            logistic regression models and shap.KernelExplainer for non standard models

        n_samples: int
            the number of samples taken from the data to build the kernel explainer from.
            Default of 50

        is_classification: bool
            True if the model is built for classification, False if built for
            regression

        """

        if is_classification is None:
            raise ValueError(
                f"is_classification input required to fit kernel explainer. True for a classification model, False for a regression model, received {is_classification}"
            )

        elif (is_classification is True) and (not hasattr(model, "predict_proba")):
            raise ValueError("classification model must have a predict_proba method")

        if (is_classification is False) and (not hasattr(model, "predict")):
            raise ValueError("regression model must have a predict method")

        if is_classification is True:
            self.explainer = shap.KernelExplainer(
                model=model.predict_proba, data=shap.sample(X_train, n_samples)
            )

        if is_classification is False:
            self.explainer = shap.KernelExplainer(
                model=model.predict, data=shap.sample(X_train, n_samples)
            )

    def fit(self, model, X_train=None, n_samples=50, is_classification=None):
        """
        Fit explainer for given model.
        shap.TreeExplainer is used for tree-based models
        shap.LinearExplainer is used for linear  models
        shap.KernelExplainer is used for other shap supported models

        Parameters:
        -----------
        model: model to be explained

        X_train: np.ndarray or pd.DataFrame
            training data required to fit shap.Linear explainer for linear and
            logistic regression models and shap.KernelExplainer for non standard models

        n_samples: int
            the number of samples taken from the data to build the kernel explainer from.
            Default of 50

        is_classification: bool
            True if the model is built for classification, False if built for
            regression

        Returns:
        -----------
        model explainer

        Examples:
        ---------
        Fit on a tree-based model here for example/demonstration

        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from sklearn.datasets import make_classification
        >>> from sklearn.model_selection import train_test_split
        >>> X, y = make_classification(
        ...     n_samples=100, n_features=4, n_informative=2,
        ...     n_redundant=0, random_state=0, shuffle=False,
        ... )
        >>> X_train, X_test, y_train, y_test = train_test_split(
        ...     X, y, test_size=0.33, random_state=42,
        ... )
        >>> clf = RandomForestClassifier(max_depth=2, random_state=0, n_estimators=10)
        >>> _ = clf.fit(X_train, y_train)
        >>> mi = ModelInterpreter(['a', 'b', 'c', 'd'])
        >>> explainer = mi.fit(clf)
        >>> type(explainer).__name__
        'TreeExplainer'
        """

        logging.debug(f"Model interpreter version {self.version_} initialised")

        # recursively try to fit tree, linear and kernel shap explainers
        explainer_fitted = self._tree_explainer_setup(model)

        if not explainer_fitted:
            explainer_fitted = self._linear_explainer_setup(model, X_train)

        if not explainer_fitted:
            self._kernel_explainer_setup(model, X_train, is_classification, n_samples)

        return self.explainer

    def _aggregate_ohe_feature_values(self, dict_value, dict_contrib):
        """
        aggregate feature values and contributions by inverse one hot encoding
        the one hot encoded columns defined in init

        Returns:
        ----------
        dict_value_c: dict of feature values

        dict_contrib: dict of feature contributions
        """

        d_value = {}

        # need to loop through dict_value so need to copy it for any
        # extra execution
        dict_value_c = dict_value.copy()

        d_contrib = {}
        for col in self.one_hot_cols:
            contrib = 0
            for k, v in dict_value.items():
                if col in k:
                    if v == 1:
                        val = k.replace(col + "_", "")
                        d_value[col] = val
                    contrib += dict_contrib[k]
                    dict_value_c.pop(k)
                    dict_contrib.pop(k)
            d_contrib[col] = contrib

        for col in self.one_hot_cols:
            if col not in d_value.keys():
                d_value[col] = np.nan
                logging.warning(f"{col} has no value, filled with np.nan")

        dict_value_c.update(d_value)
        dict_contrib.update(d_contrib)

        return dict_value_c, dict_contrib  # values, contrib after inverse ohe

    def _get_single_model_contribution(
        self, single_row, predict_class_index, return_precision
    ):
        """
        get the contribution breakdown of one observation rounded to a specified
        number of decimal places

        Returns:
        -----------
        dict_value: dict of feature values

        dict_contrib: list of feature contributions

        predict_class_index: int, default
                when the return from SHAP is a list each element in the list
                represents a different class. default is 1 for binary classification.
                Only change for multi-class predictions to obtain shap values for
                the defined index class in the list.

        """
        if not isinstance(predict_class_index, int):
            raise ValueError("predicted class not a valid int")

        if isinstance(single_row, pd.DataFrame):
            number_of_features = single_row.shape[1]
            feature_values = single_row.values[0]

        else:
            number_of_features = len(single_row[0])
            feature_values = single_row[0]

        shap_vals = self.explainer.shap_values(single_row)

        # if values are all 0, shap doesn't bother splitting by class which impacts
        # array dimension and our logic, so separate out this case
        if (shap_vals == 0).all():
            shap_values = [0] * number_of_features

        # this condition checks for classification
        elif shap_vals.ndim == 3:
            # this condition hits for multi-classification
            if shap_vals.shape[2] > 2:
                if predict_class_index not in np.arange(
                    shap_vals.shape[2]
                ) or not isinstance(predict_class_index, int):
                    raise ValueError("predicted class not a valid int")

            # this condition hits for binary classification
            elif shap_vals.shape[2] == 2:
                if predict_class_index not in [0, 1]:
                    raise ValueError("predicted class must be 0 or 1 for binary data")

            # index first (only) row, all features, relevant class
            shap_values = shap_vals[0, :, predict_class_index]

        # this condition hits for regression
        else:
            shap_values = shap_vals[0, :]

        if len(self.feature_names) != number_of_features:
            raise ValueError(
                "the number of elements in feature_names should be the same as that in single_row"
            )

        dict_value = dict(zip(self.feature_names, feature_values))
        dict_contrib = dict(zip(self.feature_names, shap_values))

        # cast shap values to python float and round to specified precision
        dict_contrib = {
            feature: round(float(contribution), return_precision)
            for feature, contribution in dict_contrib.items()
        }

        if self.one_hot_cols is not None:
            if not isinstance(self.one_hot_cols, (tuple, list)):
                raise TypeError("one_hot_cols should be a list/tuple")
            dict_value, dict_contrib = self._aggregate_ohe_feature_values(
                dict_value, dict_contrib
            )

        return dict_value, dict_contrib

    @staticmethod
    def _get_grouped_contribution(dict_value, dict_contrib, feature_mappings):
        """
        group features together or rename features

        Returns:
        ----------
        dict_resp_contrib: dict of feature contributions

        Examples:
        ---------
        Rename features with a one-to-one mapping:

        >>> dict_value = {'a': 1.0, 'b': 2.0, 'c': 3.0}
        >>> dict_contrib = {'a': 0.5, 'b': -0.3, 'c': 0.1}
        >>> mapping = {'a': 'Alpha', 'b': 'Beta', 'c': 'Gamma'}
        >>> ModelInterpreter._get_grouped_contribution(
        ...     dict_value, dict_contrib, mapping,
        ... )
        ({'Alpha': 1.0, 'Beta': 2.0, 'Gamma': 3.0}, {'Alpha': 0.5, 'Beta': -0.3, 'Gamma': 0.1})

        Group features together (contributions are summed):

        >>> grouping = {'a': 'G1', 'b': 'G1', 'c': 'G2'}
        >>> ModelInterpreter._get_grouped_contribution(
        ...     dict_value, dict_contrib, grouping,
        ... )
        (None, {'G1': 0.2, 'G2': 0.1})
        """

        if not isinstance(feature_mappings, dict):
            raise TypeError("feature_mappings should be a dictionary")
        for k in dict_value.keys():
            if k not in feature_mappings.keys():
                raise ValueError(f"missing {k} in groups dictionary")

        # if lengths the same, each original feature is mapped to a unique key
        # i.e. 1 to 1 mapping
        if len(feature_mappings) == len(set(feature_mappings.values())):
            dict_resp_contrib = {
                v: dict_contrib[k] for k, v in feature_mappings.items()
            }
            dict_resp_value = {v: dict_value[k] for k, v in feature_mappings.items()}
            return dict_resp_value, dict_resp_contrib

        # otherwise, original features combined and mapped to groups of features
        else:
            # reverse key and value in groups
            rev_groups = {}
            for k, v in feature_mappings.items():
                rev_groups.setdefault(v, []).append(k)

            # sum up contribution together
            dict_resp_contrib = {}
            for k, v in rev_groups.items():
                contrib = 0
                for i in v:
                    contrib += dict_contrib[i]
                dict_resp_contrib[k] = contrib

            return None, dict_resp_contrib

    def transform(
        self,
        single_row,
        sorting="abs",
        feature_mappings=None,
        n_return=None,
        pred_label=None,
        return_feature_values=False,
        return_type="name_value_dicts",
        enable_categorical: bool = False,
        predict_class: int = 1,
        return_precision: int = 16,
    ):
        """
        wrap final response message

        Parameters
        -----------
            single_row: array like
                the observation to be explained

            sorting: str, 'abs'/'label'/'positive', default='abs'
                requirement to sort feature contribution
                - 'abs': sorted with descending absolute contribution
                - 'label': sorted contribution according to label
                        pred_label > 0 - descending order
                        pred_label = 0 ascending order
                                     (not working for negative numbers)
                - 'positive': sorted contribution descending

            feature_mappings: dictionary, default=None
                {feature_name1: group_name1, ...}

            n_return: int, default=None
                number of most significant features to return

            pred_label: bool, default = None
                label value for 'label' sorting option, described above

            return_feature_values: bool, default = False
                flag weather to return feature values in response. These will
                not be returned if return_type is set to 'name_value_dicts'

            return_type: str, default = 'name_value_dicts'
                format for return
                - 'name_value_dicts': list of dictionary pairs containing a dict
                  with Name: feature and Value: contribution. This return_type
                  ignores feature values and does not return them
                           [{"Name": feature, "Value": contribution}, ... ]
                - 'dicts': list of dictionaries
                           [{feature: contribution}, ... ]
                - 'single_dict': single dict of features names and contributions
                        {feature: contribution, ... }
                - 'tuples': list of tuples
                            [(feature, contribution), ... ]

            enable_categorical: bool, default = False
                Flag to represent if the model predicts on categorical fields and must
                be treated as a pd.DataFrame as opposed to a np.array

            predict_class: int, default = 1
                when the return from SHAP is a list each element in the list
                represents a different class. default is 1 for binary classification.
                Only change for multi-class predictions.

            return_precision: int, default = 16
                how many decimal places to round the shap values in the return
                message to. Default is 16 which maintains the decimal places
                returned by shap explainers

        Return:
        -----------
        Features and their contributions in structure according to specified
        return_type

        Examples:
        ---------
        **Binary classification (RandomForest / tree-based):**

        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from sklearn.datasets import make_classification
        >>> from sklearn.model_selection import train_test_split
        >>> X, y = make_classification(
        ...     n_samples=1000, n_features=4, n_informative=2,
        ...     n_redundant=0, random_state=0, shuffle=False,
        ... )
        >>> X_train, X_test, y_train, y_test = train_test_split(
        ...     X, y, test_size=0.33, random_state=42,
        ... )
        >>> clf = RandomForestClassifier(max_depth=2, random_state=0)
        >>> _ = clf.fit(X_train, y_train)
        >>> features = ['feature1', 'feature2', 'feature3', 'feature4']
        >>> mi = ModelInterpreter(features)
        >>> _ = mi.fit(clf)
        >>> single_row = X[0]

        Default return type (name_value_dicts), sorted by absolute contribution:

        >>> mi.transform(single_row, return_precision=4)
        [{'Name': 'feature2', 'Value': -0.3491}, {'Name': 'feature1', 'Value': -0.0039}, {'Name': 'feature4', 'Value': 0.0032}, {'Name': 'feature3', 'Value': 0.0014}]

        Rename features using a one-to-one mapping. You can provide a `feature_mapping` dictionary which can either map feature names to more interpretable names, or group features together:

        >>> mapping = {
        ...     'feature1': 'feature 1 was mapped', 'feature2': 'feature 2 was mapped',
        ...     'feature3': 'feature 3 was mapped', 'feature4': 'feature 4 was mapped',
        ... }
        >>> mi.transform(
        ...     single_row, feature_mappings=mapping,
        ... )
        [{'Name': 'feature 2 was mapped', 'Value': -0.3491295830480707}, {'Name': 'feature 1 was mapped', 'Value': -0.0039231513013799}, {'Name': 'feature 4 was mapped', 'Value': 0.0031653931724603}, {'Name': 'feature 3 was mapped', 'Value': 0.0013787609499949}]

        **Multiclass classification (RandomForest):**

        Use predict_class to select which class to get contributions for.
        To find the class with the highest predicted probability, use
        ``np.argmax`` on ``predict_proba``:

        >>> import numpy as np
        >>> X, y = make_classification(
        ...     n_samples=1000, n_features=4, n_informative=2,
        ...     n_redundant=0, random_state=0, shuffle=False,
        ... )
        >>> y[200:500] = 2
        >>> X_train, X_test, y_train, y_test = train_test_split(
        ...     X, y, test_size=0.33, random_state=42,
        ... )
        >>> clf = RandomForestClassifier(max_depth=2, random_state=0)
        >>> _ = clf.fit(X_train, y_train)
        >>> mi = ModelInterpreter(features)
        >>> _ = mi.fit(clf)
        >>> single_row = X[0]

        Find the class with the highest predicted probability and use it
        as predict_class.

        >>> probs = clf.predict_proba(single_row.reshape(1, -1))
        >>> predict_class = int(clf.classes_[np.argmax(probs)])
        >>> predict_class
        0

        Get contributions for predicted class (0) and other classes individually:

        >>> mi.transform(
        ...     single_row, predict_class=0,
        ...     return_type='single_dict', return_precision=4,
        ... )
        {'feature2': 0.3113, 'feature1': -0.0253, 'feature3': -0.0048, 'feature4': 0.0014}

        >>> mi.transform(
        ...     single_row, predict_class=1,
        ...     return_type='single_dict', return_precision=4,
        ... )
        {'feature2': -0.1429, 'feature1': -0.083, 'feature3': 0.0016, 'feature4': -0.0014}

        >>> mi.transform(
        ...     single_row, predict_class=2,
        ...     return_type='single_dict', return_precision=4,
        ... )
        {'feature2': -0.1684, 'feature1': 0.1082, 'feature3': 0.0032, 'feature4': 0.0}

        **XGBoost regression (tree-based):**

        >>> import xgboost as xgb
        >>> from sklearn.datasets import fetch_california_housing
        >>> X, y = fetch_california_housing(return_X_y=True, as_frame=True)
        >>> X_train, X_test, y_train, y_test = train_test_split(
        ...     X, y, test_size=0.33, random_state=42,
        ... )
        >>> dtrain = xgb.DMatrix(data=X_train, label=y_train)
        >>> feature_names = list(X.columns)
        >>> xgb_model = xgb.train(
        ...     params={"seed": 1, "max_depth": 6, "min_child_weight": 20},
        ...     dtrain=dtrain,
        ... )
        >>> mi = ModelInterpreter(feature_names)
        >>> _ = mi.fit(xgb_model)
        >>> single_row = X_test[feature_names].head(1)
        >>> mi.transform(
        ...     single_row, return_type='single_dict', return_precision=4,
        ... )
        {'MedInc': -0.7147, 'AveOccup': -0.2794, 'Latitude': -0.2189, 'Longitude': -0.1588, 'AveRooms': 0.0442, 'Population': -0.0064, 'HouseAge': -0.0029, 'AveBedrms': -0.0015}

        **Non-standard models (KernelExplainer):**

        For non-standard models that require shap.KernelExplainer, you must
        provide X_train and is_classification when calling fit().

        >>> from sklearn.neighbors import KNeighborsClassifier
        >>> X, y = make_classification(
        ...     n_samples=1000, n_features=4, n_informative=2,
        ...     n_redundant=0, random_state=0, shuffle=False,
        ... )
        >>> X_train, X_test, y_train, y_test = train_test_split(
        ...     X, y, test_size=0.33, random_state=42,
        ... )
        >>> model = KNeighborsClassifier(n_neighbors=5)
        >>> _ = model.fit(X_train, y_train)
        >>> mi = ModelInterpreter(features)
        >>> _ = mi.fit(model, X_train=X_train, is_classification=True)
        >>> single_row = X_test[0].reshape(1, -1)
        >>> mi.transform(
        ...     single_row, return_type='single_dict', return_precision=4,
        ... )
        {'feature2': -0.3783, 'feature3': -0.033, 'feature1': -0.0177, 'feature4': 0.005}
        """

        if sorting not in ["abs", "label", "positive"]:
            raise ValueError("sorting method not supported")

        if return_type not in ["name_value_dicts", "dicts", "tuples", "single_dict"]:
            raise ValueError("return_type not supported")

        if not isinstance(enable_categorical, bool):
            raise TypeError("enable_categorical must be boolean")

        # get single row contributions

        if not enable_categorical:
            single_row = np.array(single_row).reshape(1, -1)

        dict_value, dict_contrib = self._get_single_model_contribution(
            single_row, predict_class, return_precision
        )

        # apply mapping or grouping if applicable
        if feature_mappings:
            # check if mapping is one-to-one, if not return_feature_values must
            # be false as summing values doesn't make sense
            if (
                len(set(feature_mappings.values())) != len(self.feature_names)
                and return_feature_values
            ):
                raise ValueError(
                    "return_feature_values not supported when feature_mappings are not one-to-one i.e. grouping is applied"
                )
            dict_value, dict_contrib = self._get_grouped_contribution(
                dict_value, dict_contrib, feature_mappings
            )

        # create a copy to avoid changing dict_contrib
        dict_contrib_c = dict_contrib.copy()

        # sort contrib list according to specified sorting method
        if sorting == "abs":
            dict_contrib_c = sorted(
                dict_contrib_c.items(), key=lambda x: abs(x[1]), reverse=True
            )
        elif sorting == "label":
            if pred_label is None:
                logging.warning("pred_label not defined, sorting defaulted to negative")
            dict_contrib_c = sorted(
                dict_contrib_c.items(),
                key=lambda x: x[1],
                reverse=bool(pred_label),
            )
        elif sorting == "positive":
            dict_contrib_c = sorted(
                dict_contrib_c.items(), key=lambda x: x[1], reverse=True
            )

        # if return_feature_values not set, we only want to return feature contrib
        if not return_feature_values:
            dict_response = dict(dict_contrib_c)
        # otherwise return feature value and contrib
        else:
            dict_response = {}
            for key, contrib in dict_contrib_c:
                dict_response[key] = (dict_value[key], contrib)

        # if n_return specified, shorten dict
        if n_return:
            dict_response = dict(list(dict_response.items())[0:n_return])

        if return_type == "name_value_dicts":
            if return_feature_values:
                logging.warning(
                    "Feature values not returned when return_type set to name_value_dicts"
                )
            return self._name_value_dicts_return(dict_response)
        if return_type == "dicts":
            return self._dicts_return(dict_response)
        if return_type == "tuples":
            return self._tups_return(dict_response)
        elif return_type == "single_dict":
            return dict_response

    @staticmethod
    def _name_value_dicts_return(dict_response):
        """Swaps single dictionary return to list of dictionaries. Ignores
        feature values if included in the contributions

        Return:
        -----------
        list of dictionaries with structure [{"Name": feature, "Value": contribution}, ... ]

        Examples:
        ---------
        >>> ModelInterpreter._name_value_dicts_return({'a': 0.5, 'b': -0.3})
        [{'Name': 'a', 'Value': 0.5}, {'Name': 'b', 'Value': -0.3}]

        Tuple values (feature_value, contribution) are reduced to contribution only:

        >>> ModelInterpreter._name_value_dicts_return({'a': (1.0, 0.5), 'b': (2.0, -0.3)})
        [{'Name': 'a', 'Value': 0.5}, {'Name': 'b', 'Value': -0.3}]
        """

        dicts = []

        for k, v in dict_response.items():
            # if feature values in contribution, only feature importances are returned
            if isinstance(v, tuple):
                v = v[1]

            dicts.append({"Name": k, "Value": v})

        return dicts

    @staticmethod
    def _dicts_return(dict_response):
        """Swaps single dictionary return to list of dictionaries

        Return:
        -----------
        list of dictionaries with structure [{feature: contribution}, ... ]

        Examples:
        ---------
        >>> ModelInterpreter._dicts_return({'a': 0.5, 'b': -0.3})
        [{'a': 0.5}, {'b': -0.3}]
        """

        dicts = []

        for k, v in dict_response.items():
            dicts.append({k: v})

        return dicts

    @staticmethod
    def _tups_return(dict_response):
        """Swaps single dictionary return to list of tuples

        Return:
        -----------
        list of tuples with structure [(feature: contribution), ... ]

        Examples:
        ---------
        >>> ModelInterpreter._tups_return({'a': 0.5, 'b': -0.3})
        [('a', 0.5), ('b', -0.3)]
        """

        tups = []

        for k, v in dict_response.items():
            tups.append((k, v))

        return tups
