MODEL INTERPRETER

----

`model_interpreter` returns feature importance values for a single row model prediction with functionality to:
- handle regression and binary / multiclass classification models
- sort features by importance
- map feature names to more interpretable names
- aggregate feature importance's across features
- handle categorical data within the input features

'model_interpreter' uses [SHAP](https://github.com/shap/shap) to calculate the single row feature importance values.

The package tries to fit one of the three SHAP explainers below in the following order;

- TreeExplainer
- LinearExplainer
- KernelExplainer

Here is a simple example of generating single row feature importance's for a classification model;

```python
from sklearn.ensemble import RandomForestClassifier

from sklearn.datasets import make_classification
from model_interpreter.interpreter import ModelInterpreter

# generate a classification dataset
X, y = make_classification(
    n_samples=1000,
    n_features=4,
    n_informative=2,
    n_redundant=0,
    random_state=0,
    shuffle=False,
)

# fit a model
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X, y)

# fit model interpreter to model
feature_names = ["feature1", "feature2", "feature3", "feature4"]
model_interpreter = ModelInterpreter(feature_names)
model_interpreter.fit(clf)

# return feature contribution importances for a single row
single_row = X.head(1)
contribution_list = single_model_contribution.transform(
    single_row, return_type="name_value_dicts"
)

print(contribution_list)
```

Which will return the following output:
```
[{'Name': 'feature2', 'Value': -0.349129583}, {'Name': 'feature1', 'Value': -0.0039231513}, {'Name': 'feature4', 'Value': 0.0031653932}, {'Name': 'feature3', 'Value': 0.0013787609}]
```


## Installation

The easiest way to get `model_interpreter` is directly from [pypi](https://pypi.org/project/model_interpreter/) with;

 `pip install model_interpreter`


## Examples

To help get started there are example notebooks in the [examples](https://github.com/lvgig/model_interpreter/tree/main/examples) folder in the repo that show how to use each transformer.

To open the example notebooks in [binder](https://mybinder.org/) click [here](https://mybinder.org/v2/gh/lvgig/model_interpreter/HEAD?labpath=examples) or click on the `launch binder` shield above and then click on the directory button in the side bar to the left to navigate to the specific notebook.

## Issues

For bugs and feature requests please open an [issue](https://github.com/lvgig/model_interpreter/issues).

## Build and test

The test framework we are using for this project is [pytest](https://docs.pytest.org/en/stable/). To build the package locally and run the tests follow the steps below.

First clone the repo and move to the root directory;

```shell
git clone https://github.com/lvgig/model_interpreter.git
cd model_interpreter
```

Next install `model_interpreter` and development dependencies;

```shell
pip install . -r requirements-dev.txt
```

Finally run the test suite with `pytest`;

```shell
pytest
```

## Contribute

`model_interpreter` is under active development, we're super excited if you're interested in contributing! 
