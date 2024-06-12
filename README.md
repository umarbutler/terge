![terge logo](https://github.com/umarbutler/terge/raw/main/assets/banner.svg)
-----------------------------------------------------------------------------
<a href="https://pypi.org/project/terge/" alt="PyPI Version"><img src="https://img.shields.io/pypi/v/terge"></a> <a href="https://github.com/umarbutler/terge/actions/workflows/ci.yml" alt="Build Status"><img src="https://img.shields.io/github/actions/workflow/status/umarbutler/terge/ci.yml?branch=main"></a> <a href="https://app.codecov.io/gh/umarbutler/terge" alt="Code Coverage"><img src="https://img.shields.io/codecov/c/github/umarbutler/terge"></a> <!-- <a href="https://pypistats.org/packages/terge" alt="Downloads"><img src="https://img.shields.io/pypi/dm/terge"></a> -->

terge is an *easy-to-use* Python library for merging PyTorch models. It works with models of any size and architecture, including Hugging Face 🤗 Transformers.

## Features 🎯
- **👌 Easy-to-use**: a single line of code is all you need to get started.
- **⚡ Lightning-fast**: billions of parameters can be merged in mere seconds.
- **📐 Architecture-agnostic**: models of any size and architecture can be merged, provided they share a couple parameters with the same name and shape.
- **🛠️ Hyper-customizable**: parameters can be filtered in or out with regex, and custom weights can be assigned to models or even to their individual parameters.
- **🌳 Lineage tracking**: maps of merged parameter names to models' weightings can be produced to document precisely how models were merged.
- **🤗 Hugging Face-friendly**: Hugging Face 🤗 Transformers are supported out of the box.

## Installation 🧑‍🔧
`terge` can be installed with `pip`:
```bash
pip install terge
```

## Usage 👩‍💻
The following code snippet demonstrates how you can get started with `terge`:
```python
import re
import torch
import terge

from transformers import AutoModel # NOTE `transformers` isn't required, this is just for demo purposes.

# A single line is all it takes to merge any number of models.
model = terge.merge([torch.nn.Linear(10, 1) for _ in range(3)])

# This also works for models of different architectures...
model = terge.merge([torch.nn.LSTM(10, 1, num_layers = 1), torch.nn.LSTM(10, 1, num_layers = 2)])

# And models of different sizes...
model = terge.merge([torch.nn.LSTM(10, 1, num_layers = 1), torch.nn.LSTM(100, 1, num_layers = 2)])

# And even Hugging Face 🤗 Transformers...
model = terge.merge([AutoModel.from_pretrained('umarbutler/emubert'),
                     AutoModel.from_pretrained('roberta-base')],
                     progress = True)

# Just make sure there's at least one shared named parameter in there.
model = terge.merge([torch.nn.Linear(10, 1), torch.nn.Linear(1, 10)]) # -> terge.NoParametersToMergeWarning
```

If you want even greater control over the merging process, `terge` has got you covered:
```python
# Changing how parameters are merged and what model serves as the base is trivial.
model = terge.merge(
    [torch.nn.Linear(10, 1) for _ in range(3)],
    base = torch.nn.Linear(10, 1), # The base model doesn't even need to be getting merged! You can also
    # use the index of a model in the input models. The default is 0.
    weights = [1, 2, 3], # Weights are relative and correspond to the order of the input models such that,
    # here, the second model is weighted double the weight of the first model and the third model is weighted
    # triple the weight of the first model. The default is [1, 1, ...].
)

# Assigning custom weights to individual parameters is also easy.
model = terge.merge(
    [torch.nn.Linear(10, 1) for _ in range(3)],
    weights = {re.compile(r'weight'): [1, 2, 3], 'bias': [3, 2, 1]}, # Anything that doesn't match this map
    # will get a weight of 1. You can change that adding `re.compile(r'.*'): [...]` to the *end* of your
    # weights map.
)

# If you want to filter specific parameters in or out, that can be done too.
model = terge.merge(
    [torch.nn.Linear(10, 1) for _ in range(3)],
    included = re.compile(r'weight'), # Only parameters with 'weight' in their name will be merged.
    # You could also pass a string for an exact match.
    excluded = ['bias', re.compile(r'bias')], # Lists of strings and regex patterns work as well.
    # NOTE Exclusions execute after inclusions, so this isn't actually necessary.
)

# You can also enable lineage tracking to understand exactly how models got merged.
model, lineage = terge.merge(
    [torch.nn.Linear(10, 1) for _ in range(3)],
    lineage = True,
) # -> {'weight': ('arithmetic', [(0, 0.3333333333333333), (1, 0.3333333333333333), (2, 0.3333333333333333)]),
  #     'bias': ('arithmetic', [(0, 0.3333333333333333), (1, 0.3333333333333333), (2, 0.3333333333333333)])}

# Finally, for an extra speed boost, you can merge in-place (just keep in mind, this will modify your base model).
models = terge.merge(
    [torch.nn.Linear(10, 1) for _ in range(3)],
    inplace = True,
)
```

## API 🧩
### `merge()`
```python
def merge(
    models: list[torch.nn.Module],
    base: torch.nn.Module | int = 0,
    method: Literal['arithmetic'] | dict[str | re.Pattern, Literal['arithmetic']] = 'arithmetic',
    weights: list[float] | dict[str | re.Pattern, list[float]] = None,
    included: re.Pattern | str | list[str | re.Pattern] = None,
    excluded: re.Pattern | str | list[str | re.Pattern] = None,
    inplace: bool = False,
    dtype: torch.dtype = torch.float64,
    lineage: bool = False,
    progress: bool = False,
) -> torch.nn.Module | tuple[torch.nn.Module, dict[str, tuple[str, list[tuple[int, float]]]]]
```

`merge()` merges PyTorch models.

`models` represents the models to be merged.

`base` represents the model whose parameters will be used as defaults and that, if `inplace` is set to `True`, will be merged into; or the index of such a model in `models`. It defaults to `0`, that is, the index of the first model in `models`.

`method` represents the method to be used for merging the models' parameters, or a map of parameter names or regex patterns matching parameter names to the methods to be used to merge them. Currently, only the `'arithmetic'` method is supported (that is, the merging of parameters by taking their ordinary or weighted arithmetic mean). `method` defaults to `'arithmetic'`.

`weights` represents a list of all of the relative weights to be assigned to the models' parameters, or a map of parameter names or regex patterns matching parameter names to lists of weights. If set to `None`, all models will be weighted equally. If a dictionary is provided and there are any parameters to be merged that do not match any of the keys of that dictionary, they will be also weighted equally. `weights` defaults to `None`.

`included` represents a regex pattern, string or list of regex patterns and strings matching parameter names to be merged. If set to `None`, all parameters will be merged. `included` defaults to `None`.

`excluded` represents a regex pattern, string or list of regex patterns and strings matching parameter names to be excluded from merging. If set to `None`, no parameters will be excluded. If `included` is provided, this argument will apply to the subset of parameters that match `included`. `excluded` defaults to `None`.

`inplace` represents whether, for the sake of expediency or memory conservation, the `base` should be merged into in place instead of being deep copied. It defaults to `False`.

`dtype` represents the data type to be used for storing the weightings. It defaults to `torch.float64`.

`lineage` represents whether to output a tuple containing the merged model along with a dictionary mapping the names of merged parameters to a tuple containing the names of merge methods and a list of tuples containing the indices of merged models that contributed to those parameters and the weights they were assigned. It defaults to `False`.

`progress` represents whether to display a progress bar. It defaults to `False`.

`merge()` will return either a merged model, or, if `lineage` is `True`, a tuple containing the merged model along with a dictionary mapping the names of merged parameters to a tuple containing the names of merge methods and a list of tuples containing the indices of merged models that contributed to those parameters and the weights they were assigned, which looks like this:
```python
{
    'parameter_name': ('method', [(model_index, weight), ...]),
    ...
}
```

## Changelog 🔄
terge adheres to [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and [Semantic Versioning](https://semver.org/spec/v2.0.0.html). All notable changes to terge are documented in its [Changelog 🔄](https://github.com/umarbutler/terge/blob/main/CHANGELOG.md).

## License 📜
terge is licensed under the [MIT License](https://github.com/umarbutler/terge/blob/main/LICENSE).