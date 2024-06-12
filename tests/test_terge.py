"""Test terge."""
import re
import warnings

from copy import deepcopy

import torch

import terge


class TergeLayer(torch.nn.Module):
    def __init__(self):
        super(TergeLayer, self).__init__()
        
        self.input = torch.nn.Linear(10, 5)
        self.hidden = torch.nn.Linear(5, 5)
        self.output = torch.nn.Linear(5, 1)
        
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.5)

class TergeModel(torch.nn.Module):
    def __init__(self):
        super(TergeModel, self).__init__()
        
        self.input = torch.nn.Linear(500, 50)
        self.hidden = torch.nn.Linear(50, 10)
        self.output = TergeLayer()
        
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.5)

def test_merge():
    """Test `merge()`."""

    n_models = 3
    models = [TergeModel().cuda() for _ in range(n_models)]
    base_index = 1
    base = models[base_index]
    abs_tolerance = 1e-8

    merge_combo_1 = terge.merge(
        models,
        base = base_index,
        method = 'arithmetic',
        weights = None,
        included = ['input.weight', re.compile(r'output')],
        excluded = ['output.output.weight', re.compile('output.input')],
        progress = True,
        lineage = True,
    )

    merge_combo_2 = terge.merge(
        models,
        base = base,
        method = {re.compile(r'.*'): 'arithmetic'},
        weights = [1] * n_models,
        included = re.compile(r'(^input\.weight$)|(output)'),
        excluded = re.compile(r'(^output\.output\.weight$)|(output\.input)'),
        progress = False,
        lineage = False,
    ), merge_combo_1[1]

    for merged_model, lineage in [merge_combo_1, merge_combo_2]:
        # Verify that parameters that should have been excluded were indeed excluded.
        assert base.hidden.weight.data.equal(merged_model.hidden.weight.data) # `base.hidden.weight` should have been excluded by `included`.
        assert base.output.output.weight.data.equal(merged_model.output.output.weight.data) # `base.output.output.weight` should have been excluded by `excluded`.
        assert base.output.input.bias.data.equal(merged_model.output.input.bias.data) # `base.output.input.bias` should have been included by `excluded`.

        # Verify that parameters that should have been merged were indeed merged.
        assert torch.allclose(merged_model.input.weight.data, torch.mean(torch.stack([model.input.weight for model in models]), dim = 0), atol = abs_tolerance) # `base.input.weight` should have been merged.
        assert torch.allclose(merged_model.output.output.bias.data, torch.mean(torch.stack([model.output.output.bias for model in models]), dim = 0), atol = abs_tolerance) # `base.output.output.bias` should have been merged.

        # Verify that the lineage is correct.
        assert {
            param: (
                'arithmetic',
                [
                    (i, float(str(w)[:6])) for i, w in enumerate([1 / n_models] * n_models)
                ]
            )
            
            for param in {'input.weight', 'output.hidden.weight', 'output.hidden.bias', 'output.output.bias'}
        } == {
            param: (
                method,
                [
                    (i, float(str(w)[:6])) for i, w in weights
                ]
            )
            
            for param, (method, weights) in lineage.items()
        }

    # Verify that a warning is raised when there are no parameters to merge but no errors are raised.
    with warnings.catch_warnings(record=True) as w:
        terge.merge([TergeModel(), TergeLayer()])
        assert issubclass(w[-1].category, terge.NoParametersToMergeWarning)

    # Verify that it is possible to weight models differently.
    weights = torch.tensor([i + 1 for i in range(n_models)], dtype = torch.float64)

    merged_model = terge.merge(
        models,
        base = base,
        weights = weights,
        included = 'input.weight',
        excluded = 'output.output.weight',
    )

    weighted_sum = sum(model.input.weight * weight for model, weight in zip(models, weights))
    normalized_weighted_sum = weighted_sum / weights.sum()
    assert torch.allclose(merged_model.input.weight.data, normalized_weighted_sum, atol=abs_tolerance)

    # Verify that it is possible to selectively weight model parameters differently.
    special_weights = weights

    merged_model, lineage = terge.merge(
        models,
        base = base,
        weights = {
            'input.weight': special_weights,
            re.compile(r'output'): list(special_weights),
        },
        lineage=True,
    )

    assert torch.allclose(merged_model.input.weight.data, sum(model.input.weight * weight for model, weight in zip(models, special_weights)) / sum(special_weights), atol=abs_tolerance)
    assert torch.allclose(merged_model.output.hidden.weight.data, sum(model.output.hidden.weight * weight for model, weight in zip(models, special_weights)) / sum(special_weights), atol=abs_tolerance)
    assert (
        'arithmetic',
        [
            (i, float(str(w / sum(special_weights.tolist()))[:6]))
            
            for i, w in enumerate(special_weights.tolist())
        ]
    ) == {
        param: (
            method,
            [
                (i, float(str(w)[:6])) for i, w in weights
            ]
        )
        
        for param, (method, weights) in lineage.items()
    }['input.weight']

    assert (
        'arithmetic',
        [
            (
                i,
                float(str(1 / n_models)[:6])
            )
            
            for i in range(n_models)
        ]
    ) == {
        param: (
            method,
            [
                (i, float(str(w)[:6])) for i, w in weights
            ]
        )
        
        for param, (method, weights) in lineage.items()
    }['hidden.weight']

    # Verify that it is possible to merge models in place.
    inplace_models = deepcopy(models)

    merged_model = terge.merge(
        inplace_models,
        base = 0,
        inplace = True,
    )

    assert inplace_models[0] is merged_model

    # Verify that `NoMergeMethodFoundForParameterError` is raised when no merge method is found for a parameter.
    try:
        terge.merge(
            models,
            method = dict(),
        )

    except terge.NoMergeMethodFoundForParameterError:
        assert True

    else:
        assert False

    # Verify that `ParameterModelWeightsSumToZeroError` is raised when the model weights for a particular parameter sum to zero.
    try:
        terge.merge(
            models,
            weights = [0] * n_models,
        )

    except terge.ParameterModelWeightsSumToZeroError:
        assert True

    else:
        assert False