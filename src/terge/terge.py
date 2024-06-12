from __future__ import annotations

import re
import warnings

from copy import deepcopy
from typing import Literal

import torch

from tqdm import tqdm


class NoParametersToMergeWarning(UserWarning):
    """A warning raised where there are no parameters to merge."""

class NoMergeMethodFoundForParameterError(ValueError):
    """An error raised when no appropriate merge method can be found for a parameter."""

class ParameterModelWeightsSumToZeroError(ZeroDivisionError):
    """An error raised when models' weights for a parameter sum to zero."""

def arithmetic_merge(parameters: list[torch.nn.Parameter], weights: torch.Tensor) -> torch.Tensor:
    """Merge parameters by taking their arithmetic mean.
    
    Arguments:
        parameters (`list[torch.nn.Parameter]`): The parameters to be merged.
        
        weights (`torch.Tensor`): The weights to be assigned to the parameters.
    
    Returns:
        `torch.Tensor`: The merged parameters."""
    
    return torch.stack([parameter.data * weight for parameter, weight in zip(parameters, weights)]).sum(dim=0)

MERGE_FUNCTIONS = {
    'arithmetic': arithmetic_merge,
}
"""A map of merge methods to their corresponding functions."""

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
) -> torch.nn.Module | tuple[torch.nn.Module, dict[str, tuple[str, list[tuple[int, float]]]]]:
    """Merge PyTorch models.
    
    Arguments:
        models (`list[torch.nn.Module]`): The models to be merged.
        
        base (`torch.nn.Module | int`, optional): The model whose parameters will be used as defaults and that, if `inplace` is set to `True`, will be merged into; or the index of such a model in `models`. Defaults to `0`, that is, the index of the first model in `models`.
        
        method (`Literal['arithmetic']`, optional): The method to be used for merging the models' parameters, or a map of parameter names or regex patterns matching parameter names to the methods to be used to merge them. Currently, only the `'arithmetic'` method is supported (that is, the merging of parameters by taking their ordinary or weighted arithmetic mean). Defaults to `'arithmetic'`.
        
        weights (`list[float] | dict[str | re.Pattern, list[float]]`, optional): A list of all of the relative weights to be assigned to the models' parameters, or a map of parameter names or regex patterns matching parameter names to lists of weights. If `None`, all models will be weighted equally. If a dictionary is provided and there are any parameters to be merged that do not match any of the keys of that dictionary, they will be also weighted equally. Defaults to `None`.
        
        included (`re.Pattern | str | list[str | re.Pattern]`, optional): A regex pattern, string or list of regex patterns and strings matching parameter names to be merged. If `None`, all parameters will be merged. Defaults to `None`.
        
        excluded (`re.Pattern | str | list[str | re.Pattern]`, optional): A regex pattern, string or list of regex patterns and strings matching parameter names to be excluded from merging. If `None`, no parameters will be excluded. If `included` is provided, this argument will apply to the subset of parameters that match `included`. Defaults to `None`.
        
        inplace (`bool`, optional): Whether, for the sake of expediency or memory conservation, the `base` should be merged into in place instead of being deep copied. Defaults to `False`.
        
        dtype (`torch.dtype`, optional): The data type to be used for storing the weightings. Defaults to `torch.float64`.
        
        lineage (`bool`, optional): Whether to output a tuple containing the merged model along with a dictionary mapping the names of merged parameters to a tuple containing the names of merge methods and a list of tuples containing the indices of merged models that contributed to those parameters and the weights they were assigned. Defaults to `False`.
                
        progress (`bool`, optional): Whether to display a progress bar. Defaults to `False`.
    
    Returns:
        `torch.nn.Module | tuple[torch.nn.Module, dict[str, tuple[str, list[tuple[int, float]]]]]`: The merged model, or, if `lineage` is `True`, a tuple containing the merged model along with a dictionary mapping the names of merged parameters to a tuple containing the names of merge methods and a list of tuples containing the indices of merged models that contributed to those parameters and the weights they were assigned."""
    
    # Identify the model to be merged into.
    merged_model = models[base] if isinstance(base, int) else base
    
    # Create a deep copy of the model to be merged into if we are not merging in place.
    if not inplace:
        merged_model = deepcopy(merged_model)
    
    # If the merge method is a map of parameter names to merge methods, convert any string keys into regular expressions, otherwise, identify the appropriate merge function and cache the method.
    if (method_is_a_map := isinstance(method, dict)):
        method = {key if isinstance(key, re.Pattern) else re.compile(f'^{re.escape(key)}$'): value for key, value in method.items()}
    
    else:
        merge_method = method
        merge_function = MERGE_FUNCTIONS[method]
    
    # If the weights are stored as map of parameter names to models' weights, convert any string keys into regular expressions and convert their values into tensors if they are not tensors already.
    # NOTE This block comes first so that `weights_are_a_map` is always assigned.
    if (weights_are_a_map := isinstance(weights, dict)):
        weights = {key if isinstance(key, re.Pattern) else re.compile(f'^{re.escape(key)}$'): (value if isinstance(value, torch.Tensor) else torch.tensor(value, dtype = dtype)) for key, value in weights.items()}
        
        # Cache a uniform set of weights that can be used for parameters that do not match any parameters in the map.
        uniform_weights = torch.ones(len(models), dtype = dtype)
    
    # If weights were not provided, assume that the parameters of all models should be weighted equally when merging.
    elif weights is None:
        weights = torch.ones(len(models), dtype = dtype)
    
    # Convert the weights into a tensor if they are not one already.
    elif not isinstance(weights, torch.Tensor):
        weights = torch.tensor(weights, dtype = dtype)
    
    # If the weights are already a tensor, cast them to the desired data type.
    else:
        weights = weights.to(dtype)
        
    # Identify parameters to be merged.
    parameters = merged_model.named_parameters()
    
    # Add a progress bar if necessary.
    if progress:
        parameters = tqdm(parameters, total = sum(1 for _ in merged_model.parameters()), desc = "Merging the models' parameters", unit = 'parameters')
    
    # Include only parameters that should be merged.
    if included is not None:
        # Convert any parameter inclusion patterns that are not regular expressions into regular expressions.
        if isinstance(included, (str, re.Pattern)):
            included = [included]
        
        included = tuple(parameter if isinstance(parameter, re.Pattern) else re.compile(f'^{re.escape(parameter)}$') for parameter in included)
        
        # Filter for parameters that should be merged.
        parameters = ((name, parameter) for name, parameter in parameters if any(included_parameter.search(name) for included_parameter in included))

    # Exclude parameters from merging that should not be merged.
    if excluded is not None:
        # Convert any parameter exclusion patterns that are not regular expressions into regular expressions.
        if isinstance(excluded, (str, re.Pattern)):
            excluded = [excluded]
        
        excluded = tuple(parameter if isinstance(parameter, re.Pattern) else re.compile(f'^{re.escape(parameter)}$') for parameter in excluded)
        
        # Filter out parameters that should not be merged.
        parameters = ((name, parameter) for name, parameter in parameters if not any(excluded_parameter.search(name) for excluded_parameter in excluded))
    
    # Cache the names of the models' parameters and shapes.
    models_parameters = [{(name, parameter.shape) for name, parameter in model.named_parameters()} for model in models]
    
    # Merge the models' parameters.
    has_merged_at_least_one_parameter = False
    
    if lineage:
        lineage_tree = {}
    
    for parameter_name, parameter in parameters:
        # Identify the appropriate model weights for this parameter.
        if weights_are_a_map:
            for key, parameter_weights in weights.items():
                if key.search(parameter_name):
                    break
            
            else:
                parameter_weights = uniform_weights
        
        else:
            parameter_weights = weights
        
        # Cache this parameter's name and shape.
        parameter_name_and_shape = (parameter_name, parameter.shape)
        
        # Identify models that have a parameter with the same name and shape as this parameter.
        models_with_this_parameter = [i for i, model_parameters in enumerate(models_parameters) if parameter_name_and_shape in model_parameters]
        
        # If there are at least two models with this parameter (ie, it is only the base model that has this parameter), merge them and flag that we have merged at least one parameter.
        if len(models_with_this_parameter) > 1:
            # Identify the appropriate merge function for this parameter.
            if method_is_a_map:
                for key, merge_method in method.items():
                    if key.search(parameter_name):
                        break
                
                else:
                    raise NoMergeMethodFoundForParameterError(f"The map of parameter names to merge methods provided to `merge()` did not contain a matching merge method for the parameter '{parameter_name}'. You are advised to either exclude that parameter via the `included` or `excluded` arguments, or to assign it a merge method.")
                
                merge_function = MERGE_FUNCTIONS[merge_method]
            
            # Drop weights for models without this parameter.
            parameter_weights = parameter_weights[models_with_this_parameter]
            
            # Raise an error if the weights sum to 0.
            parameter_weights_sum = parameter_weights.sum()
            
            if parameter_weights_sum == 0:
                raise ParameterModelWeightsSumToZeroError(f"The model weights for the parameter '{parameter_name}' that were passed to `merge()` sum to 0 (ie, they cancel each other out). Either ensure that the weights never sum to 0 or exclude the parameter via the `included` or `excluded` arguments.")
            
            # Normalize the remaining weights such that they sum to 1 (ie, convert them into a probability distribution).
            parameter_weights /= parameter_weights_sum
            
            # Merge the parameters.
            parameter.data = merge_function((models[i].get_parameter(parameter_name) for i in models_with_this_parameter), parameter_weights)
            
            # Flag that we have merged at least one parameter.
            has_merged_at_least_one_parameter = True
            
            # Preserve the lineage of this parameter if necessary.
            if lineage:
                lineage_tree[parameter_name] = ('arithmetic', [(i, weight.item()) for i, weight in zip(models_with_this_parameter, parameter_weights)])
    
    # Raise a warning if we were unable to merge any parameters.
    if not has_merged_at_least_one_parameter:
        warnings.warn("Of all the models and desired parameters passed to `merge()`, none of them share any mergeable parameters with the same name and shape. This means that the models are either of totally different architectures, are of different sizes, have only unnamed parameters, or that whatever parameters they share were not included in your desired merge parameters. Returning the base model as it is.", NoParametersToMergeWarning)
        
    return merged_model if not lineage else (merged_model, lineage_tree)