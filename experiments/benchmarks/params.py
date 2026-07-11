"""Parameter-dict plumbing shared by the benchmarkables.

Ported from the ``treeflow-benchmarks`` repository's ``treeflow_benchmarks/params.py``
(generic tensor/param handling, no BEAST dependency).
"""
import typing as tp

import tensorflow as tf

from treeflow.model.phylo_model import (
    JC_KEY,
    PhyloModel,
    get_params,
    get_subst_model_params,
)


def get_return_value_of_empty_generator(gen: tp.Generator[object, object, object]) -> object:
    try:
        next(iter(gen))
    except StopIteration as ex:
        return ex.value
    raise ValueError("Generator was not empty")


def get_tensor_params_dict(phylo_model: PhyloModel):
    subst_model_params, _ = get_return_value_of_empty_generator(
        get_subst_model_params(phylo_model.subst_model, phylo_model.subst_params)
    )
    site_model_params, _ = get_return_value_of_empty_generator(
        get_params(phylo_model.site_params)
    )
    clock_model_params, _ = get_return_value_of_empty_generator(
        get_params(phylo_model.clock_params)
    )
    return dict(
        subst_model_params=subst_model_params,
        site_model_params=site_model_params,
        clock_model_params=clock_model_params,
    )


def move_entry(from_dict, to_dict, section, key):
    to_dict[section][key] = from_dict[section].pop(key)


def split_gradient_params(phylo_model: PhyloModel, calculate_clock_rate_gradient=False):
    gradient_params = get_tensor_params_dict(phylo_model)
    non_gradient_params = dict(
        subst_model_params=dict(), site_model_params=dict(), clock_model_params=dict()
    )
    if phylo_model.subst_model == JC_KEY:
        move_entry(gradient_params, non_gradient_params, "subst_model_params", "frequencies")
    if phylo_model.site_model != "none":
        move_entry(gradient_params, non_gradient_params, "site_model_params", "category_count")
    if (not calculate_clock_rate_gradient) and phylo_model.clock_model == "strict":
        move_entry(gradient_params, non_gradient_params, "clock_model_params", "clock_rate")
    return gradient_params, non_gradient_params


def merge_entries(x, y, key):
    return {**x[key], **y[key]}


def merge_params(gradient_params, non_gradient_params):
    return {
        key: merge_entries(gradient_params, non_gradient_params, key)
        for key in gradient_params.keys()
    }


def get_numpy_gradient_params_dict(phylo_model: PhyloModel, calculate_clock_rate_gradient=False):
    tensor_dict, _ = split_gradient_params(
        phylo_model, calculate_clock_rate_gradient=calculate_clock_rate_gradient
    )
    return tf.nest.map_structure(lambda x: x.numpy(), tensor_dict)
