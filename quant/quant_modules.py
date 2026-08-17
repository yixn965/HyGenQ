import random
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from deap import algorithms, base, creator, tools
from sklearn.cluster import KMeans
from tqdm import tqdm

from util.logger_utils import logger
from .quantizer import (
    UniformQuantizer_ar,
    UniformQuantizer_diff,
    UniformQuantizer_group,
    UniformQuantizer_group_scaling,
)


def build_outlier_selector():
    return GeneticKMeans()


class GeneticKMeans:
    """Select abnormal activation channels with the HyGenQ genetic clustering path."""

    def __init__(self):
        self.abnormal_channels = []

    def optimize(self, activation_tensor, layer_name=""):
        reshaped = activation_tensor.reshape(-1, activation_tensor.shape[-1])
        max_values = reshaped.max(dim=0).values.detach().cpu().numpy().reshape(-1, 1)

        logger.info(
            "Processing layer %s with %d channels", layer_name, max_values.shape[0]
        )
        best_k, abnormal_channels = self._find_optimal_clustering(
            max_values, reshaped
        )
        self.abnormal_channels = abnormal_channels
        logger.info(
            "Layer %s - optimal k=%d, found %d abnormal channels",
            layer_name,
            best_k,
            len(abnormal_channels),
        )
        return self.abnormal_channels

    def _find_optimal_clustering(self, max_values, activation_tensor):
        max_clusters = min(14, len(max_values) - 1)
        if max_clusters < 2:
            return 1, []

        candidates = []
        for k in tqdm(range(2, max_clusters + 1), desc="Testing clustering options"):
            _, labels = self._genetic_kmeans(max_values, k)
            quantization_error, abnormal_channels = self._compute_quantization_error(
                labels, activation_tensor, max_values
            )
            candidates.append((quantization_error, k, abnormal_channels))
            logger.info("k=%d - quantization error: %.6f", k, quantization_error)

        _, best_k, abnormal_channels = min(candidates, key=lambda item: item[0])
        return best_k, abnormal_channels

    def _genetic_kmeans(
        self,
        data,
        k,
        pop_size=50,
        ngen=20,
        cxpb=0.5,
        mutpb=0.2,
        random_state=42,
    ):
        np.random.seed(random_state)
        random.seed(random_state)
        n_samples, n_features = data.shape

        if not hasattr(creator, "HyGenQFitnessMin"):
            creator.create("HyGenQFitnessMin", base.Fitness, weights=(-1.0,))
        if not hasattr(creator, "HyGenQIndividual"):
            creator.create(
                "HyGenQIndividual",
                list,
                fitness=creator.HyGenQFitnessMin,
            )

        toolbox = base.Toolbox()

        def create_individual():
            indices = np.random.choice(n_samples, size=k, replace=False)
            return data[indices].flatten().tolist()

        toolbox.register(
            "individual", tools.initIterate, creator.HyGenQIndividual, create_individual
        )
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", self._kmeans_fitness, data=data)
        toolbox.register("mate", tools.cxBlend, alpha=0.5)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.1)
        toolbox.register("select", tools.selTournament, tournsize=3)

        population = toolbox.population(n=pop_size)
        algorithms.eaSimple(
            population,
            toolbox,
            cxpb=cxpb,
            mutpb=mutpb,
            ngen=ngen,
            verbose=False,
        )

        best_individual = tools.selBest(population, k=1)[0]
        centers = np.asarray(best_individual).reshape(k, n_features)
        distances = np.asarray(
            [np.sum((data - center) ** 2, axis=1) for center in centers]
        )
        return centers, np.argmin(distances, axis=0)

    @staticmethod
    def _kmeans_fitness(individual, data):
        k = len(individual) // data.shape[1]
        centers = np.asarray(individual).reshape(k, data.shape[1])
        distances = np.asarray(
            [np.sum((data - center) ** 2, axis=1) for center in centers]
        )
        return (np.min(distances, axis=0).sum(),)

    @staticmethod
    def _compute_quantization_error(labels, activation_tensor, max_values):
        labels = np.asarray(labels)
        tensor_np = activation_tensor.detach().cpu().numpy()
        unique_labels = np.unique(labels)

        if len(unique_labels) == 2:
            cluster_max = {
                label: np.max(max_values[labels == label]) for label in unique_labels
            }
            main_cluster = min(cluster_max, key=cluster_max.get)
            normal_channels = np.where(labels == main_cluster)[0]
            abnormal_channels = np.where(labels != main_cluster)[0]
        else:
            cluster_maxima = np.asarray(
                [np.max(max_values[labels == label]) for label in unique_labels]
            ).reshape(-1, 1)
            secondary_kmeans = KMeans(
                n_clusters=2,
                init="k-means++",
                random_state=42,
                n_init=10,
            ).fit(cluster_maxima)
            secondary_labels = secondary_kmeans.labels_
            main_group = min(
                np.unique(secondary_labels),
                key=lambda group: np.mean(cluster_maxima[secondary_labels == group]),
            )
            normal_labels = unique_labels[secondary_labels == main_group]
            normal_mask = np.isin(labels, normal_labels)
            normal_channels = np.where(normal_mask)[0]
            abnormal_channels = np.where(~normal_mask)[0]

        if normal_channels.size == 0 or abnormal_channels.size == 0:
            return 0.0, abnormal_channels.tolist()

        normal_max = np.max(tensor_np[..., normal_channels])
        scaled_tensor = tensor_np.copy()
        scale_factors = {}
        for channel in abnormal_channels:
            channel_max = np.max(tensor_np[..., channel])
            scale_factor = normal_max / channel_max if channel_max else 1.0
            scale_factors[channel] = scale_factor
            scaled_tensor[..., channel] *= scale_factor

        minimum = np.min(scaled_tensor)
        maximum = np.max(scaled_tensor)
        scale = 255.0 / (maximum - minimum) if maximum != minimum else 1.0
        quantized_tensor = (
            np.round((scaled_tensor - minimum) * scale).astype(np.uint8).astype(np.float32)
            / scale
            + minimum
        )

        abnormal_tensor = scaled_tensor[..., abnormal_channels]
        quantized_abnormal = quantized_tensor[..., abnormal_channels]
        sign_mask = np.sign(quantized_abnormal) != np.sign(abnormal_tensor)
        quantized_abnormal[sign_mask] *= -1
        quantized_tensor[..., abnormal_channels] = quantized_abnormal

        for channel in abnormal_channels:
            quantized_tensor[..., channel] /= scale_factors[channel]

        error = np.sum(
            (quantized_tensor[..., abnormal_channels] - tensor_np[..., abnormal_channels]) ** 2
        )
        return error, abnormal_channels.tolist()


class QuantLinear_ar(nn.Linear):
    """Quantized linear layer for autoregressive MAR components."""

    def __init__(self, in_features, out_features, input_quant_params={}, weight_quant_params={}, i=None):
        super().__init__(in_features, out_features)
        self.input_quantizer = UniformQuantizer_ar(**input_quant_params)
        self.weight_quantizer = UniformQuantizer_ar(**weight_quant_params)
        self.use_input_quant = False
        self.use_weight_quant = False
        self.i = -1

    def __repr__(self):
        return "({}input_quant={}, weight_quant={})".format(
            super().__repr__(), self.use_input_quant, self.use_weight_quant
        )

    def set_quant_state(self, input_quant=True, weight_quant=True):
        self.use_input_quant = input_quant
        self.use_weight_quant = weight_quant

    def forward(self, x, i=None, step=None, calib5=False, d=False, layer_name=None, params=None, adjustment=False, num=None, a=False, num_bsz=-1, scale_quant=None, shift_quant=None):
        self.i = i
        if self.use_input_quant:
            x = self.input_quantizer(
                x, i=i, step=step, calib5=calib5, adjustment=adjustment
            )
        if self.use_weight_quant:
            weight = self.weight_quantizer(
                self.weight, i=i, step=step, calib5=calib5, is_weight=True
            )
        else:
            weight = self.weight
        return F.linear(x, weight=weight, bias=self.bias)


class QuantLinear_ar_outlier(nn.Linear):
    """Autoregressive linear layer with genetic-clustered outlier channels."""

    def __init__(self, in_features, out_features, input_quant_params={}, weight_quant_params={}):
        super().__init__(in_features, out_features)
        self.input_quantizer = UniformQuantizer_group(**input_quant_params)
        self.weight_quantizer = UniformQuantizer_ar(**weight_quant_params)
        self.use_input_quant = False
        self.use_weight_quant = False
        self.abnormal_channels = []

    def __repr__(self):
        return "({}input_quant={}, weight_quant={})".format(
            super().__repr__(), self.use_input_quant, self.use_weight_quant
        )

    def set_quant_state(self, input_quant=True, weight_quant=True):
        self.use_input_quant = input_quant
        self.use_weight_quant = weight_quant

    def adjust_and_quantize(self, tensor, quantizer, is_weight=False, i=None, step=None, calib5=False, layer_name=None, adjustment=False):
        if is_weight:
            return quantizer(
                tensor.clone(),
                i=i,
                step=step,
                calib5=calib5,
                is_weight=True,
                adjustment=adjustment,
            )

        if calib5 and not self.abnormal_channels:
            activation_tensor = tensor.reshape(-1, tensor.shape[-1]).clone()
            self.abnormal_channels = build_outlier_selector().optimize(
                activation_tensor=activation_tensor,
                layer_name=layer_name,
            )

        if not self.abnormal_channels:
            return quantizer(
                tensor,
                group_name="low",
                i=i,
                step=step,
                calib5=calib5,
                is_weight=False,
            )

        if calib5:
            max_values = tensor.reshape(-1, tensor.shape[-1]).max(dim=0).values
            normal_mask = torch.ones_like(max_values, dtype=torch.bool)
            normal_mask[self.abnormal_channels] = False
            cluster_max = max_values[normal_mask].max().item()
            quantized_tensor = quantizer(
                tensor,
                group_name="low",
                i=i,
                step=step,
                calib5=True,
                is_weight=False,
                threshold=cluster_max,
            )
        else:
            quantized_tensor = quantizer(
                tensor,
                group_name="low",
                i=i,
                step=step,
                calib5=False,
                is_weight=False,
            )

        abnormal_tensor = tensor[..., self.abnormal_channels]
        quantized_tensor[..., self.abnormal_channels] = quantizer(
            abnormal_tensor,
            group_name="high",
            i=i,
            step=step,
            calib5=calib5,
            is_weight=False,
        )
        return quantized_tensor

    def forward(self, x, i=None, step=None, calib5=False, d=False, layer_name=None, params=None, adjustment=False, num=None, a=False, num_bsz=-1):
        if params:
            self.weight_quantizer.channel_wise = params.get(
                "channel_wise", self.weight_quantizer.channel_wise
            )
        if self.use_input_quant:
            x = self.adjust_and_quantize(
                x,
                self.input_quantizer,
                i=i,
                step=step,
                calib5=calib5,
                layer_name=layer_name,
                adjustment=adjustment,
            )
        weight = (
            self.adjust_and_quantize(
                self.weight,
                self.weight_quantizer,
                is_weight=True,
                i=i,
                step=step,
                calib5=calib5,
                layer_name=layer_name,
                adjustment=adjustment,
            )
            if self.use_weight_quant
            else self.weight
        )
        return F.linear(x, weight, self.bias)


class QuantLinear_scaling(nn.Linear):
    """Quantized linear layer for the scaled DiffLoss input and output layers."""

    def __init__(self, in_features, out_features, input_quant_params={}, weight_quant_params={}, i=None):
        super().__init__(in_features, out_features)
        self.input_quantizer = UniformQuantizer_group_scaling(**input_quant_params)
        self.weight_quantizer = UniformQuantizer_diff(**weight_quant_params)
        self.use_input_quant = False
        self.use_weight_quant = False
        self.i = None

    def __repr__(self):
        return "({}input_quant={}, weight_quant={})".format(
            super().__repr__(), self.use_input_quant, self.use_weight_quant
        )

    def set_quant_state(self, input_quant=True, weight_quant=True):
        self.use_input_quant = input_quant
        self.use_weight_quant = weight_quant

    def forward(self, x, i=None, step=None, calib5=False, d=False, layer_name=None, params=None, adjustment=False, num=None, a=False, num_bsz=-1, sign_scaling=False, scale_quant=None, shift_quant=None):
        if params:
            self.weight_quantizer.channel_wise = params.get(
                "channel_wise", self.weight_quantizer.channel_wise
            )
        self.i = i
        if self.use_input_quant:
            x = self.input_quantizer(
                x,
                i=i,
                step=step,
                calib5=calib5,
                adjustment=adjustment,
                layer_name=layer_name,
                scale_quant=scale_quant,
                shift_quant=shift_quant,
            )
        weight = (
            self.weight_quantizer(
                self.weight, i=i, step=step, calib5=calib5, is_weight=True
            )
            if self.use_weight_quant
            else self.weight
        )
        return F.linear(x, weight=weight, bias=self.bias)


class QuantLinear_diff(nn.Linear):
    """Quantized linear layer for DiffLoss components."""

    def __init__(self, in_features, out_features, input_quant_params={}, weight_quant_params={}, i=None):
        super().__init__(in_features, out_features)
        self.input_quantizer = UniformQuantizer_diff(**input_quant_params)
        self.weight_quantizer = UniformQuantizer_diff(**weight_quant_params)
        self.use_input_quant = False
        self.use_weight_quant = False
        self.initial_zs = 128
        self.initial_ss = torch.full((6400,), 0.04347, device=self.weight.device)
        self.adjustment_factor = torch.zeros(6400, device=self.weight.device)
        self.i = None

    def __repr__(self):
        return "({}input_quant={}, weight_quant={})".format(
            super().__repr__(), self.use_input_quant, self.use_weight_quant
        )

    def set_quant_state(self, input_quant=True, weight_quant=True):
        self.use_input_quant = input_quant
        self.use_weight_quant = weight_quant

    def forward(self, x, i=None, step=None, calib5=False, d=False, layer_name=None, params=None, adjustment=False, num=None, a=False, num_bsz=-1, scale_quant=None, shift_quant=None):
        if params:
            self.weight_quantizer.channel_wise = params.get(
                "channel_wise", self.weight_quantizer.channel_wise
            )
        self.i = i
        if self.use_input_quant:
            x = self.input_quantizer(
                x, i=i, step=step, calib5=calib5, adjustment=adjustment
            )
        weight = (
            self.weight_quantizer(
                self.weight, i=i, step=step, calib5=calib5, is_weight=True
            )
            if self.use_weight_quant
            else self.weight
        )
        return F.linear(x, weight=weight, bias=self.bias)


class QuantMatMul(nn.Module):
    """Quantized matrix multiplication for attention score and value products."""

    def __init__(self, input_quant_params={}):
        super().__init__()
        input_quant_params = deepcopy(input_quant_params)
        self.quantizer_A = UniformQuantizer_ar(**input_quant_params)
        self.quantizer_B = UniformQuantizer_ar(**input_quant_params)
        self.use_input_quant = False
        self.i = None

    def __repr__(self):
        return "({}input_quant={})".format(super().__repr__(), self.use_input_quant)

    def set_quant_state(self, input_quant=False, weight_quant=False):
        self.use_input_quant = input_quant

    def forward(self, A, B, i=None, calib5=False, step=None, d=False, layer_name=None):
        self.i = i
        if self.use_input_quant:
            A = self.quantizer_A(A, i=i, calib5=calib5, step=step)
            B = self.quantizer_B(B, i=i, calib5=calib5, step=step)
        return A @ B
