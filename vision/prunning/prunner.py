import torch
import torch.nn as nn
import logging
from heapq import nsmallest

from ..utils.model_book import ModelBook


class ModelPrunner:
    def __init__(self, model, train_fun, ignored_paths=[]):
        self.model = model
        self.train_fun = train_fun
        self.ignored_paths = ignored_paths
        self.book = ModelBook(self.model)
        self.outputs = {}
        self.grads = {}
        self.handles = []
        self.decendent_batch_norms = {}
        self.last_conv_path = None
        self.descendent_convs = {}
        self.descendent_linears = {}
        self.last_linear_path = None

    def _make_new_conv(self, conv, filter_index, channel_type="out"):
        if not isinstance(conv, nn.Conv2d):
            raise TypeError(f"The module is not Conv2d, but {type(conv)}.")

        if channel_type == "out":
            new_conv = nn.Conv2d(conv.in_channels, conv.out_channels - 1, conv.kernel_size, conv.stride,
                                 conv.padding, conv.dilation, conv.groups, conv.bias is not None)
            mask = torch.ones(conv.out_channels, dtype=torch.bool)
            mask[filter_index] = 0
            new_conv.weight.data = conv.weight.data[mask, :, :, :]
            if conv.bias is not None:
                new_conv.bias.data = conv.bias.data[mask]

        elif channel_type == 'in':
            new_in_channels = conv.in_channels - 1
            if new_in_channels % conv.groups != 0:
                raise ValueError(f"After pruning, in_channels={new_in_channels} not divisible by groups={conv.groups}")
            new_conv = nn.Conv2d(new_in_channels, conv.out_channels, conv.kernel_size, conv.stride,
                                 conv.padding, conv.dilation, conv.groups, conv.bias is not None)
            mask = torch.ones(conv.in_channels, dtype=torch.bool)
            mask[filter_index] = 0
            new_conv.weight.data = conv.weight.data[:, mask, :, :]
            if conv.bias is not None:
                new_conv.bias.data = conv.bias.data
        else:
            raise ValueError(f"{channel_type} should be either 'in' or 'out'.")
        return new_conv

    def remove_conv_filter(self, path, filter_index):
        conv = self.book.get_module(path)
        logging.info(f'Prune Conv: {"/".join(path)}, Filter: {filter_index}, Layer: {conv}')
        new_conv = self._make_new_conv(conv, filter_index, channel_type="out")
        self._update_model(path, new_conv)

        next_conv_path = self.descendent_convs.get(path)
        if next_conv_path:
            next_conv = self.book.get_module(next_conv_path)
            new_next_conv = self._make_new_conv(next_conv, filter_index, channel_type="in")
            self._update_model(next_conv_path, new_next_conv)

        batch_norm_path = self.decendent_batch_norms.get(path)
        if batch_norm_path:
            batch_norm = self.book.get_module(batch_norm_path)
            new_batch_norm = nn.BatchNorm2d(batch_norm.num_features - 1)
            self._update_model(batch_norm_path, new_batch_norm)

        linear_path = self.descendent_linears.get(path)
        if linear_path:
            linear = self.book.get_module(linear_path)
            new_linear = self._make_new_linear(linear, filter_index, conv, channel_type="in")
            self._update_model(linear_path, new_linear)

    @staticmethod
    def _make_new_linear(linear, feature_index, conv=None, channel_type="out"):
        if channel_type == "out":
            new_linear = nn.Linear(linear.in_features, linear.out_features - 1,
                                   bias=linear.bias is not None)
            mask = torch.ones(linear.out_features, dtype=torch.bool)
            mask[feature_index] = 0
            new_linear.weight.data = linear.weight.data[mask, :]
            if linear.bias is not None:
                new_linear.bias.data = linear.bias.data[mask]
        elif channel_type == "in":
            if conv:
                block = int(linear.in_features / conv.out_channels)
            else:
                block = 1
            new_linear = nn.Linear(linear.in_features - block, linear.out_features,
                                   bias=linear.bias is not None)
            start_index = feature_index * block
            end_index = (feature_index + 1) * block
            mask = torch.ones(linear.in_features, dtype=torch.bool)
            mask[start_index: end_index] = 0
            new_linear.weight.data = linear.weight.data[:, mask]
            if linear.bias is not None:
                new_linear.bias.data = linear.bias.data
        else:
            raise ValueError(f"{channel_type} should be either 'in' or 'out'.")
        return new_linear

    def prune_conv_layers(self, num=1):
        self.register_conv_hooks()
        before_loss, before_accuracy = self.train_fun(self.model)
        ranks = []
        for path, output in self.outputs.items():
            output = output.data
            grad = self.grads[path].data
            v = grad * output
            v = v.sum(0).sum(1).sum(1)
            v = torch.abs(v)
            v = v / torch.sqrt(torch.sum(v * v))
            for i, e in enumerate(v):
                ranks.append((path, i, e))
        to_prune = nsmallest(num, ranks, key=lambda t: t[2])
        to_prune = sorted(to_prune, key=lambda t: (t[0], -t[1]))
        for path, filter_index, value in to_prune:
            self.remove_conv_filter(path, filter_index)
        self.deregister_hooks()
        after_loss, after_accuracy = self.train_fun(self.model)
        return after_loss - before_loss, after_accuracy - before_accuracy

    def register_conv_hooks(self):
        self.outputs.clear()
        self.grads.clear()
        self.handles.clear()
        self.last_conv_path = None
        self.decendent_batch_norms.clear()
        self.descendent_convs.clear()
        self.descendent_linears.clear()

        def forward_hook(m, input, output):
            path = self.book.get_path(m)
            if isinstance(m, nn.Conv2d):
                if path not in self.ignored_paths:
                    self.outputs[path] = output
                if self.last_conv_path:
                    self.descendent_convs[self.last_conv_path] = path
                self.last_conv_path = path
            elif isinstance(m, nn.BatchNorm2d):
                if self.last_conv_path:
                    self.decendent_batch_norms[self.last_conv_path] = path
            elif isinstance(m, nn.Linear):
                if self.last_conv_path:
                    self.descendent_linears[self.last_conv_path] = path
                self.last_conv_path = None

        def backward_hook(m, grad_input, grad_output):
            path = self.book.get_path(m)
            self.grads[path] = grad_output[0]

        for path, m in self.book.modules(module_type=(nn.Conv2d, nn.BatchNorm2d, nn.Linear)):
            h = m.register_forward_hook(forward_hook)
            self.handles.append(h)
            h = m.register_full_backward_hook(backward_hook)
            self.handles.append(h)

    def deregister_hooks(self):
        for handle in self.handles:
            handle.remove()

    def prune_linear_layers(self, num=1):
        self.register_linear_hooks()
        before_loss, before_accuracy = self.train_fun(self.model)
        ranks = []
        for path, output in self.outputs.items():
            output = output.data
            grad = self.grads[path].data
            v = grad * output
            v = v.sum(0)
            v = torch.abs(v)
            v = v / torch.sqrt(torch.sum(v * v))
            for i, e in enumerate(v):
                ranks.append((path, i, e))
        to_prune = nsmallest(num, ranks, key=lambda t: t[2])
        to_prune = sorted(to_prune, key=lambda t: (t[0], -t[1]))
        for path, feature_index, value in to_prune:
            self.remove_linear_feature(path, feature_index)
        self.deregister_hooks()
        after_loss, after_accuracy = self.train_fun(self.model)
        return after_loss - before_loss, after_accuracy - before_accuracy

    def register_linear_hooks(self):
        self.outputs.clear()
        self.grads.clear()
        self.handles.clear()
        self.descendent_linears.clear()
        self.last_linear_path = None

        def forward_hook(m, input, output):
            path = self.book.get_path(m)
            if path not in self.ignored_paths:
                self.outputs[path] = output
            if self.last_linear_path:
                self.descendent_linears[self.last_linear_path] = path
            self.last_linear_path = path

        def backward_hook(m, grad_input, grad_output):
            path = self.book.get_path(m)
            self.grads[path] = grad_output[0]

        for _, m in self.book.linear_modules():
            h = m.register_forward_hook(forward_hook)
            self.handles.append(h)
            h = m.register_full_backward_hook(backward_hook)
            self.handles.append(h)

    def remove_linear_feature(self, path, feature_index):
        linear = self.book.get_module(path)
        logging.info(f'Prune Linear: {"/".join(path)}, Filter: {feature_index}, Layer: {linear}')
        new_linear = self._make_new_linear(linear, feature_index, channel_type="out")
        self._update_model(path, new_linear)

        next_linear_path = self.descendent_linears.get(path)
        if next_linear_path:
            next_linear = self.book.get_module(next_linear_path)
            new_next_linear = self._make_new_linear(next_linear, feature_index, channel_type='in')
            self._update_model(next_linear_path, new_next_linear)

    def _update_model(self, path, module):
        parent = self.book.get_module(path[:-1])
        parent._modules[path[-1]] = module
        self.book.update(path, module)
