# Adapted from https://github.com/facebookresearch/SlowFast
import torch
import torch.nn.functional as F


def get_layer(model, layer_name):
    """
    Return the targeted layer (nn.Module Object) given a hierarchical layer name,
    separated by /.
    Args:
        model (model): model to get layers from.
        layer_name (str): name of the layer.
    Returns:
        prev_module (nn.Module): the layer from the model with `layer_name` name.
    """
    layer_ls = layer_name.split("/")
    prev_module = model
    for layer in layer_ls:
        prev_module = prev_module._modules[layer]

    return prev_module


class GradCAM:
    """
    GradCAM class helps create localization maps using the Grad-CAM method for input videos
    and overlap the maps over the input videos as heatmaps.
    https://arxiv.org/pdf/1610.02391.pdf
    """

    def __init__(self, model, target_layers, data_mean, data_std, colormap="viridis"):
        """
        Args:
            model (model): the model to be used.
            target_layers (list of str(s)): name of convolutional layer to be used to get
                gradients and feature maps from for creating localization maps.
            data_mean (tensor or list): mean value to add to input videos.
            data_std (tensor or list): std to multiply for input videos.
            colormap (Optional[str]): matplotlib colormap used to create heatmap.
                See https://matplotlib.org/3.3.0/tutorials/colors/colormaps.html
        """

        self.model = model
        # Run in eval mode.
        self.model.eval()
        self.target_layers = target_layers

        self.gradients = {}
        self.activations = {}
        self.data_mean = data_mean
        self.data_std = data_std
        self._register_hooks()

    def _register_single_hook(self, layer_name):
        """
        Register forward and backward hook to a layer, given layer_name,
        to obtain gradients and activations.
        Args:
            layer_name (str): name of the layer.
        """

        def get_activations(_module, _input, output):
            self.activations[layer_name] = output

        target_layer = get_layer(self.model, layer_name=layer_name)
        target_layer.register_forward_hook(get_activations)

    def _register_hooks(self):
        """
        Register hooks to layers in `self.target_layers`.
        """
        for layer_name in self.target_layers:
            self._register_single_hook(layer_name=layer_name)

    def _calculate_localization_map(self, inputs, labels=None):
        """
        Calculate localization map for all inputs with Grad-CAM.
        Args:
            inputs (list of tensor(s)): the input clips.
            labels (Optional[tensor]): labels of the current input clips.
        Returns:
            localization_maps (list of ndarray(s)): the localization map for
                each corresponding input.
            preds (tensor): shape (n_instances, n_class). Model predictions for `inputs`.
        """
        assert (
                len(inputs) == len(self.target_layers)
        ), "Must register the same number of target layers as the number of input pathways."
        preds = self.model(inputs)

        if labels is None:
            score = torch.max(preds, dim=-1)[0]
        else:
            if labels.ndim == 1:
                labels = labels.unsqueeze(-1)
            score = torch.gather(preds, dim=1, index=labels)

        score = torch.sum(score)
        localization_maps = []
        for i, inp in enumerate(inputs):
            _, _, T, H, W = inp.size()

            activations = self.activations[self.target_layers[i]]
            gradients = torch.autograd.grad(score,
                                            activations,
                                            create_graph=True,
                                            retain_graph=True)[0]
            B, C, Tg, _, _ = gradients.size()

            weights = torch.mean(gradients.view(B, C, Tg, -1), dim=3)

            weights = weights.view(B, C, Tg, 1, 1)
            localization_map = torch.sum(weights * activations, dim=1, keepdim=True)
            localization_map = F.relu(localization_map)
            localization_map = F.interpolate(
                localization_map,
                size=(T, H, W),
                mode="trilinear",
                align_corners=False,
            )
            localization_map_min, localization_map_max = (
                torch.min(localization_map.view(B, -1), dim=-1, keepdim=True)[0],
                torch.max(localization_map.view(B, -1), dim=-1, keepdim=True)[0],
            )
            localization_map_min = torch.reshape(
                localization_map_min, shape=(B, 1, 1, 1, 1)
            )
            localization_map_max = torch.reshape(
                localization_map_max, shape=(B, 1, 1, 1, 1)
            )
            # Normalize the localization map.
            localization_map = (localization_map - localization_map_min) / (
                    localization_map_max - localization_map_min + 1e-6
            )
            localization_map = localization_map.data

            localization_maps.append(localization_map)

        return localization_maps, preds

    def __call__(self, inputs, labels=None, alpha=0.5):
        """
        Visualize the localization maps on their corresponding inputs as heatmap,
        using Grad-CAM.
        Args:
            inputs (list of tensor(s)): the input clips.
            labels (Optional[tensor]): labels of the current input clips.
            alpha (float): transparency level of the heatmap, in the range [0, 1].
        Returns:
            localization_maps (list of tensor(s)): the visualized inputs.
            preds (tensor): shape (n_instances, n_class). Model predictions for `inputs`.
        """
        localization_maps, preds = self._calculate_localization_map(
            inputs, labels=labels
        )
        return localization_maps, preds
