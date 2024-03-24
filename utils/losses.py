import torch
import torch.nn as nn
import torchvision.models as models


class GANLoss(nn.Module):
    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0, loss_weight=1.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type
        self.loss_weight = loss_weight
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        else:
            raise NotImplementedError(
                f'GAN type {self.gan_type} is not implemented.')

    def get_target_label(self, input, target_is_real):
        """Get target label.

        Args:
            input (Tensor): Input tensor.
            target_is_real (bool): Whether the target is real or fake.

        Returns:
            (bool | Tensor): Target tensor. Return bool for wgan, otherwise,
                return Tensor.
        """

        target_val = (
            self.real_label_val if target_is_real else self.fake_label_val)
        return input.new_ones(input.size()) * target_val

    def forward(self, input, target_is_real, is_disc=False):
        """
        Args:
            input (Tensor): The input for the loss module, i.e., the network
                prediction.
            target_is_real (bool): Whether the targe is real or fake.
            is_disc (bool): Whether the loss for discriminators or not.
                Default: False.

        Returns:
            Tensor: GAN loss value.
        """
        target_label = self.get_target_label(input, target_is_real)
        loss = self.loss(input, target_label)
        # loss_weight is always 1.0 for discriminators
        return loss if is_disc else loss * self.loss_weight
        # return loss


class PerceptualVGG(nn.Module):

    def __init__(self,
                 layer_name_list,
                 vgg_type='vgg19',
                 use_input_norm=True):
        super().__init__()
        self.layer_name_list = layer_name_list
        self.use_input_norm = use_input_norm

        # get vgg model and load pretrained vgg weight
        # remove _vgg from attributes to avoid `find_unused_parameters` bug

        assert vgg_type in ('vgg16', 'vgg19')
        if vgg_type == 'vgg16':
            _vgg = models.vgg16(pretrained=True)
        else:
            _vgg = models.vgg19(pretrained=True)

        num_layers = max(map(int, layer_name_list)) + 1
        assert len(_vgg.features) >= num_layers
        # only borrow layers that will be used from _vgg to avoid unused params
        self.vgg_layers = _vgg.features[:num_layers]

        if self.use_input_norm:
            # the mean is for image with range [0, 1]
            self.register_buffer(
                'mean',
                torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            # the std is for image with range [-1, 1]
            self.register_buffer(
                'std',
                torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        for v in self.vgg_layers.parameters():
            v.requires_grad = False

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """

        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = {}

        for name, module in self.vgg_layers.named_children():
            x = module(x)
            if name in self.layer_name_list:
                output[name] = x.clone()
        return output


class PerceptualLoss(nn.Module):
    """Perceptual loss with commonly used style loss.

    Args:
        layers_weights (dict): The weight for each layer of vgg feature.
            Here is an example: {'4': 1., '9': 1., '18': 1.}, which means the
            5th, 10th and 18th feature layer will be extracted with weight 1.0
            in calculating losses.
    """

    def __init__(self,
                 layer_weights,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 perceptual_weight=1.0,
                 style_weight=0.,
                 norm_img=False,
                 criterion='l1'):
        super().__init__()
        self.norm_img = norm_img
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.layer_weights = layer_weights
        self.vgg = PerceptualVGG(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm)

        criterion = criterion.lower()
        if criterion == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif criterion == 'mse':
            self.criterion = torch.nn.MSELoss()
        else:
            raise NotImplementedError(
                f'{criterion} criterion has not been supported in'
                ' this version.')

    def forward(self, x, gt):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """

        if self.norm_img:
            x = (x + 1.) * 0.5
            gt = (gt + 1.) * 0.5
        # extract vgg features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        # calculate perceptual loss
        if self.perceptual_weight > 0:
            percep_loss = 0
            for k in x_features.keys():
                percep_loss += self.criterion(
                    x_features[k], gt_features[k]) * self.layer_weights[k]
            percep_loss *= self.perceptual_weight
        else:
            percep_loss = None

        # calculate style loss
        if self.style_weight > 0:
            style_loss = 0
            for k in x_features.keys():
                style_loss += self.criterion(
                    self._gram_mat(x_features[k]),
                    self._gram_mat(gt_features[k])) * self.layer_weights[k]
            style_loss *= self.style_weight
        else:
            style_loss = None

        return percep_loss

    def _gram_mat(self, x):
        """Calculate Gram matrix.

        Args:
            x (torch.Tensor): Tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Gram matrix.
        """
        (n, c, h, w) = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram
    