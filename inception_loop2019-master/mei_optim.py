import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
import warnings

# 移除 DataJoint 依赖 - 不需要 dj 相关的导入

def fft_smooth(grad, factor=1/4):
    """
    Tones down the gradient with 1/sqrt(f) filter in the Fourier domain.
    Equivalent to low-pass filtering in the spatial domain.
    """
    if factor == 0:
        return grad
    h, w = grad.size()[-2:]
    tw = np.minimum(np.arange(0, w), np.arange(w-1, -1, -1), dtype=np.float32)
    th = np.minimum(np.arange(0, h), np.arange(h-1, -1, -1), dtype=np.float32)
    t = 1 / np.maximum(1.0, (tw[None, :] ** 2 + th[:, None] ** 2) ** (factor))
    F = grad.new_tensor(t / t.mean()).unsqueeze(-1)
    pp = torch.rfft(grad.data, 2, onesided=False)
    return torch.irfft(pp * F, 2, onesided=False)

def blur(img, sigma):
    if sigma > 0:
        for d in range(len(img)):
            img[d] = ndimage.filters.gaussian_filter(img[d], sigma, order=0)
    return img

def blur_in_place(tensor, sigma):
    blurred = np.stack([blur(im, sigma) for im in tensor.cpu().numpy()])
    tensor.copy_(torch.Tensor(blurred))

def roll(tensor, shift, axis):
    if shift == 0:
        return tensor

    if axis < 0:
        axis += tensor.dim()

    dim_size = tensor.size(axis)
    after_start = dim_size - shift
    if shift < 0:
        after_start = -shift
        shift = dim_size - abs(shift)

    before = tensor.narrow(axis, 0, dim_size - shift)
    after = tensor.narrow(axis, after_start, shift)
    return torch.cat([after, before], axis)

def process(x, mu=0.4, sigma=0.224):
    """ Normalize and move channel dim in front of height and width"""
    x = (x - mu) / sigma
    if isinstance(x, torch.Tensor):
        return x.transpose(-1, -2).transpose(-2, -3)
    else:
        return np.moveaxis(x, -1, -3)

def unprocess(x, mu=0.4, sigma=0.224):
    """Inverse of process()"""
    x = x * sigma + mu
    if isinstance(x, torch.Tensor):
        return x.transpose(-3, -2).transpose(-2, -1)
    else:
        return np.moveaxis(x, -3, -1)

def make_step(net, src, step_size=1.5, sigma=None, precond=0, step_gain=1,
              blur=True, jitter=0, eps=1e-12, clip=True, bias=0.4, scale=0.224,
              train_norm=None, norm=None, add_loss=0, _eps=1e-12):
    """ Update src in place making a gradient ascent step in the output of net.

    Arguments:
        net (nn.Module or function): A backpropagatable function/module that receives
            images in (B x C x H x W) form and outputs a scalar value per image.
        src (torch.Tensor): Batch of images to update (B x C x H x W).
        step_size (float): Step size to use for the update: (im_old += step_size * grad)
        sigma (float): Standard deviation for gaussian smoothing (if used, see blur).
        precond (float): Strength of gradient smoothing.
        step_gain (float): Scaling factor for the step size.
        blur (boolean): Whether to blur the image after the update.
        jitter (int): Randomly shift the image this number of pixels before forwarding
            it through the network.
        eps (float): Small value to avoid division by zero.
        clip (boolean): Whether to clip the range of the image to be in [0, 255]
        train_norm (float): Decrease standard deviation of the image feed to the
            network to match this norm. Expressed in original pixel values. Unused if
            None
        norm (float): Decrease standard deviation of the image to match this norm after
            update. Expressed in z-scores. Unused if None
        add_loss (function): An additional term to add to the network activation before
            calling backward on it. Usually, some regularization.
    """
    if src.grad is not None:
        src.grad.zero_()

    # apply jitter shift
    if jitter > 0:
        ox, oy = np.random.randint(-jitter, jitter + 1, 2)  # use uniform distribution
        ox, oy = int(ox), int(oy)
        src.data = roll(roll(src.data, ox, -1), oy, -2)

    img = src
    if train_norm is not None and train_norm > 0.0:
        # normalize the image in backpropagatable manner
        img_idx = src.data.std(dim=(1, 2, 3), keepdim=True) + _eps > train_norm / scale
        if img_idx.any():
            img = src.clone()  # avoids overwriting original image but lets gradient through
            img[img_idx] = ((src[img_idx] / (src[img_idx].std(dim=(1, 2, 3), keepdim=True) + _eps)) * (train_norm / scale))

    y = net(img)
    (y.mean() + add_loss).backward()

    grad = src.grad
    if precond > 0:
        grad = fft_smooth(grad, precond)

    # Modern PyTorch gradient update
    with torch.no_grad():
        update = (step_size / (torch.abs(grad.data).mean() + eps)) * (step_gain / 255) * grad.data
        src.data.add_(update)

    #print(src.data.std() * scale)
    if norm is not None and norm > 0.0:
        data_idx = src.data.std(dim=(1, 2, 3), keepdim=True) + _eps > norm / scale
        src.data[data_idx] = (src.data / (src.data.std(dim=(1, 2, 3), keepdim=True) + _eps) * norm / scale)[data_idx]

    if jitter > 0:
        # undo the shift
        src.data = roll(roll(src.data, -ox, -1), -oy, -2)

    if clip:
        src.data = torch.clamp(src.data, -bias / scale, (255 - bias) / scale)

    if blur:
        blur_in_place(src.data, sigma)

def deepdraw(net, base_img, octaves, random_crop=True, original_size=None,
             bias=None, scale=None, device='cpu', **step_params):
    """ Generate an image by iteratively optimizing activity of net.

    Arguments:
        net (nn.Module or function): A backpropagatable function/module that receives
            images in (B x C x H x W) form and outputs a scalar value per image.
        base_img (np.array): Initial image (h x w x c)
        octaves (list of dict): Configurations for each octave:
            n_iter (int): Number of iterations in this octave
            start_sigma (float): Initial standard deviation for gaussian smoothing (if
                used, see blur)
            end_sigma (float): Final standard deviation for gaussian smoothing (if used,
                see blur)
            start_step_size (float): Initial value of the step size used each iteration to
                update the image (im_old += step_size * grad).
            end_step_size (float): Initial value of the step size used each iteration to
                update the image (im_old += step_size * grad).
            (optionally) scale (float): If set, the image will be scaled using this factor
                during optimization. (Original image size is left unchanged).
        random_crop (boolean): If image to optimize is bigger than networks input image,
            optimize random crops of the image each iteration.
        original_size (triplet): (channel, height, width) expected by the network. If
            None, it uses base_img's.
        bias (float), scale (float): Values used for image normalization (at the very
            start of processing): (base_img - bias) / scale.
        device (torch.device or str): Device where the network is located.
        step_params (dict): A handful of optional parameters that are directly sent to
            make_step() (see docstring of make_step for a description).

    Returns:
        A h x w array. The optimized image.
    """
    # prepare base image
    image = process(base_img, mu=bias, sigma=scale)  # (3,224,224)

    # get input dimensions from net
    if original_size is None:
        print('getting image size:')
        c, w, h = image.shape[-3:]
    else:
        c, w, h = original_size

    print("starting drawing")

    src = torch.zeros(1, c, w, h, requires_grad=True, device=device)

    for e, o in enumerate(octaves):
        if 'scale' in o:
            # resize by o['scale'] if it exists
            image = ndimage.zoom(image, (1, o['scale'], o['scale']))
        _, imw, imh = image.shape
        for i in range(o['iter_n']):
            if imw > w:
                if random_crop:
                    # randomly select a crop
                    mid_x = (imw - w) / 2.
                    width_x = imw - w
                    ox = np.random.normal(mid_x, width_x * 0.3, 1)
                    ox = int(np.clip(ox, 0, imw - w))
                    mid_y = (imh - h) / 2.
                    width_y = imh - h
                    oy = np.random.normal(mid_y, width_y * 0.3, 1)
                    oy = int(np.clip(oy, 0, imh - h))
                    # insert the crop into src.data[0]
                    src.data[0].copy_(torch.Tensor(image[:, ox:ox + w, oy:oy + h]))
                else:
                    ox = int((imw - w) / 2)
                    oy = int((imh - h) / 2)
                    src.data[0].copy_(torch.Tensor(image[:, ox:ox + w, oy:oy + h]))
            else:
                ox = 0
                oy = 0
                src.data[0].copy_(torch.Tensor(image))

            sigma = o['start_sigma'] + ((o['end_sigma'] - o['start_sigma']) * i) / o['iter_n']
            step_size = o['start_step_size'] + ((o['end_step_size'] - o['start_step_size']) * i) / o['iter_n']

            make_step(net, src, bias=bias, scale=scale, sigma=sigma, step_size=step_size, **step_params)

            if i % 10 == 0:
                print('finished step %d in octave %d' % (i, e))

            # insert modified image back into original image (if necessary)
            image[:, ox:ox + w, oy:oy + h] = src.data[0].cpu().numpy()

    # returning the resulting image
    return unprocess(image, mu=bias, sigma=scale)