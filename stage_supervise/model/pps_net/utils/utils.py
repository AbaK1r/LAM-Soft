import torch

from model.pps_net.utils import optical_flow_funs as OF


def get_normals_from_depth(
        depth, intrinsics, depth_is_along_ray=False, diff_type='center', normalized_intrinsics=True):
    """

    :param depth: (b,1,m,n)
    :param intrinsics: (b,3,3)
    :param depth_is_along_ray:
    :param diff_type:
    :param normalized_intrinsics:
    :return: (b,3,m,n), (b,3,m,n)
    """
    dirs = OF.get_camera_pixel_directions(depth.shape[2:4], intrinsics, normalized_intrinsics=normalized_intrinsics)
    dirs = dirs.permute(0, 3, 1, 2)
    if depth_is_along_ray:
        dirs = torch.nn.functional.normalize(dirs, dim=1)
    pc = dirs * depth

    normal = point_cloud_to_normals(pc, diff_type=diff_type)
    return normal, pc


def point_cloud_to_normals(pc, diff_type='center'):
    # pc (b,3,m,n)
    # return (b,3,m,n)
    dp_du, dp_dv = image_derivatives(pc, diff_type=diff_type)
    normal = torch.nn.functional.normalize(torch.cross(dp_du, dp_dv, dim=1))
    return normal


def image_derivatives(image, diff_type='center'):
    c = image.size(1)
    if diff_type == 'center':
        sobel_x = 0.5 * torch.tensor(
            [[0.0, 0, 0], [-1, 0, 1], [0, 0, 0]],
            device=image.device
        ).view(1, 1, 3, 3).repeat(c, 1, 1, 1)
        sobel_y = 0.5 * torch.tensor(
            [[0.0, 1, 0], [0, 0, 0], [0, -1, 0]],
            device=image.device
        ).view(1, 1, 3, 3).repeat(c, 1, 1, 1)
    elif diff_type == 'forward':
        sobel_x = torch.tensor(
            [[0.0, 0, 0], [0, -1, 1], [0, 0, 0]],
            device=image.device
        ).view(1, 1, 3, 3).repeat(c, 1, 1, 1)
        sobel_y = torch.tensor(
            [[0.0, 1, 0], [0, -1, 0], [0, 0, 0]],
            device=image.device
        ).view(1, 1, 3, 3).repeat(c, 1, 1, 1)
    else:
        raise NotImplementedError

    dp_du = torch.nn.functional.conv2d(image, sobel_x, padding=1, groups=3)
    dp_dv = torch.nn.functional.conv2d(image, sobel_y, padding=1, groups=3)
    return dp_du, dp_dv


def normalize_images(tensor):
    """
    Normalize each image in the batch to the range [0, 1].

    :param  tensor: A PyTorch tensor of shape [batch_size, channels, height, width].
    :return: The normalized tensor.
    """
    # Flatten the tensor along the channel, height, and width dimensions
    flattened_tensor = tensor.flatten(1)

    # Calculate the minimum and maximum values for each image in the batch
    mins, _ = torch.min(flattened_tensor, dim=1, keepdim=True)
    maxs, _ = torch.max(flattened_tensor, dim=1, keepdim=True)

    # Reshape the mins and maxs back to the original shape
    mins = mins.view(-1, 1, 1, 1)
    maxs = maxs.view(-1, 1, 1, 1)

    # Avoid division by zero
    epsilon = 1e-8

    # Perform the normalization
    normalized_tensor = (tensor - mins) / (maxs - mins + epsilon)

    return normalized_tensor
