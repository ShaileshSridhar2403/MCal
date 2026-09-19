import numpy as np
import torch
import random

def patch_segment(image, patch_height=8, patch_width=8,permute = None,dtype = "torch"):
    #Input: C,H,W
    if permute is not None:
        image = np.transpose(image,permute)

    channels, image_height, image_width = image.shape
    if image_height % patch_height != 0 or image_width % patch_width != 0:
        raise ValueError("patch height and width need to perfectly divide image")

    row_indices = torch.arange(image_height)
    column_indices = torch.arange(image_width)

    row_factors = row_indices // patch_height
    column_factors = column_indices // patch_width

    row_factor_matrix = row_factors[:, None]
    column_factor_matrix = column_factors[None, :]

    segment = column_factor_matrix * (image_height // patch_height) + row_factor_matrix
    if dtype == "torch":
        return segment.int()
    elif dtype == "numpy":
        return segment.numpy()
    else:
        raise ValueError



def remove_random_features(image, segmentation_fn, removal_fraction, patch_height,patch_width,fill_val=0):

    segments = segmentation_fn(image,patch_height,patch_width).to(image.device)

    # Calculate the number of segments to retain
    n_to_retain = int((1 - removal_fraction) * (segments.max() - segments.min() + 1))
    # Generate random indices for retained features
    min_segment = segments.min()
    max_segment = segments.max()
    retained_features = torch.tensor(random.sample(range(min_segment, max_segment + 1), n_to_retain)).to(image.device)

    # Create a mask using broadcasting and logical operations
    expanded_retained_features = retained_features.view(1, -1, 1, 1)
    mask = (segments == expanded_retained_features)
    mask = mask.sum(1) > 0
    result = segments * mask.float()

    # Apply the mask to the image using broadcasting
    masked_image = (image * mask.unsqueeze(0)).squeeze()

    # Fill the non-retained segments with the specified fill value
    masked_image = masked_image + (1 - mask.int()).unsqueeze(0) * torch.tensor(fill_val).to(image.device).view(-1, 1, 1)
    return masked_image.squeeze()


def remove_mask(image, segmentation_fn, mask_vector, patch_height,patch_width,fill_val=0):

    segments = segmentation_fn(image,patch_height,patch_width)

    # Calculate the number of segments to retain
    # n_to_retain = int((1 - removal_fraction) * (segments.max() - segments.min() + 1))
    # Generate random indices for retained features
    min_segment = segments.min()
    max_segment = segments.max()
    # retained_features = torch.tensor(random.sample(range(min_segment, max_segment + 1), n_to_retain))
    retained_features = torch.tensor(np.where(1 - mask_vector)[0])


    # Create a mask using broadcasting and logical operations
    expanded_retained_features = retained_features.view(1, -1, 1, 1)
    mask = (segments == expanded_retained_features)
    mask = mask.sum(1) > 0
    result = segments * mask.float()

    # Apply the mask to the image using broadcasting
    masked_image = (image * mask.unsqueeze(0)).squeeze()

    # Fill the non-retained segments with the specified fill value
    masked_image = masked_image + (1 - mask.int()) * fill_val

    return masked_image