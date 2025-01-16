import torch
from torch import Tensor
from gymnasium.spaces import Box


def coord_to_index(act_space: Box, x, y):
    width = act_space.high[0] - act_space.low[0] + 1
    return (y - act_space.low[1]) * width + (x - act_space.low[0])


def coord_to_index_batch(act_space: Box, action0s_tensor: Tensor) -> Tensor:
    # Calculate the width of the action space
    width = act_space.high[0] - act_space.low[0] + 1

    # Extract x and y coordinates from the tensor
    x_coords = action0s_tensor[:, 0]
    y_coords = action0s_tensor[:, 1]

    # Compute the indices using tensor operations
    indices = (y_coords - act_space.low[1]) * width + (x_coords - act_space.low[0])

    return indices


def index_to_coord(act_space: Box, action_index):
    width = act_space.high[0] - act_space.low[0] + 1
    x = action_index % width + act_space.low[0]
    y = action_index // width + act_space.low[1]
    return x, y


def index_to_coord_batch(act_space: Box, action_indices: Tensor) -> Tensor:
    # Calculate the width of the action space
    width = act_space.high[0] - act_space.low[0] + 1

    # Ensure action_indices is a tensor and move it to the GPU
    if not isinstance(action_indices, torch.Tensor):
        action_indices = torch.tensor(action_indices, dtype=torch.long)
    action_indices = action_indices.cuda()

    # Compute x and y coordinates using tensor operations
    x_coords = (action_indices % width) + act_space.low[0]
    y_coords = (action_indices // width) + act_space.low[1]

    # Stack x and y coordinates into a single tensor of shape (batch_size, 2)
    coords = torch.stack((x_coords, y_coords), dim=1).squeeze()

    return coords
