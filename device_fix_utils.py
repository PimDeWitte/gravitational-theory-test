"""
Utility functions to ensure device consistency across the codebase.
"""
import torch
from typing import Union, Optional


def ensure_tensor_on_device(
    value: Union[float, torch.Tensor], 
    device: torch.device, 
    dtype: torch.dtype
) -> torch.Tensor:
    """Convert any value to a tensor on the specified device with specified dtype."""
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    else:
        return torch.tensor(value, device=device, dtype=dtype)


def ensure_same_device(*tensors) -> list:
    """Ensure all tensors are on the same device as the first tensor."""
    if not tensors:
        return []
    
    # Get device from first tensor
    device = None
    dtype = None
    for t in tensors:
        if isinstance(t, torch.Tensor):
            device = t.device
            dtype = t.dtype
            break
    
    if device is None:
        # No tensors found, return as is
        return list(tensors)
    
    # Convert all to same device
    result = []
    for t in tensors:
        if isinstance(t, torch.Tensor):
            result.append(t.to(device=device, dtype=dtype))
        else:
            result.append(torch.tensor(t, device=device, dtype=dtype))
    
    return result


def safe_stack(tensors: list, dim: int = 0, device: Optional[torch.device] = None) -> torch.Tensor:
    """Stack tensors ensuring they're all on the same device."""
    if not tensors:
        raise ValueError("Cannot stack empty list of tensors")
    
    # Determine target device
    if device is None:
        for t in tensors:
            if isinstance(t, torch.Tensor):
                device = t.device
                break
    
    if device is None:
        device = torch.device('cpu')
    
    # Convert all to same device
    tensors_on_device = []
    for t in tensors:
        if isinstance(t, torch.Tensor):
            tensors_on_device.append(t.to(device))
        else:
            tensors_on_device.append(torch.tensor(t, device=device))
    
    return torch.stack(tensors_on_device, dim=dim) 