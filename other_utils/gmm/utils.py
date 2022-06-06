import torch

def calculate_matmul_n_times(n_components, mat_a, mat_b):
    """
    Calculate matrix product of two matrics with mat_a[0] >= mat_b[0].
    Bypasses torch.matmul to reduce memory footprint.
    args:
        mat_a:      torch.Tensor (n, k, 1, d)
        mat_b:      torch.Tensor (1, k, d, d)
    """
    res = torch.zeros(mat_a.shape).to(mat_a.device)
    
    for i in range(n_components):
        mat_a_i = mat_a[:, i, :, :].squeeze(-2)
        mat_b_i = mat_b[0, i, :, :].squeeze()
        # by bug
        if len(mat_b_i.size()) == 0:
            mat_b_i = mat_b_i.reshape(1, 1)
        res[:, i, :, :] = mat_a_i.mm(mat_b_i).unsqueeze(1)
    
    return res


def calculate_matmul(mat_a, mat_b):
    """
    Calculate matrix product of two matrics with mat_a[0] >= mat_b[0].
    Bypasses torch.matmul to reduce memory footprint.
    args:
        mat_a:      torch.Tensor (n, k, 1, d)
        mat_b:      torch.Tensor (n, k, d, 1)
    """
    assert mat_a.shape[-2] == 1 and mat_b.shape[-1] == 1
    return torch.sum(mat_a.squeeze(-2) * mat_b.squeeze(-1), dim=2, keepdim=True)
