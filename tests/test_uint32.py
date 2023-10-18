import torch
import pytest

def test_gather_uint32():
    from tetranerf.utils.extension import gather_uint32

    vals = torch.rand((5)).cuda()
    indices = torch.randint(0, 5, (12,), dtype=torch.int32).cuda()
    res = gather_uint32(vals, 0, indices).cuda()
    torch.testing.assert_allclose(res, vals[indices.long()])
    
    with pytest.raises(Exception):
        # Does not support dim > 1
        vals = torch.rand((5, 3)).cuda()
        indices = torch.randint(0, 3, (5, 8), dtype=torch.int32).cuda()
        gather_uint32(vals, 0, indices.long())


def test_scatter_ema_uint32():
    from tetranerf.utils.extension import scatter_ema_uint32_

    torch.manual_seed(0)
    tensor = torch.rand((10)).cuda()
    indices = torch.tensor([4,3,5,8,2,1,0], dtype=torch.int32).cuda()
    vals = torch.rand((7,)).cuda()
    res = tensor.clone()
    decay = 0.5
    scatter_ema_uint32_(res, 0, indices, decay, vals)

    gt = torch.scatter(tensor, 0, indices.long(), tensor[indices.long()] * decay + (1-decay) * vals)
    torch.testing.assert_allclose(res, gt)
    
    with pytest.raises(Exception):
        # Does not support dim > 1
        tensor = torch.rand((5, 3)).cuda()
        indices = torch.randint(0, 3, (5, 8), dtype=torch.int32).cuda()
        vals = torch.rand((5, 8)).cuda()
        res = tensor.clone()
        scatter_ema_uint32_(res, 0, indices, decay, vals)