
import torch

def test_barycentrics():
    barycentrics = torch.rand((5, 3)) * 0.25

    vertices = torch.randn((5, 4, 3))
    full_barycentrics = torch.cat([1 - barycentrics.sum(dim=-1, keepdim=True), barycentrics], dim=-1)
    points = (vertices * full_barycentrics.unsqueeze(-1)).sum(-2)


    t_mat = (vertices[..., 1:, :] - vertices[..., :1, :]).transpose(-1, -2)
    computed_barycentrics = torch.linalg.solve(t_mat, points - vertices[..., 0, :])
    torch.testing.assert_allclose(computed_barycentrics, barycentrics)
    
def test_barycentrics_grad():
    gt_barycentrics = torch.rand((5, 3)) * 0.25
    vertices = torch.randn((5, 4, 3))
    vertices.detach_().requires_grad_(True)
    full_barycentrics = torch.cat([1 - gt_barycentrics.sum(dim=-1, keepdim=True), gt_barycentrics], dim=-1)
    points = (vertices * full_barycentrics.unsqueeze(-1)).sum(-2)
    points.detach_().requires_grad_(True)

    t_mat = (vertices[..., 1:, :] - vertices[..., :1, :])
    barycentrics = torch.linalg.solve(t_mat.transpose(-1, -2), points - vertices[..., 0, :])
    torch.testing.assert_allclose(barycentrics, gt_barycentrics)

    # Compute grads automatically
    barycentrics.retain_grad()
    comb = torch.randn((5, 3))
    (barycentrics * comb).sum().backward()

    # Compute grads manually
    t_mat = (vertices[..., 1:, :] - vertices[..., :1, :])
    grad_barycentrics = barycentrics.grad
    m_vec = torch.linalg.solve(t_mat, grad_barycentrics)
    points_grad = m_vec
    torch.testing.assert_allclose(points_grad, points.grad)

    grad_vertices = -(full_barycentrics.unsqueeze(-1) * m_vec.unsqueeze(-2))
    torch.testing.assert_allclose(grad_vertices, vertices.grad)

def test_barycentrics_util():
    from tetranerf.utils.extension import add_barycentrics_grad
    
    gt_barycentrics = torch.rand((5, 3)) * 0.25
    vertices = torch.randn((5, 4, 3))
    vertices.detach_().requires_grad_(True)
    full_barycentrics = torch.cat([1 - gt_barycentrics.sum(dim=-1, keepdim=True), gt_barycentrics], dim=-1)
    points = (vertices * full_barycentrics.unsqueeze(-1)).sum(-2)
    points.detach_().requires_grad_(True)

    t_mat = (vertices[..., 1:, :] - vertices[..., :1, :])
    barycentrics = torch.linalg.solve(t_mat.transpose(-1, -2), points - vertices[..., 0, :])
    torch.testing.assert_allclose(barycentrics, gt_barycentrics)

    # Compute grads automatically
    barycentrics.retain_grad()
    comb = torch.randn((5, 3))
    (barycentrics * comb).sum().backward()

    # Compute grads manually
    barycentrics2 = barycentrics.detach().requires_grad_(True)
    vertices2 = vertices.detach().requires_grad_(True)
    points2 = points.detach().requires_grad_(True)
    barycentrics2 = add_barycentrics_grad(barycentrics2, vertices2, points2)
    (barycentrics2 * comb).sum().backward()
    torch.testing.assert_allclose(points2.grad, points.grad)
    torch.testing.assert_allclose(vertices2.grad, vertices.grad)

    