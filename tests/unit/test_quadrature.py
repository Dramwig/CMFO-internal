import torch

from cmfo.data.fields import pack_geometry
from cmfo.data.quadrature import (
    integrate,
    interpolate_field,
    make_1d_coordinates,
    make_2d_coordinates,
    normalized_field_l2,
)


def test_regular_2d_quadrature_converges_for_quadratic_function() -> None:
    errors = []
    exact = torch.tensor([[2.0 / 3.0]])
    for count in (4, 16, 64):
        coords = make_2d_coordinates(count, "regular")
        geometry = pack_geometry([coords])
        values = (coords[:, 0].square() + coords[:, 1].square()).view(1, count, 1)
        estimate = integrate(values, geometry)
        errors.append(float((estimate - exact).abs().item()))
    assert errors[2] < errors[1] < errors[0]


def test_random_and_jittered_layouts_have_valid_domains() -> None:
    generator = torch.Generator().manual_seed(13)
    for layout in ("regular", "jittered", "random"):
        coords_1d = make_1d_coordinates(7, layout, generator)
        coords_2d = make_2d_coordinates(16, layout, generator)
        assert coords_1d.shape == (7, 1)
        assert coords_2d.shape == (16, 2)
        assert torch.all((coords_1d >= 0) & (coords_1d <= 1))
        assert torch.all((coords_2d >= 0) & (coords_2d <= 1))


def test_interpolation_preserves_constant_field() -> None:
    source = pack_geometry([make_2d_coordinates(16, "regular")])
    target = pack_geometry([make_2d_coordinates(25, "regular")])
    values = torch.full((1, 16, 3), 2.5)
    interpolated = interpolate_field(values, source, target)
    torch.testing.assert_close(interpolated, torch.full((1, 25, 3), 2.5))


def test_normalized_field_l2_is_zero_for_equal_fields() -> None:
    geometry = pack_geometry([make_1d_coordinates(5)])
    values = torch.randn(1, 5, 4)
    error = normalized_field_l2(values, values.clone(), geometry.mask)
    torch.testing.assert_close(error, torch.zeros_like(error))
