import torch

from cmfo.data.fields import (
    pack_fields,
    pack_queries,
    random_valid_permutation,
)


def test_pack_fields_supports_variable_lengths_and_normalized_weights() -> None:
    values = [torch.tensor([1, 2, 3]), torch.tensor([4, 5])]
    coords = [
        torch.tensor([[0.1], [0.5], [0.9]]),
        torch.tensor([[0.25], [0.75]]),
    ]
    field = pack_fields(values, coords)
    assert field.values.shape == (2, 3)
    assert field.mask.tolist() == [[True, True, True], [True, True, False]]
    torch.testing.assert_close(field.weights.sum(dim=1), torch.ones(2))
    assert field.lengths.tolist() == [3, 2]


def test_field_permutation_keeps_values_geometry_and_weights_aligned() -> None:
    values = [torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])]
    coords = [torch.tensor([[0.1], [0.5], [0.9]])]
    weights = [torch.tensor([0.2, 0.3, 0.5])]
    field = pack_fields(values, coords, weights)
    permutation = torch.tensor([[2, 0, 1]])
    permuted = field.permuted(permutation)
    assert permuted.values[0, :, 0].tolist() == [5.0, 1.0, 3.0]
    assert permuted.coords[0, :, 0].tolist() == field.coords[0, :, 0][permutation[0]].tolist()
    torch.testing.assert_close(permuted.weights, field.weights.gather(1, permutation))


def test_random_valid_permutation_leaves_padding_after_valid_nodes() -> None:
    mask = torch.tensor([[True, True, True, False], [True, True, False, False]])
    permutation = random_valid_permutation(mask, torch.Generator().manual_seed(7))
    assert sorted(permutation[0, :3].tolist()) == [0, 1, 2]
    assert permutation[0, 3].item() == 3
    assert sorted(permutation[1, :2].tolist()) == [0, 1]
    assert permutation[1, 2:].tolist() == [2, 3]


def test_query_packing_preserves_targets() -> None:
    queries = pack_queries(
        [torch.tensor([[0.1, 0.2], [0.3, 0.4]]), torch.tensor([[0.5, 0.6]])],
        [torch.tensor([0.0, 1.0]), torch.tensor([1.0])],
    )
    assert queries.targets.tolist() == [[0.0, 1.0], [1.0, 0.0]]
    assert queries.mask.tolist() == [[True, True], [True, False]]
