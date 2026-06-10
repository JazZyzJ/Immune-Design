from inverse_folding.evaluation.esmfold_runner import cache_key


def test_cache_key_replaces_path_separators():
    key = cache_key("O74409|strain_972_/_ATCC_24843|Uricase", "ACDE")

    assert "/" not in key
    assert "\\" not in key
    assert key.endswith("_f50b9a1db876")
