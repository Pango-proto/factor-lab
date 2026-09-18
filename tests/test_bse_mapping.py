import polars as pl

from factor_matrix.normalize import canonicalize_bse_codes


def test_old_bse_code_maps_to_stable_new_asset_id() -> None:
    frame = pl.DataFrame({"ts_code": ["839680.BJ", "600000.SH"], "value": [1, 2]})
    mapping = pl.DataFrame(
        {"o_code": ["839680.BJ"], "n_code": ["920680.BJ"], "name": ["广道数字"], "list_date": ["20211115"]}
    )
    result = canonicalize_bse_codes(frame, mapping)
    assert result.get_column("ts_code").to_list() == ["920680.BJ", "600000.SH"]
