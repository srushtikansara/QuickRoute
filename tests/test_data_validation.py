import io

import pandas as pd
import pytest

from src.data_generator import generate_dataset
from src.data_validation import (DataValidationError, clean_dataframe, load_and_validate_csv,
                                 validate_dataframe)


def good_df():
    return generate_dataset(6, seed=4)


def test_generated_dataset_is_valid_and_reproducible():
    assert validate_dataframe(good_df()) == []
    assert good_df().equals(generate_dataset(6, seed=4))
    assert not good_df().equals(generate_dataset(6, seed=5))
    df = good_df()
    assert df["is_depot"].sum() == 1 and bool(df.iloc[0]["is_depot"])
    assert df["priority"].between(1, 5).all()


def test_missing_values_are_reported():
    df = good_df()
    df.loc[2, "x_coordinate"] = None
    assert any("missing" in p for p in validate_dataframe(df))


def test_duplicate_ids_are_reported():
    df = good_df()
    df.loc[2, "delivery_id"] = df.loc[1, "delivery_id"]
    assert any("Duplicate delivery_id" in p for p in validate_dataframe(df))


def test_invalid_priority_is_reported():
    for bad in (0, 6, 2.5):
        df = good_df()
        df["priority"] = df["priority"].astype(float)
        df.loc[1, "priority"] = bad
        assert any("Priority" in p for p in validate_dataframe(df)), bad


def test_invalid_coordinates_are_reported():
    df = good_df()
    df["x_coordinate"] = df["x_coordinate"].astype(object)
    df.loc[1, "x_coordinate"] = "abc"
    assert any("non-numeric" in p for p in validate_dataframe(df))
    df = good_df()
    df.loc[1, "y_coordinate"] = float("inf")
    assert any("infinite" in p or "outside" in p for p in validate_dataframe(df))
    df = good_df()
    df.loc[1, "y_coordinate"] = 1e9
    assert any("outside" in p for p in validate_dataframe(df))


def test_missing_column_and_depot_problems():
    df = good_df().drop(columns=["priority"])
    assert "Missing required column" in validate_dataframe(df)[0]
    df = good_df()
    df["is_depot"] = False
    assert any("depot" in p for p in validate_dataframe(df))
    df = good_df()
    df["is_depot"] = True
    assert any("depot" in p for p in validate_dataframe(df))


def test_empty_dataframe():
    assert validate_dataframe(pd.DataFrame()) == ["The dataset is empty."]


def test_clean_dataframe_puts_depot_first_and_raises_on_invalid():
    df = good_df().iloc[::-1].reset_index(drop=True)      # depot is now last
    cleaned = clean_dataframe(df)
    assert bool(cleaned.iloc[0]["is_depot"])
    bad = good_df()
    bad.loc[1, "priority"] = 9
    with pytest.raises(DataValidationError, match="Priority"):
        clean_dataframe(bad)


def test_csv_round_trip_and_error_messages():
    csv_text = good_df().to_csv(index=False)
    assert len(load_and_validate_csv(io.StringIO(csv_text))) == 7
    with pytest.raises(DataValidationError, match="empty"):
        load_and_validate_csv(io.StringIO(""))
    with pytest.raises(DataValidationError, match="Missing required column"):
        load_and_validate_csv(io.StringIO("a,b\n1,2\n"))
    broken = csv_text.replace("Customer 01", "")
    with pytest.raises(DataValidationError, match="missing"):
        load_and_validate_csv(io.StringIO(broken))


def test_optional_package_size_column_may_be_absent():
    df = good_df().drop(columns=["package_size"])
    assert "package_size" in clean_dataframe(df).columns


def test_generator_rejects_bad_arguments():
    with pytest.raises(ValueError):
        generate_dataset(0)
    with pytest.raises(ValueError):
        generate_dataset(5, speed=0)
