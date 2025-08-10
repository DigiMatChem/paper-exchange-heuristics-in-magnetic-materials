import os
import pytest

@pytest.fixture
def mcif_dir():
    """
    Fixture that returns the absolute path to the mcif directory.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mcif_path = os.path.join(base_dir, "data_retrieval_and_preprocessing_MAGNDATA", "mcifs")
    return mcif_path


@pytest.fixture
def mp_db_dir():
    """
    Fixture that returns the absolute path of the MP database json (stable and unique).
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mp_db_path = os.path.join(base_dir, "statistical_analysis", "MP", "data_and_plots", "df_stable_and_unique_MP_db.json")
    return mp_db_path