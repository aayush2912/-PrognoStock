import numpy as np
import pandas as pd
import pytest
from ML_Models import mse, model, y_pred, X_test, gb_model, mse_gb, corr, df_predictTest
from ML_Models import y_test_no_outliers, y_pred_gb, residuals
from ML_Models import IQR, Q1, Q3

@pytest.mark.mpl_image_compare(remove_text=True, savefig_kwargs={'dpi': 300})
def test_df_not_empty():
    assert not y_pred.empty, "The DataFrame 'tech_df' is empty"
    assert not X_test.empty, "The DataFrame 'fin_df' is empty"
    assert not gb_model.empty, "The DataFrame 'energy_df' is empty"
    assert not corr.empty, "The DataFrame 'statistics_df' is empty"
    assert not df_predictTest.empty, "The DataFrame 'statistics_df' is empty"

@pytest.mark.mpl_image_compare(remove_text=True, savefig_kwargs={'dpi': 300})
def test_preds():
    assert len(y_pred) == len(X_test), "Incorrect length of predictions"
    assert len(y_pred) == len(df_predictTest), "Incorrect length of predictions"

@pytest.mark.mpl_image_compare(remove_text=True, savefig_kwargs={'dpi': 300})
def test_residuals():
    assert y_test_no_outliers - y_pred_gb == residuals

@pytest.mark.mpl_image_compare(remove_text=True, savefig_kwargs={'dpi': 300})
def test_iqr():
    assert Q3 - Q1 == IQR
