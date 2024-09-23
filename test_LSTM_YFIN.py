import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt
from LSTM_YFIN import predictions, Y_test as test_data, residuals, model

@pytest.mark.mpl_image_compare(remove_text=True, savefig_kwargs={'dpi': 300})
def test_lstm_model():
    plt.switch_backend('Agg')
    assert predictions.shape[0] == test_data.shape[0]
    assert residuals.shape[0] == test_data.shape[0]
