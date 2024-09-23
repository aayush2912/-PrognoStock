import numpy as np
import pandas as pd
import pytest
from MLAnalysis import df, grouped_df, tech_df, fin_df, energy_df, statistics_df
from MLAnalysis import forecast as forecast_df

def test_df_not_empty():
    assert not df.empty, "The DataFrame 'df' is empty"
    assert not grouped_df.empty, "The DataFrame 'grouped_df' is empty"
    assert not tech_df.empty, "The DataFrame 'tech_df' is empty"
    assert not fin_df.empty, "The DataFrame 'fin_df' is empty"
    assert not energy_df.empty, "The DataFrame 'energy_df' is empty"
    assert not statistics_df.empty, "The DataFrame 'statistics_df' is empty"

def test_sector_assignment():
    # Define expected sectors based on the logic in the code
    expected_sectors = {
        'apple': 'Technology',
        'microsoft': 'Technology',
        'google': 'Technology',
        'jpmc': 'Finance',
        'boa': 'Finance',
        'wfc': 'Finance',
        'jnj': 'Healthcare',
        'pfizer': 'Healthcare',
        'merck': 'Healthcare',
        'exxon': 'Energy',
        'chevron': 'Energy',
        'facebook': 'Communication',
        'verizon': 'Communication',
        'atnt': 'Communication'
    }
    
    unique_companies = df[['company_name', 'Sector']].drop_duplicates()
    
    for _, row in unique_companies.iterrows():
        company_name = row['company_name']
        actual_sector = row['Sector']
        expected_sector = expected_sectors.get(company_name.lower())
        assert actual_sector == expected_sector, f"Sector for {company_name} is {actual_sector}, expected {expected_sector}"

def test_grouped_dfs():

    assert isinstance(grouped_df.index, pd.DatetimeIndex), "Index is not a DateTimeIndex for 'grouped_df' DataFrame"
    assert isinstance(tech_df.index, pd.DatetimeIndex), "Index is not a DateTimeIndex for 'tech_df' DataFrame"
    assert isinstance(fin_df.index, pd.DatetimeIndex), "Index is not a DateTimeIndex for 'fin_df' DataFrame"
    assert isinstance(energy_df.index, pd.DatetimeIndex), "Index is not a DateTimeIndex for 'energy_df' DataFrame"
    
    assert grouped_df['Adj Close'].equals(df.groupby(['Sector', 'Date'])['Adj Close'].mean()), "Values of 'grouped_df' do not match original DataFrame"
    assert tech_df['Adj Close'].equals(df[df['Sector'] == 'Technology']['Adj Close']), "Values of 'tech_df' do not match original DataFrame"
    assert fin_df['Adj Close'].equals(df[df['Sector'] == 'Finance']['Adj Close']), "Values of 'fin_df' do not match original DataFrame"
    assert energy_df['Adj Close'].equals(df[df['Sector'] == 'Energy']['Adj Close']), "Values of 'energy_df' do not match original DataFrame"

def test_forecast_df():
    assert not forecast_df.empty, "The DataFrame 'forecast_df' is empty"
    assert isinstance(forecast_df.index, pd.DatetimeIndex), "Index is not a DateTimeIndex for 'forecast_df' DataFrame"
    expected_columns = ['forecast']
    assert set(forecast_df.columns) == set(expected_columns), "Columns of 'forecast_df' do not match expected columns"
    assert forecast_df.applymap(lambda x: isinstance(x, (int, float))).all().all(), "Non-numeric values found in 'forecast_df' DataFrame"
    assert len(forecast_df) == len(test), "Length of 'forecast_df' does not match length of test data"
