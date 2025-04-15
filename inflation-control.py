#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
import numpy as np
import kagglehub

# Download latest version
path = kagglehub.dataset_download("kalilurrahman/berkshire-hathaway-stock-latest-and-updated")

print("Path to dataset files:", path)

# In[9]:


berk_hist = pd.read_csv("./data/BekshireHathaway_stock_history.csv")
cpi = pd.read_excel("./data/cpi.xlsx",sheet_name="BLS Data Series")


# In[31]:


berk_hist.sort_values("Date",ascending=False).head()


# In[10]:


cpi.head()


# In[11]:


# Reshape the cpi dataframe so that each row represents a month and year with its CPI value
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
melted = cpi.melt(id_vars="Year", value_vars=months, var_name="Month", value_name="CPI")

# Map month abbreviations to their corresponding two-digit numbers
month_map = {
    "Jan": "01", "Feb": "02", "Mar": "03", "Apr": "04",
    "May": "05", "Jun": "06", "Jul": "07", "Aug": "08",
    "Sep": "09", "Oct": "10", "Nov": "11", "Dec": "12"
}
melted["Month_Num"] = melted["Month"].map(month_map)

# Create a date column in the format YYYY-MM
melted["Date"] = melted["Year"].astype(str) + "-" + melted["Month_Num"]

# Create the new dataframe with just the date and CPI columns – drop rows with missing CPI values if any
cpi_by_date = melted[["Date", "CPI"]].dropna().reset_index(drop=True)
cpi_by_date.head()


# In[ ]:


DOLLAR_IN_TERMS_DATE = cpi_by_date["Date"].max() # this means that we are having all our dollars in terms of the most recent cpi data we have
CPI_OF_REFERENCE = cpi_by_date[cpi_by_date["Date"] == DOLLAR_IN_TERMS_DATE]["CPI"].tolist()[0]
CPI_OF_REFERENCE

cpi_by_date["dollar_coeff"] = CPI_OF_REFERENCE / cpi_by_date["CPI"] # gets coefficients that is value at DOLLAR_IN_TERMS_DATE over the value of the dollar at the given time in the df



# In[32]:


# Extract the Year-Month from the daily date in berk_hist
berk_hist['YearMonth'] = berk_hist['Date'].str[:7]

# Map the corresponding dollar coefficient by matching YearMonth to the Date in cpi_by_date
berk_hist['dollar_coeff'] = berk_hist['YearMonth'].map(cpi_by_date.set_index('Date')['dollar_coeff'])

# Create a new dataframe with inflation adjusted columns
inflation_adjusted_berk_hist = pd.DataFrame({
    'Date': berk_hist['Date'],
    'Open_adjusted': berk_hist['Open'] * berk_hist['dollar_coeff'],
    'High_adjusted': berk_hist['High'] * berk_hist['dollar_coeff'],
    'Low_adjusted': berk_hist['Low'] * berk_hist['dollar_coeff'],
    'Close_adjusted': berk_hist['Close'] * berk_hist['dollar_coeff']
})

inflation_adjusted_berk_hist.sort_values("Date",ascending=False).head()


# In[33]:


inflation_adjusted_berk_hist.to_csv("./data/inflation_adjusted_berkshire_stocks.csv")

