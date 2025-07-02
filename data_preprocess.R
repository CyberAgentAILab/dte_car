rm(list = ls())
##------------------------------------------------------------------------------
## Libraries and source files
##------------------------------------------------------------------------------
library(haven) # For reading .dta files
library(dplyr) # For data manipulation
library(readr) # For writing CSV files
##------------------------------------------------------------------------------
## Load raw data and preprocess
##------------------------------------------------------------------------------
# To begin, you need to download the raw data from the OpenICPSR repository.
# Replication data for: The Impacts of Microfinance: Evidence from Joint-Liability Lending in Mongolia
# Authors: Orazio Attanasio; Britta Augsburg; Ralph De Haas; Emla Fitzsimons; Heike Harmgart
# Download the data from the following URL:
# https://www.openicpsr.org/openicpsr/project/113597/version/V1/view. 
# Set the working directory to the location where the data is stored
# Load raw data
income = read_dta("113597-V1/Analysis-files/data/Followup/Income.dta")
# Filter baseline survey data and select treatment and control groups
income_baseline = income %>%
  filter(followup==0) %>%
  filter(treatment==0| treatment==2) %>%
  mutate(W = (as.factor(1*(treatment==2))))
# Filter followup survey data and select treatment and control groups
income_followup = income %>%
  filter(followup==1) %>%
  filter(treatment==0| treatment==2) %>%
  mutate(W = (as.factor(1*(treatment==2)))) %>%
  select(-followup) # remove followup indicator
# Merge baseline and followup data with common identifiers and rename followup columns
df_income = merge(income_baseline, income_followup, by = c("rescode", "aimag",
                                                           "soum", "treatment"),
                                                           suffixes = c("", "_f"))
# Save the merged data to a CSV file
df_income %>% write_csv("microcredit.csv")
##------------------------------------------------------------------------------
