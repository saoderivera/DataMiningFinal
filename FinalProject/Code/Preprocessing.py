'''
Author: Lilian Sao de Rivera
Date : 04/22/2019
Machine Learning Final Project
'''

import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn import preprocessing
import warnings
warnings.filterwarnings("ignore")


path_files = ''

## Happiness DataSet
happiness = pd.read_csv(path_files+'2017.csv')
happiness_subset = happiness[["Country","Happiness.Score"]]


def create_scale(x):
    '''
    Create scale to bin the happiness score
    :param x:
    :return:
    '''
    scale = ((x >= 5.5) & (x < 6.5)) * 2 + (x >= 6.5) * 1 + ((x >= 4.5) & (x < 5.5)) * 3 + ((x >= 0) & (x < 4.5)) * 4

    return scale


mbin = happiness["Happiness.Score"]
happiness_subset['Happiness.Scale'] = mbin.apply(create_scale)

##ISO Dataset ( ISO CODES for each country in happiness
## There is the need to eliminate the first character

ISO = pd.read_csv(path_files+'ISOCODES.csv')
q=ISO.iloc[:,0]
m=q.apply(lambda x: x[1:len(x)])
ISO.iloc[:,0] =  m

y = pd.DataFrame(happiness["Country"])
y.columns = ["Country"]
result = pd.merge(y,ISO[["Country","Alpha-3-code"]], on='Country', how="left")
print(result[result["Alpha-3-code"].isna()])

## Rename countries from happiness dataset

y.iloc[13,0] = "United States of America"
y.iloc[18,0] = "United Kingdom of Great Britain and Northern Ireland"
y.iloc[22,0] = "Czechia"
y.iloc[32,0] = "Taiwan, Province of China[a]"
y.iloc[48,0] = "Russian Federation"
y.iloc[54,0] = "Korea, Republic of"
y.iloc[55,0] = "Moldova, Republic of"
y.iloc[57,0] = "Bolivia (Plurinational State of)"
y.iloc[60,0] = "Cyprus"
y.iloc[70,0] = "Hong Kong"
y.iloc[81,0] = "Venezuela (Bolivarian Republic of)"
y.iloc[102,0] = "Palestine, State of"
y.iloc[107,0] = "Iran (Islamic Republic of)"
y.iloc[123,0] = "Congo"
y.iloc[125,0] = "Congo, Democratic Republic of the"
y.iloc[127,0] = "Cote d'Ivoire"
y.iloc[151,0] = "Syrian Arab Republic"
y.iloc[152,0] = "Tanzania, United Republic of"

result = pd.merge(y,ISO, on='Country', how="left")
print('Countries without ISO:')
print(result[result["Alpha-3-code"].isna()])

result_final=result[["Country","Alpha-3-code"]]
result_final.columns = ["Country","Country_code"]
print('Final ISO for happiness')
print(result_final.head(5))

happiness_subset["Country"]= y["Country"]
happiness_subset =  pd.merge(happiness_subset, result_final, on="Country", how='left')
print(happiness_subset.head())

## Now lets process GDP

GDP = pd.read_csv(path_files+'GDP.csv', skiprows=4)

GDP_subset = GDP[["Country Name","Country Code","2017"]]
GDP_index_null = GDP_subset[GDP_subset["2017"].isnull()].index

mm = GDP.iloc[254,4:]
mm = pd.DataFrame(mm)
mm1=mm[254].fillna(method='ffill')

GDP.iloc[254,4:] = mm1


def GDP_fill_null_values():
    for i in GDP_index_null:
        row_GDP = pd.DataFrame(GDP.iloc[i, 4:])
        row_GDP1 = row_GDP[i].fillna(method='ffill')
        GDP.iloc[i, 4:] = row_GDP1


GDP_fill_null_values()

GDP_subset = GDP[["Country Code", "2017"]]

GDP_index_null = GDP_subset[GDP_subset["2017"].isnull()].index

##Values that do not have any information in the original file

print(GDP_subset[GDP_subset["2017"].isnull()])

GDP_subset = GDP_subset[GDP_subset["2017"].notnull()]

vcolumns = ["Country_code", "GDP"]
GDP_subset.columns = vcolumns

print(GDP_subset.head())

## Now lets process GINI

GINI = pd.read_csv(path_files+'GINI.csv', skiprows=4)
GINI_subset = GINI[["Country Name","Country Code","2017"]]
GINI_index_null = GINI_subset[GINI_subset["2017"].isnull()].index


def GINI_fill_null_values():
    for i in GINI_index_null:
        row_GINI = pd.DataFrame(GINI.iloc[i, 4:])
        row_GINI1 = row_GINI[i].fillna(method='ffill')
        GINI.iloc[i, 4:] = row_GINI1


GINI_fill_null_values()

GINI_subset = GINI[["Country Code", "2017"]]
GINI_subset.columns = ["Country_code", "GINI"]
GINI_index_null = GINI_subset[GINI_subset["GINI"].isnull()].index

GINI_subset = GINI_subset[GINI_subset["GINI"].notnull()]
print(GINI_subset.head())

### Now Lets process Governace indicators

def creates_colnames(estimate_cols):
    columns = []
    len_cols = estimate_cols.shape[1]
    for i in range((len_cols)):

        row1 = str(estimate_cols.iloc[1, i])
        row0 = str(estimate_cols.iloc[0, i])

        if pd.isnull(row0) or row0 == 'nan':
            row0 = ""
        if pd.isnull(row1):
            row1 = ""
        if row0 == "":
            final_row_name = row1
        else:
            final_row_name = row1 + "_" + row0
        columns.append(final_row_name)

    return (columns)


voice_columns = pd.read_excel(path_files + 'wgidataset.xlsx', sheet_name="VoiceandAccountability", skiprows=12, nrows=2,
                              names=None)
voice_colnames = creates_colnames(voice_columns)

political_columns = pd.read_excel(path_files + 'wgidataset.xlsx', sheet_name="PoliticalStabilityNoViolence",
                                  skiprows=12, nrows=2, names=None)
political_colnames = creates_colnames(voice_columns)

effectiveness_columns = pd.read_excel(path_files + 'wgidataset.xlsx', sheet_name="GovernmentEffectiveness", skiprows=12,
                                      nrows=2, names=None)
effectiveness_colnames = creates_colnames(voice_columns)

regulatory_columns = pd.read_excel(path_files + 'wgidataset.xlsx', sheet_name="RegulatoryQuality", skiprows=12, nrows=2,
                                   names=None)
regulatory_colnames = creates_colnames(voice_columns)

ruleoflaw_columns = pd.read_excel(path_files + 'wgidataset.xlsx', sheet_name="RuleofLaw", skiprows=12, nrows=2,
                                  names=None)
ruleoflaw_colnames = creates_colnames(voice_columns)

corruption_columns = pd.read_excel(path_files + 'wgidataset.xlsx', sheet_name="ControlofCorruption", skiprows=12,
                                   nrows=2, names=None)
corruption_colnames = creates_colnames(voice_columns)

####

voice = pd.read_excel(path_files+'wgidataset.xlsx', sheet_name="VoiceandAccountability" ,skiprows=14, names=voice_colnames)
political = pd.read_excel(path_files+'wgidataset.xlsx', sheet_name="PoliticalStabilityNoViolence" ,skiprows=14, names=political_colnames)
effectiveness = pd.read_excel(path_files+'wgidataset.xlsx', sheet_name="GovernmentEffectiveness" ,skiprows=14, names=effectiveness_colnames)
regulatory = pd.read_excel(path_files+'wgidataset.xlsx', sheet_name="RegulatoryQuality" ,skiprows=14, names=regulatory_colnames)
ruleoflaw = pd.read_excel(path_files+'wgidataset.xlsx', sheet_name="RuleofLaw" ,skiprows=14, names=ruleoflaw_colnames)
corruption = pd.read_excel(path_files+'wgidataset.xlsx', sheet_name="ControlofCorruption" ,skiprows=14, names=corruption_colnames)


####

# Voice
col_names = ["Country/Territory","WBCode"]
col_names1= [col for col in voice.columns if 'Estimate' in col]
col_names.extend(col_names1)
voice_subset = voice[col_names]
#political
col_names = ["Country/Territory","WBCode"]
col_names1= [col for col in political.columns if 'Estimate' in col]
col_names.extend(col_names1)
political_subset = political[col_names]
#effectiveness
col_names = ["Country/Territory","WBCode"]
col_names1= [col for col in effectiveness.columns if 'Estimate' in col]
col_names.extend(col_names1)
effectiveness_subset = effectiveness[col_names]
#regulatory
col_names = ["Country/Territory","WBCode"]
col_names1= [col for col in regulatory.columns if 'Estimate' in col]
col_names.extend(col_names1)
regulatory_subset = regulatory[col_names]
#ruleoflaw
col_names = ["Country/Territory","WBCode"]
col_names1= [col for col in ruleoflaw.columns if 'Estimate' in col]
col_names.extend(col_names1)
ruleoflaw_subset = ruleoflaw[col_names]
#corruption
col_names = ["Country/Territory","WBCode"]
col_names1= [col for col in corruption.columns if 'Estimate' in col]
col_names.extend(col_names1)
corruption_subset = corruption[col_names]

###

voice_null_index = voice_subset[voice_subset["Estimate_2017"].isnull()].index
political_null_index = political_subset[voice_subset["Estimate_2017"].isnull()].index
effectiveness_null_index = effectiveness_subset[voice_subset["Estimate_2017"].isnull()].index
regulatory_null_index = regulatory_subset[voice_subset["Estimate_2017"].isnull()].index
ruleoflaw_null_index = ruleoflaw_subset[voice_subset["Estimate_2017"].isnull()].index
corruption_null_index = corruption_subset[voice_subset["Estimate_2017"].isnull()].index

###

def ESTIMATE_fill_null_values(index_null, EST_dataset):
    for i in index_null:
        # if data_v == 1:
        row_EST = pd.DataFrame(EST_dataset.iloc[i, 2:])
        row_EST1 = row_EST[i].fillna(method='ffill')
        # if data_v == 1:
        EST_dataset.iloc[i, 2:] = row_EST1


x_data = voice_subset.copy()
ESTIMATE_fill_null_values(voice_null_index, x_data)
voice_subset = x_data.copy()

x_data = political_subset.copy()
ESTIMATE_fill_null_values(political_null_index, x_data)
political_subset = x_data.copy()

x_data = effectiveness_subset.copy()
ESTIMATE_fill_null_values(effectiveness_null_index, x_data)
effectiveness_subset = x_data.copy()

x_data = regulatory_subset.copy()
ESTIMATE_fill_null_values(regulatory_null_index, x_data)
regulatory_subset = x_data.copy()

x_data = ruleoflaw_subset.copy()
ESTIMATE_fill_null_values(ruleoflaw_null_index, x_data)
ruleoflaw_subset = x_data.copy()

x_data = corruption_subset.copy()
ESTIMATE_fill_null_values(corruption_null_index, x_data)
corruption_subset = x_data.copy()

# Joining all the values
#Voide
next_subset= voice_subset[["WBCode","Estimate_2017"]]
next_subset.columns = ["Country_code","VoiceandAccountability"]
final_dataset = next_subset.copy()
#PoliticalStabilityNoViolence
next_subset= political_subset[["WBCode","Estimate_2017"]]
next_subset.columns = ["Country_code","PoliticalStabilityNoViolence"]
final_dataset = pd.merge(final_dataset, next_subset, on="Country_code", how="inner")
#Effectiveness
next_subset= effectiveness_subset[["WBCode","Estimate_2017"]]
next_subset.columns = ["Country_code","GovermentEffectiveness"]
final_dataset = pd.merge(final_dataset, next_subset, on="Country_code", how="inner")
#Regulatory
next_subset= regulatory_subset[["WBCode","Estimate_2017"]]
next_subset.columns = ["Country_code","RegulatoryQuality"]
final_dataset = pd.merge(final_dataset, next_subset, on="Country_code", how="inner")
#Ruleoflaw
next_subset= ruleoflaw_subset[["WBCode","Estimate_2017"]]
next_subset.columns = ["Country_code","RuleofLaw"]
final_dataset = pd.merge(final_dataset, next_subset, on="Country_code", how="inner")
#ControlofCorruption
next_subset= corruption_subset[["WBCode","Estimate_2017"]]
next_subset.columns = ["Country_code","ControlofCorruption"]
final_dataset = pd.merge(final_dataset, next_subset, on="Country_code", how="inner")

print(final_dataset.head())

### Now Happiness, GDP, GINI, Governance factors will be joined in one datasets to be processed

final_happiness = happiness_subset.copy()
final_happiness = pd.merge(final_happiness,GDP_subset, on="Country_code")
final_happiness = pd.merge(final_happiness,GINI_subset, on="Country_code")
final_happiness = pd.merge(final_happiness,final_dataset, on="Country_code")
#print(final_happiness)

print(final_happiness.info())

## Let's normalize the numeric data
## Get all the numerical data

numerical_data = final_happiness[["Happiness.Score","GDP","GINI","VoiceandAccountability","PoliticalStabilityNoViolence","GovermentEffectiveness","RegulatoryQuality","RuleofLaw","ControlofCorruption" ]]

## Get column names first

happ_columns = numerical_data.columns

# Create the Scaler object
scaler = preprocessing.StandardScaler()
# Fit your data on the scaler object
scaled_happ = scaler.fit_transform(numerical_data)
scaled_happ = pd.DataFrame(scaled_happ, columns=happ_columns)

str_data = final_happiness[['Country','Country_code','Happiness.Scale']]
ff_happiness = pd.concat([str_data,scaled_happ], axis=1)


## Save the file to be used with graphics and machine learning algorithms.

ff_happiness.to_csv(path_files+"final_happiness_dataset.csv", index=False)