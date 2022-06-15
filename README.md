# 10 Toughest Places to Feed a Family

In the wake of Putin’s unprovoked attack on Ukraine, food prices are skyrocketing. 
This is compounding a crisis that already existed. In March, the UN’s Food Price Index reached its highest level ever,
up 30% on a year before. COVID related supply chain challenges, poor harvests and climate change have played a role 
in making 2022 a bad year for food.

Droughts in North and East Africa, a heatwave in India and conflicts in Ethiopia, Yemen and Somalia mean that Putin’s
invasion lit a match on an already primed fire. While advanced countries respond with social protection programs, 
vulnerable countries have limited fiscal reserves to respond.

New analysis from the ONE Campaign presents the toughest places to feed a family.


## Methodology
### Indicators

_Insufficient Food Consumption:_ An indicator using data from the WFP HungerMapLive which tracks the number of people who 
report having insufficient food consumption. This indicator reflects the latest data (up to end May). 
The data is presented as the share of population.

_Headline inflation:_ Changes in the consumer price index. This indicator reflects the latest data (up to end May).

_Wasting:_ This refers to "Prevalence of wasting, weight for height (% of children under 5)."

### How the index is produced

- The index considers only Low income and Lower middle income countries. Additionally, it considers Upper middle income countries with good data coverage.
 A total of 102 countries are included in the analysis.
- The latest available data (up to end of May 2022) is used for all indicators. 
- Since the data is on different scales, it is rescaled using a Quantile Transformer with a normal distribution.
A quantile transformer was chosen as it deals well with extreme outliers (a particular problem of inflation data). 
A uniform scale across indicators was important given that the index is created through a simple arithmetic mean of the indicators. 
- Missing data is imputed using a K Nearest Neighbours, using 10 neighbours and all the index indicators as features.
This method was chosen in order to avoid assigning arbitrary values to countries with missing data (like the regional
or income level mean of the missing indicator, for example). 
- Finally, for simplicity, the resulting index is a simple arithmetic mean (equal weights) of the indicators.

### Limitations
- By definition, the index assumes that the toughest places can only be found in lower income countries.
- Only the inflation and hunger indicators reflect countries' current/recent situation.
- Equal weights are used for simplicity but the relative importance of each indicator could be different in reality. 

## About this repository

This repository contains the data and scripts necessary to reproduce the analysis.
Python (>=3.10) is needed. Other required packages are listed in requirements.txt.

The repository includes the following sub-folders:
- `data`: contains the raw data used in the index and analysis
- `index`: scripts for the analysis
- `results`: an Excel file with the results of the analysis

