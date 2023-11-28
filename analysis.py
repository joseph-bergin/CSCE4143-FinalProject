import pandas as pd

# Load data
teams = pd.read_csv('MTeams.csv')
seeds = pd.read_csv('MNCAATourneySeeds.csv')
detailed_results = pd.read_csv('MNCAATourneyDetailedResults.csv')

# Merge data
merged_data = pd.merge(detailed_results, teams, left_on=['WTeamID'], right_on=['TeamID'], how='left', suffixes=('_W', ''))
merged_data = pd.merge(merged_data, seeds, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='left', suffixes=('_W', '_Seed'))
merged_data = pd.merge(merged_data, teams, left_on=['LTeamID'], right_on=['TeamID'], how='left', suffixes=('_L', ''))
merged_data = pd.merge(merged_data, seeds, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], how='left', suffixes=('_L', '_Seed'))

# Drop unnecessary columns
columns_to_drop = ['TeamID_W', 'TeamID_L','TeamID_Seed']
merged_data = merged_data.drop(columns=columns_to_drop)

# Rename columns
merged_data = merged_data.rename(columns={'TeamName': 'LTeamName', 'FirstD1Season': 'LFirstD1Season', 'LastD1Season': 'LLastD1Season',
                                          'Seed_Seed': 'LSeed', 'TeamName_L': 'WTeamName', 'FirstD1Season_L': 'WFirstD1Season', 'LastD1Season_L': 'WLastD1Season',
                                          'Seed_L': 'WSeed'})

# Create new CSV file
merged_data.to_csv('merged_data.csv', index=False)

