# Load in all of the stays*.csv.gz files, group by a users path for a day, and save all of these paths into individual files for each sequence of stays. These files than then be incrementally read in during training.


import pandas as pd, numpy as np, os, glob
from tqdm import tqdm

data_directory = '../data/'


all_stays_csvs = glob.glob(data_directory+'/*.csv.gz')
all_geoids = []
for stays_csv in all_stays_csvs:
    stays = pd.read_csv(stays_csv)
    stays['datetime'] = pd.to_datetime(stays['ini_dat'], unit='s')
    stays['time'] = stays['datetime'].dt.hour + stays['datetime'].dt.minute/60
    grouped_users = stays.groupby('user')
    for user, user_df in tqdm(grouped_users):
        # home = user_df['GEOID_home'].unique()[0]
        user_df.sort_values('ini_dat', inplace=True, ascending=True)
        # Currently not using the home information
        user_df[['GEOID','time', 'duration']].to_csv(data_directory+'.each_traj/traj_'+user+'.csv', index=False)

    all_geoids.extend(list(stays['GEOID'].unique()))
all_geoids = list(set(all_geoids))
geoid_mapping = dict(zip(all_geoids, range(1,len(all_geoids)+1)))
# Save geoid mapping
import pickle
with open(data_directory+'geoid_mapping.pickle', "wb") as f:
    pickle.dump(geoid_mapping, f,)