"""
 Load in all of the stays*.csv.gz files, group by a users path for a day, and save all of these paths into individual files for each sequence of stays. These files than then be incrementally read in during training.

 This version of the script will bin each trajectory into hour long time bins where the location is the GEOID where the user spent the most time.
"""

import pandas as pd, numpy as np, os, glob, pickle
from tqdm import tqdm

data_directory = '/mas/projects/privacy-pres-mobility/data/'


all_stays_csvs = glob.glob(data_directory+'/*.csv.gz')
all_geoids = set()
all_sequences = []
for stays_csv in all_stays_csvs:
    stays = pd.read_csv(stays_csv)
    stays['datetime'] = pd.to_datetime(stays['ini_dat'], unit='s')
    stays['time'] = stays['ini_dat'] // 3600 
    stays['dur_h'] = stays['duration'] // 3600 
    grouped_users = stays.groupby('user')
    for i, (user, user_df) in enumerate(tqdm(grouped_users)):
        all_sequences_user = []
        user_df = user_df.sort_values(by='ini_dat')

        # Split into time binned seq.
        home = user_df['GEOID_home'].unique()[0]
        seq = [home]
        start_time = user_df['time'].iloc[0] // 1 # Starting Hour
        end_time =((user_df['time'].iloc[-1] + user_df['dur_h'].iloc[-1]) // 1)+1 # Ending Hour
        for hour in np.arange(start_time, end_time):
            time_df = user_df[(user_df['time'] + user_df['dur_h'] >= hour) & (user_df['time'] < hour+1)]
            if len(time_df) == 0:
                seq.append(-1)
            elif len(time_df) == 1:
                seq.append(time_df['GEOID'].iloc[0])
            else:
                # Calculate how much time each GEOID spent in the hour.
                best, best_idx = -1, -1
                for i, (t,d) in enumerate(zip(time_df['time'], time_df['dur_h'])):
                    if t == hour:
                        contribution = max(d, t % 1)
                    elif t < hour:
                        contribution = (t+d) % 1
                    else:
                        print('This should not be hit.',t,d,hour)
                    if contribution > best:
                        best = contribution
                        best_idx = i
                seq.append(time_df['GEOID'].iloc[best_idx])
        all_geoids.update(seq)

        sub_seq, na_counter = [], 0
        for i, geo in enumerate(seq):
            na_counter = na_counter + 1 if geo == -1 else 0
            if na_counter > 3:
                if len(sub_seq) > 0 and sum(sub_seq) > -1:
                    all_sequences_user.append(sub_seq[:-3])
                sub_seq, na_counter = [], 0
                continue

            if geo == -1 and len(sub_seq) == 0:
                continue # Skip prepending -1 to the sequence.
            else:
                sub_seq.append(geo)
        all_sequences_user.append(sub_seq) # Catch final

        for seq in all_sequences_user:
            if len(seq) > 5:
                if len(set(seq)) > 3:
                    all_sequences.append([home]+seq)
        
        if i % 10**3 == 0:
            with open(data_directory+'processed_data/cuebiq_time_binned.pickle', "wb") as f:
                pickle.dump(all_sequences, f)

with open(data_directory+'processed_data/cuebiq_time_binned.pickle', "wb") as f:
        pickle.dump(all_sequences, f)

# Save geoid mapping
all_geoids = sorted(list(all_geoids))
geoid_mapping = dict(zip(all_geoids, range(2,len(all_geoids)+2)))
with open(data_directory+'processed_data/geoid_mapping_time_binned.pickle', "wb") as f:
    pickle.dump(geoid_mapping, f,)
