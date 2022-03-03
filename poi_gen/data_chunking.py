# Load in all of the stays*.csv.gz files, group by a users path for a day, and save all of these paths into individual files for each sequence of stays. These files than then be incrementally read in during training.


import pandas as pd, numpy as np, os, glob, pickle
from tqdm import tqdm

data_directory = '/mas/projects/privacy-pres-mobility/data/'


all_stays_csvs = glob.glob(data_directory+'/*.csv.gz')
all_geoids = set()
all_pois = set()
all_sequences = []
for stays_csv in all_stays_csvs:
    stays = pd.read_csv(stays_csv)
    stays = stays[['user','GEOID_home', 'fsq_id', 'ini_dat']]
    grouped_users = stays.groupby('user')    
    for i, (user, user_df) in enumerate(tqdm(grouped_users)):
        user_df = user_df.sort_values(by='ini_dat')
        home = user_df['GEOID_home'].unique()[0]
        all_geoids.update([home])
        all_pois.update(set(user_df['fsq_id']))
        user_df = user_df[['GEOID','ini_dat','datetime','time','duration']]
        seq = [(home,0, 'H')]
        for idx, row in user_df.iterrows():
            if row['GEOID'] == home:
                if len(seq) > 5:
                    seq.append((row['GEOID'], row['time'], 'H'))
                    all_sequences.append(seq)
                    seq = []
                    continue
            seq.append((row['GEOID'], row['time'], 'A'))
        if len(seq) > 5:
            all_sequences.append(seq)
            seq = []

import pickle
with open(data_directory+'processed_data/cuebiq.pickle', "wb") as f:
    pickle.dump(all_sequences, f)

# Save geoid mapping
all_geoids = sorted(list(all_geoids))
geoid_mapping = dict(zip(all_geoids, range(2,len(all_geoids)+2)))
with open(data_directory+'processed_data/geoid_mapping.pickle', "wb") as f:
    pickle.dump(geoid_mapping, f,)



# import pandas as pd
# with open(data_path+'dataset_ubicomp2013/dataset_ubicomp2013_checkins.txt', 'r') as f:
    
# fs_checkins = pd.read_csv(f, sep='\t', header=None, columns = ['user', 'location'])


# # DeepMove Foursquare API data.
# import pickle
# with open(data_directory+'raw_data/foursquare.pk', 'rb') as f:
#     foursquare = pickle.load(f,encoding='latin')

# all_sessions = []
# for userdata in foursquare['data_neural'].values():
#     for session in userdata['sessions'].values():
#         all_sessions.append(list(tuple(t) for t in session))
#         # Turns out that all sessions are unique.
#         # There are 8871 sessions in the dataset.

# # Save as pickle
# with open(data_directory+'processed_data/foursquare.pickle', 'wb') as f:
#     pickle.dump(all_sessions, f)



# Plotting
from mpl_toolkits.axes_grid.inset_locator import inset_axes

plt.hist(lengths, bins=100, log=True)
plt.xlabel('Sequence Length')
plt.ylabel('Number of Sequences')
ax = plt.gca()
axins = ax.inset_axes([0.5, 0.3, 0.4, 0.3])
axins.hist(lengths, bins=np.arange(0,101, 1), log=True)
axins.axvline(6, color='black', linestyle='--', label='Minimum')
axins.legend()
ax.text(0.5, 0.9, 'Minimum Sequence Length: 6, Mean Length: %.2f'%np.mean(lengths), transform=ax.transAxes)
plt.title('Sequence Lengths for %s Dataset'%'Cuebiq')
plt.tight_layout()
plt.savefig('../figs/seq_lengths.png')