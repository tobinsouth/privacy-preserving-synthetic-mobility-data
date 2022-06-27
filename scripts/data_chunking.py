
"""
Script for loading in original stays dataset from cuebiq and chunking it into standardised numpy sequences of length `number_of_increments`, where each sequence is where a user was on a single day and each element is the GEOID in that time interval.

author: @tobinsouth
date: 2022-05-27

# Cleaning Process
1. We identify the range / radius we're interested in as a central point + a radius. Users who appear at least once in that circle are kept. Edit param `km` to change the radius.
2. We group by users to find their trajectories.
3. Split each day for a user into increments of time (30mins or 1hr). 48/24 time increments from 4am to 4am. Control this with `number_of_increments`
4. Any time a user is not seen in an increment, we set the value to `supernode`. A future alternative could be use to the home location of the user.
5. If a user is seen less that `overservation_threshold` times out of the `number_of_increments`, we remove them from the dataset. 

## Speed Improvements
The code has been optimized for speed. The `use_numba` and `parallel_apply` params may require extra installs but will dramatically decrease runtime on large datasets. To install: 
```!pip install numba pandarallel```

## Inputs
km = 5 # Input, how many kilometers of radius to consider
number_of_increments = 24 # How many increments of a day to use, 24 is hourly, 48 is every half hour, etc.
supernode = 99999999999
overservation_threshold = 12 # How many hours in a day do we need to see you to count as a trajectory. 
filter_by = 'user' or 'point' # Do we filter users who don't enter the radius or points that don't enter the radius.
use_numba = True # Use numba to speed up the code
parallel_apply = True # Use parallel_apply to speed up the code

data_directory = '/mas/projects/privacy-pres-mobility/data/' # Use on matlaber
data_directory = '../data/' # Use on local machine

"""

import pandas as pd, numpy as np, glob, pickle, argparse, sys
from tqdm import tqdm
tqdm.pandas()

parser = argparse.ArgumentParser()
parser.add_argument('--kms', default=1000, type=int)
parser.add_argument('--number_of_increments',  default=24, type=int)
parser.add_argument('--supernode', default=99999999999, type=int)    
parser.add_argument('--overservation_threshold', default=12, type=int)
parser.add_argument('--filter_by', default='user', type=str)
parser.add_argument('--use_numba', default=True, type=bool)
parser.add_argument('--parallel_apply', default=True, type=bool)
default_path = '../data/' if sys.platform=='darwin' else '/mas/projects/privacy-pres-mobility/data/'
parser.add_argument('--data_directory', default=default_path, type=str)

opt = parser.parse_args()

kms = opt.kms
number_of_increments = opt.number_of_increments
supernode = opt.supernode
overservation_threshold = opt.overservation_threshold
filter_by = opt.filter_by
use_numba = opt.use_numba
parallel_apply = opt.parallel_apply
data_directory = opt.data_directory

# Function definitions

def get_within_time(ini_dat_array, fin_dat_array, t0, t1, result_array, supernode) -> int:
    """
    This function returns the points in the result_array that are within the time interval [t0, t1]. 
    This has been seperated into a function to speed up the code.
    """
    within_time = np.where(((ini_dat_array >= t0) & 
                    (ini_dat_array <= t1)) | 
                    ((fin_dat_array <= t1) & 
                    (fin_dat_array >= t0)))[0]

    if len(within_time) == 1:
        return result_array[within_time[0]]
    elif len(within_time) == 0:
        return supernode
    else:
        return result_array[within_time[0]] # If there are multiple points, just use the first one
        #  We could make this more robust by choosing the location they spent the most time at, but this is a good enough approximation for now.
    
if use_numba:
    # This will use a JIT version of the slow part of the function to speed up up the list comparisons with machine code
    import numba
    get_within_time = numba.jit(get_within_time, nopython=True)

if parallel_apply == True: # Setting up parallel_apply to speed up the code. Doing this early to force an install crash for those without it.
    from pandarallel import pandarallel
    pandarallel.initialize(progress_bar=True)
    
def get_all_sequences(user_df: pd.DataFrame):
    """
    This function takes a user_df and returns a list of sequences, each a `number_of_increments` long numpy array with where the user was at each increment (starting from 4am).
        user_df: A dataframe with a single user's sorted data points.
    Expected to be applied to a pandas.DataFrameGroupBy object.
    """
    user_sequences = []
    start_date = (user_df['dt'].iloc[0].normalize() + pd.Timedelta(hours=4)).timestamp() # Get 4am of the first day


    while len(user_df) > 0: # We progressively filter off days as we go along
        
        day_sequence = np.zeros(number_of_increments) # Pre-allocate for speed
        ini_dat_array, fin_dat_array, result_array = user_df['ini_dat'].values, user_df['fin_dat'].values, user_df['GEOID'].values # Pre-extract data as arrays for speed

        for t in range(number_of_increments):
            t0 = start_date + (3600*24/number_of_increments)*t
            t1 = t0 + (3600*24/number_of_increments)
            day_sequence[t] = get_within_time(ini_dat_array, fin_dat_array, t0, t1, result_array, supernode) # Call JIT complied function
            
        if sum(day_sequence != supernode) > overservation_threshold: # Remove days with too few overservations
            user_sequences.append(day_sequence)

        user_df = user_df[user_df['ini_dat'] > (start_date + 24*3600)] # Filter off days that we've already processed

        if len(user_df) == 0:
            break
        
        # Get next start date (earliest date more that 24 hours away)
        start_date = (user_df['dt'].iloc[0].normalize() + pd.Timedelta(hours=4)).timestamp()

    return user_sequences

print('Looking for data in `{}`'.format(data_directory))
print("Reading in large stays data file...")
all_stays_csvs = glob.glob(data_directory+'/*stays*1.csv.gz') # This will look for all of the stays1.csv.gz files in the data directory. Stay2 is old data.
stays = pd.read_csv(all_stays_csvs[0]) # Currently we only have a single file, but this is ready to be expanded to multiple files.

print("Beginning the filters. Change `filter_by` and `kms` to change the filters.")
# 1. Filtering users to a circle
# Get the distance between each stay and the center point
center_point = stays['lat_medoid'].median(), stays['lon_medoid'].median()
stays['distance_from_center'] = np.sqrt((stays['lat_medoid']-center_point[0])**2 + (stays['lon_medoid']-center_point[1])**2)
stays['within_bounds'] = stays['distance_from_center'] <  kms/111.2

if filter_by == 'user':
        # Approach 1: Filter any ~USER~ that wasn't in the circle of radius kms
        grouped_users = stays.groupby('user')
        filtered_stays = stays[stays['user'].map(grouped_users['within_bounds'].any())].copy()
elif filter_by == 'point':
        # Approach 2: Filter any ~DATA POINT~ that wasn't in the circle of radius kms (this is a lot more reductive)
        filtered_stays = stays[stays['within_bounds']].copy()
else: 
        raise ValueError('filter_by must be either "user" or "point"')

print("Total original data points: %d \nFiltered total data points: %d \nPercentage reduced %.4f \nNumber of unique users: %d" % 
(len(stays), len(filtered_stays), 1-len(filtered_stays)/len(stays), len(filtered_stays['user'].unique())))

filtered_stays['dt'] = pd.to_datetime(filtered_stays['ini_dat'], unit='s', utc=True) + pd.Timedelta(hours=-7) # Set timezone to PST
filtered_stays['finishtime'] = filtered_stays['dt'] + filtered_stays['duration'].apply(pd.Timedelta, unit='s')
filtered_stays['ini_dat'] = filtered_stays['dt'].apply(lambda dt: dt.timestamp())
filtered_stays['fin_dat'] = filtered_stays['finishtime'].apply(lambda dt: dt.timestamp())
filtered_stays.sort_values(by='dt', inplace=True)
filtered_stays = filtered_stays[['dt','ini_dat', 'fin_dat', 'user', 'GEOID']]
filtered_grouped_users = filtered_stays.groupby('user')


# Looping through users to create trajectories
print("Looping over users (this is the slow part; use numba & parrallel_apply params to speed up)")

if parallel_apply == True:
    all_sequences = filtered_grouped_users.parallel_apply(get_all_sequences)
else:
    all_sequences = filtered_grouped_users.progress_apply(get_all_sequences)

all_sequences = [s for sublist in all_sequences for s in sublist] # Flatten the list of lists

print("Done. Number of final sequences: %d" % len(all_sequences))

# Save result
with open(data_directory+'24hr_cuebiq_trajectories.pickle', "wb") as f:
    pickle.dump(all_sequences, f, 4)

print('Data saved to '+data_directory+'24hr_cuebiq_trajectories.pickle')




