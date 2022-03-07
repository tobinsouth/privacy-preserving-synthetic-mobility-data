


# stays2_31080.Rds
This folder contains the stays for a selected number of users (the ones in the home files). 

user = unique id for each user
patch = id of the stay
nn = number of points in the stay
avg_speed = average speed in the stay
duration = total time spent in the stay (seconds)
lon_center = lon of the centroid of the stay
lat_center = lat of the centroid of the stay
lon_medoid = lon of the medoid of the stay
lat_medoid = lat of the medoid of the stay
dw = binary variable: dw = 1 means it is a stay (stop). dw = 0 means it is a trip
ini_dat = initial time of the stay in unix time
Home_lon_med = longitude of the home
Home_lat_med = latitude of the home
GEOID_home = GEOID (block group level) of the user home
Income = median income of the GEOID_home user
Quant = quantile of income of the GEOID_home (from 1 to 4) user
GEOID = GEOID block group where the stay happens.

## Original Data Rules: 
- Raw data can NOT leave the server. 
- Perform your analysis on the data locally in the machine. Python and Rstudio are installed on the server.
- You are free to export your aggregated results to your local machine. 
- For license reasons, we monitor every activity on the server.
- Our license requires us to check with Cuebiq any public release of information about projects, research, visualizations, etc. Do not submit, publish or post any info about your analysis without checking with me.
Data: 
Stays data is located in /data/stays. See README for a description of it. Stays are extracted using the Hariharan and - Toyama algorithm. Details are found in https://www.nature.com/articles/s41467-021-24899-8. Please, cite this paper to give the technical details of how the data was extracted.
- POIs in the different urban areas are in /data/foursquare. Details of how the data was extracted are also in the paper above.
- Demographic census information are in /data/maps.
Quotas:
 - There are no CPU or disk quotas in the server, but disk space is a shared resource, and it is scarce. Please, do not copy large files to your local directory and remove temp or unused files regularly.