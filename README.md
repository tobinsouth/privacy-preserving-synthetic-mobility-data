# Privacy Preserving Synthetic Mobility Data

This project is a demonstration of the ability to learn representation distributions of temporal mobility data with differential privacy during training, such that a synthetic dataset can be generated to operationalise the sharing of data with a privacy-preserving mechanism.


## Codebase breakdown
Data is processed in `scripts/data_chunking.py` and a dataloader is defined in `scripts/dataloader.py`. Model definition for VAE is in `scripts/VAE.py`. Old models have been removed but can be found in git history.

Newer work has focused on replicating [MoveSim](https://github.com/FIBLAB/MoveSim) with our cuebiq data at different aggregation levels. The MoveSIm codebase can be found inside `/MoveSim/` and the data chunking can be found in various places.

## Information for collaborators.

This is the code base for local use. Models, figures, and data are stored elsewhere.

### .gitignored folders
Running code here will require `/data/`, `/figs/`, and `/models/` in the local directory.

### Data 
The training data is stored in `/mas/projects/privacy-pres-mobility/data/processed_data/` and is automatically loaded in by the dataloader.

### TODO List
* Get new data in that includes home addresses.
* Document the notebook and convert to script with input params.
* Document the readme with how to run the code.
* Combine the datasheets (why are there two?)
* Make sure new data is available.
* Update dataloader.py to load new data.


