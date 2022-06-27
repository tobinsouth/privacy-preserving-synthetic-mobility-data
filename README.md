# Privacy Preserving Synthetic Mobility Data

This project is a demonstration of the ability to learn representation distributions of temporal mobility data with differential privacy during training, such that a synthetic dataset can be generated to operationalize the sharing of data with a privacy-preserving mechanism.


## Codebase breakdown
Data is processed in `scripts/data_chunking.py` and a dataloader is defined in `scripts/dataloader.py`. Old models have been removed (but can be found in git history) or are temporarily in `/deprecated/`.

Some newer work has focused on replicating [MoveSim](https://github.com/FIBLAB/MoveSim) with our cuebiq data at different aggregation levels. The MoveSim codebase can be found inside `/MoveSim/` and the data chunking can be found in various places. This will need more work (probably on Tobin's end) to make it work with our new data.

## Information for collaborators.
This is the code base for local use. Models, figures, and data are stored elsewhere. Get access to the data (e.g. use matlaber or get data with rysnc), then run:

```
python scripts/data_chunking.py --kms 10 --number_of_increments 24 --filter_by user
```
and then in the training, you can load in the data with:
```
from scripts.dataloader import get_train_test
train, test = get_train_test(train_size=0.7, batch_size=8, shuffle=True, dataset=filename, data_directory=DATA_PATH)
```

### .gitignored folders
Running code here will require `/data/`, `/figs/`, and `/models/` in the local directory.

### Data 
The training data is stored in `/mas/projects/privacy-pres-mobility/data/processed_data/` and is automatically loaded in by the dataloader.
