# Params
learning_rate = 0.001
k = 0.0025
x0 =2500
epochs = 4
batch_size=16


import torch, numpy as np
from tqdm import tqdm

# Get the dataloader
from dataloader import get_train_test
trainStays, testStays = get_train_test(train_size=0.95, batch_size=batch_size, shuffle=True, dataset='cuebiq')

# Load and define the model
from VAE import SentenceVAE, device 

# Model params
params = dict(
    vocab_size = trainStays.dataset.dataset._vocab_size,
    max_sequence_length = trainStays.dataset.dataset._max_seq_len,
    embedding_size = 256,
    rnn_type =  'gru',
    hidden_size = 256,
    num_layers = 1,
    bidirectional = False,
    latent_size = 16,
    word_dropout = 0,
    embedding_dropout = 0.5,
    sos_idx=0,
    eos_idx=0,
    pad_idx=0,
    unk_idx=1,
)
model = SentenceVAE(**params)
model = model.to(device) # Device is defined in VAE

# Custom loss function from paper

NLL = torch.nn.NLLLoss(ignore_index=0, reduction='sum')

def loss_fn(logp, target, mean, logv, step, k, x0):
    """The loss function used in the paper, taken from https://github.com/timbmg/Sentence-VAE"""
    target = target.view(-1)
    logp = logp.view(-1, logp.size(2))

    # Negative Log Likelihood
    NLL_loss = NLL(logp, target)

    # KL Divergence
    KL_loss = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())
    KL_weight = float(1/(1+np.exp(-k*(step-x0))))

    return NLL_loss, KL_loss, KL_weight

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Logging with tensorboard
from torch.utils.tensorboard import SummaryWriter
LOG_DIR = "runs/cuebiq"
comment = f' batch_size = {batch_size} lr = {learning_rate} dp = False'
train_writer = SummaryWriter(LOG_DIR + "/train", comment=comment)
val_writer = SummaryWriter(LOG_DIR + "/val", comment=comment)

# Run training loop
step = 0
for epoch in range(epochs):
    running_loss = 0.0
    for i, batch in enumerate(tqdm(trainStays, miniters=100)):
        batch = batch.to(device)
        # Forward pass
        logp, mean, logv, z = model(batch)

        # loss calculation
        NLL_loss, KL_loss, KL_weight = loss_fn(logp, batch, mean, logv, step, k, x0)

        loss = (NLL_loss + KL_weight * KL_loss) / batch_size
        loss.to(device)

        # backward + optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step += 1

        running_loss += loss.item()
        if i % 1000 == 999:  
            train_writer.add_scalar('loss',  running_loss / 1000, epoch * len(trainStays) + i)
            running_loss = 0.0


        # Periodic Validation and checkpointing
        if i % 20000 == 19999:
            model.eval()
            val_loss = 0.0
            for batch in testStays:
                batch = batch.to(device)
                logp, mean, logv, z = model(batch)
                NLL_loss, KL_loss, KL_weight = loss_fn(logp, batch, mean, logv, step, k, x0)
                loss = (NLL_loss + KL_weight * KL_loss) / batch_size
                val_loss += loss.item()
            val_writer.add_scalar('loss',  val_loss / 20000, epoch * len(trainStays) + i)
            model.train()
            print('Validation loss:', val_loss / 20000)

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss / 10000,
                }, '../models/cuebiq_vae.pt')


train_writer.close()
val_writer.close()