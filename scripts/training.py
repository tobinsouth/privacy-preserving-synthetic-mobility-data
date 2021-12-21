import torch, numpy as np
from tqdm import tqdm

# Get the dataloader
batch_size=8
from dataloader import get_train_test
trainStays, testStays = get_train_test(train_size=0.7, batch_size=batch_size, shuffle=False, dataset='foresquare')

# Load and define the model
from VAE import SentenceVAE
# Model params
params = dict(
    vocab_size = testStays.dataset._vocab_size,
    max_sequence_length = testStays.dataset._max_seq_len,
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

device = "cuda" # torch.device("cuda" if torch.cuda.is_available() else "cpu") # cuda is being a pain
model = model.to(device)


# Training

epochs = 10
learning_rate = 0.001
k = 0.0025
x0 =2500

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

step = 0
for epoch in range(epochs):
    for iteration, batch in tqdm(enumerate(trainStays)):
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
