


from DeepMoveModels import TrajPreSimple, Attn, TrajPreAttnAvgLongUser, TrajPreLocalAttnLong


import torch.optim as optim
loc_emb_size =  500
uid_emb_size = 40
voc_emb_size = 50
tim_emb_size = 10
lr = 5 * 1e-4
lr_decay = 0.1
lr_step = 2
dropout_p = 0.3
L2 = 1 * 1e-5


criterion = nn.NLLLoss().cuda()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr=lr,weight_decay=L2)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=lr_step,factor=lr_decay, threshold=1e-3)





data_neural # make


data_train, train_idx = generate_input_long_history(data_neural, 'train', candidate=candidate)
data_test, test_idx = generate_input_long_history(data_neural, 'test', candidate=candidate)



metrics = {'train_loss': [], 'valid_loss': [], 'accuracy': [], 'valid_acc': {}}


forward(self, loc, tim, target_len)




model = TrajPreLocalAttnLong(parameters=parameters).cuda()


