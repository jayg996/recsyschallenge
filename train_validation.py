import os
import torch
from torch import optim
from recsys_dataset import RecSysDataset, RecSysDataLoader
from models import FM
from hparams import HParams
import logger
import numpy as np
import pandas as pd
import argparse

logger.logging_verbosity(1)

# use GPU
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# hyper parameters
parser = argparse.ArgumentParser()
parser.add_argument('--index', type=int, help='Experiment Number', default='0')
parser.add_argument('--loss_type', type=str, help='bpr, top1, test', default='bpr')
parser.add_argument('--restore_epoch', type=int, default=1000)
args = parser.parse_args()

config = HParams.load("hparams.yaml")

# loss type print
if not config.mode['train']:
    args.loss_type = 'test'
logger.info("==== Loss Type : %s " % args.loss_type)


# train and valid data load
train_dataset = RecSysDataset(config, train_mode=config.mode['train'], toy_mode=config.mode['toy'], valid_data=False)
valid_dataset = RecSysDataset(config, train_mode=config.mode['train'], toy_mode=config.mode['toy'], valid_data=True)
train_dataloader = RecSysDataLoader(dataset=train_dataset, batch_size=config.experiment['batch_size'], drop_last=False, shuffle=True)
valid_dataloader = RecSysDataLoader(dataset=valid_dataset, batch_size=config.experiment['batch_size'], drop_last=False)

# result and model save paths
restore_epoch = args.restore_epoch
experiment_num = str(args.index)
ckpt_file_name = 'idx_'+experiment_num+'_%03d.pth.tar'
logger.info("==== Experiment Number : %d " % args.index)
if not os.path.exists(os.path.join(config.root_path, 'model')):
    os.makedirs(os.path.join(config.root_path, 'model'))
    os.makedirs(os.path.join(config.root_path, 'result'))

# model load
item_dim = train_dataset.item_vector_dim
sess_dim = 10
model = FM(config.model, item_dim, sess_dim).to(device)
optimizer = optim.Adagrad(model.parameters(), lr=config.experiment['learning_rate'], weight_decay=config.experiment['weight_decay'])

# Load model
if os.path.isfile(os.path.join(config.root_path, 'model', ckpt_file_name % restore_epoch)):
    checkpoint = torch.load(os.path.join(config.root_path, 'model', ckpt_file_name % restore_epoch))
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    logger.info("restore model with %d epochs" % restore_epoch)
else:
    logger.info("no checkpoint with %d epochs" % restore_epoch)
    restore_epoch = 0

if args.loss_type != 'test':
    current_step = 0
    for epoch in range(restore_epoch, config.experiment['max_epoch']):
        # Training
        model.train()
        train_loss_list = []
        for i, data in enumerate(train_dataloader):
            keys, session_vectors, time_infos, labels, item_idx, item_vectors = data
            session_vectors, item_vectors = session_vectors.to(device), item_vectors.to(device)
            session_vectors.requires_grad = True
            item_vectors.requires_grad = True

            optimizer.zero_grad()

            scores, loss = model(session_vectors, item_vectors, labels, item_idx, args.loss_type)

            train_loss_list.append(loss.item())
            loss.backward()
            optimizer.step()

            current_step += 1

        logger.info("training loss for %d epoch: %.4f" % (epoch + 1, np.mean(train_loss_list)))
        # logger.info("training accuracy for %d epoch: %.4f" % (epoch + 1, (correct.item() / total)))

        # Validation
        with torch.no_grad():
            model.eval()
            validation_loss = 0
            n = 0
            for i, data in enumerate(valid_dataloader):
                keys, session_vectors, time_infos, labels, item_idx, item_vectors = data
                session_vectors, item_vectors = session_vectors.to(device), item_vectors.to(device)

                scores, loss = model(session_vectors, item_vectors, labels, item_idx, args.loss_type)

                validation_loss += loss.item()

                n += 1

            validation_loss /= n

            logger.info("validation loss(%d): %.4f" % (epoch + 1, validation_loss))
            # logger.info("validation accuracy(%d): %.4f" % (epoch + 1, (val_correct.item() / val_total)))

            # save model
            if (epoch + 1) % config.experiment['save_step'] == 0:
                logger.info('saving model, Epoch %d, step %d' % (epoch + 1, current_step + 1))
                model_save_path = os.path.join(config.root_path, 'model', ckpt_file_name % (epoch + 1))
                state_dict = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
                torch.save(state_dict, model_save_path)
# for test
## todo : implementation
else:
    raise NotImplementedError

