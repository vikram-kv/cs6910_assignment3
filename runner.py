# Here, we use pytorch-lightning to eliminate boiler plate code and facilitate
# support for Early stopping and model checkpointing using available library functions.
# A single class `Runner` is defined with the necessary methods (setup, {train|val|test}_dataloader, train_step, ...)
# needed by pytorch-lightning (renamed as lightning in the most recent release).
# Also, wandb logging is automatic if a run is active
import lightning as lt
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchaudio.functional import edit_distance as edit_dist
import random
import wandb
from language import *
from dataset_dataloader import *
from encoder_decoder import *

'''
    Pytorch lightning based module that encapsulates our seq2seq model with useful
    helper functions. Note that this class instance will log metrics on wandb if 
    a wandb run is active [wandb.run is not None.]
'''
class Runner(lt.LightningModule):
    def __init__(self, SOURCE, TARGET, src_lang : Language, tar_lang : Language, common_embed_size, common_num_layers, 
                 common_hidden_size, common_cell_type, init_tf_ratio = 0.8, enc_bidirect=False, attention=False, dropout=0.0, 
                 opt_name='Adam', learning_rate=2e-3, batch_size=32):
    
        super(Runner,self).__init__()
        # save the language objects
        self.src_lang = src_lang
        self.tar_lang = tar_lang
        self.SOURCE = SOURCE
        self.TARGET = TARGET
        # create all the sub-networks and the main model
        self.encoder = EncoderNet(vocab_size=src_lang.get_size(), embed_size=common_embed_size,
                             num_layers=common_num_layers, hid_size=common_hidden_size,
                             cell_type=common_cell_type, bidirect=enc_bidirect, dropout=dropout)
        if attention:
            self.attention = True
            self.attn_layer = Attention(common_hidden_size, enc_bidirect)
        else:
            self.attention = False
            self.attn_layer = None
        
        self.decoder = DecoderNet(vocab_size=tar_lang.get_size(), embed_size=common_embed_size,
                             num_layers=common_num_layers, hid_size=common_hidden_size,
                             cell_type=common_cell_type, attention=attention, attn_layer=self.attn_layer,
                             enc_bidirect=enc_bidirect, dropout=dropout)
        
        self.model = EncoderDecoder(encoder=self.encoder, decoder=self.decoder, src_lang=src_lang, 
                                    tar_lang=tar_lang)

        # for determinism
        torch.manual_seed(42); torch.cuda.manual_seed(42); np.random.seed(42); random.seed(42)

        self.model.apply(self.init_weights) # initialize model weights
        self.batch_size = batch_size

        # optimizer for the model and loss function [that ignores locs where target = PAD token]
        self.loss_criterion = nn.CrossEntropyLoss(ignore_index=tar_lang.sym2index[PAD_SYM])
        self.opt_name = opt_name
        self.learning_rate = learning_rate

        # only adam is present in configure_optimizers as of now
        if (opt_name != 'Adam'):
            exit(-1)

        self.cur_tf_ratio = init_tf_ratio # the current epoch teacher forcing ratio
        self.min_tf_ratio = 0.01          # minimum allowed teacher forcing ratio

        # lists for tracking predictions/true words etc...
        self.pred_train_words = []; self.true_train_words = []
        self.pred_valid_words = []; self.true_valid_words = []
        self.test_X_words = []; self.pred_test_words = []; self.true_test_words = []
        self.attn_matrices = []  # used only when there is attention layer

        # lists for tracking losses
        self.train_losses = []
        self.valid_losses = []

        # dictionary for logging at end of val epoch
        self.wdb_logged_metrics = dict()
        self.best_val_acc_seen = -0.01 # to save model weights on wandb

    def configure_optimizers(self):
        optimizer = None
        if self.opt_name == 'Adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer

    @staticmethod
    def init_weights(m):
        '''
        function to initialize the weights of the model parameters
        '''
        for name, param in m.named_parameters():
            if 'weight' in name:
                 nn.init.uniform_(param.data, -0.04, 0.04)
            else:
                nn.init.constant_(param.data, 0)
    
    @staticmethod
    def exact_accuracy(pred_words, tar_words):
        ''' 
        compute the accuracy using (predicted words, target words) and return it.
        exact word matching is used.
        '''
        assert(len(pred_words) == len(tar_words))
        count = 0
        for i in range(len(pred_words)):
            if pred_words[i] == tar_words[i]:
                count += 1
        return count / len(pred_words)

    ####################
    # DATA RELATED HOOKS
    ####################

    def setup(self, stage=None):
        # load all the available data on all GPUs
        self.x_train, self.y_train = load_data(self.TARGET, 'train')
        self.x_valid, self.y_valid = load_data(self.TARGET, 'valid')
        self.x_test, self.y_test = load_data(self.TARGET, 'test')

    def train_dataloader(self):
        dataset = TransliterateDataset(self.x_train, self.y_train, src_lang=self.src_lang, tar_lang=self.tar_lang)
        dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size, collate_fn=CollationFunction(self.src_lang, self.tar_lang))
        return dataloader

    def val_dataloader(self):
        dataset = TransliterateDataset(self.x_valid, self.y_valid, src_lang=self.src_lang, tar_lang=self.tar_lang)
        dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size, collate_fn=CollationFunction(self.src_lang, self.tar_lang))
        return dataloader

    def test_dataloader(self):
        dataset = TransliterateDataset(self.x_test, self.y_test, src_lang=self.src_lang, tar_lang=self.tar_lang)
        dataloader = DataLoader(dataset=dataset, batch_size=1, collate_fn=CollationFunction(self.src_lang, self.tar_lang))
        # we do inference word by word. So, batch_size = 1
        return dataloader

    ####################
    # INTERFACE RELATED FUNCTIONS 
    ####################

    def training_step(self, train_batch, batch_idx):
        batch_X, batch_y, X_lens = train_batch
        # get the logits, preds for the current batch
        logits, preds = self.model(batch_X, batch_y, X_lens, tf_ratio=self.cur_tf_ratio)
        # ignore loss for the first time step
        targets = batch_y[:, 1:]; logits = logits[:, 1:, :]
        logits = logits.swapaxes(1, 2) # make class logits the second dimension as needed
        loss = self.loss_criterion(logits, targets)
        # for epoch-level metrics[accuracy], log all the required data
        self.true_train_words += self.tar_lang.convert_to_words(batch_y)
        self.pred_train_words += self.tar_lang.convert_to_words(preds)
        self.train_losses.append(loss) # to get train loss for epoch
        return loss
    
    def on_train_epoch_end(self):
        # for wandb logging
        self.wdb_logged_metrics['train_loss'] = torch.stack(self.train_losses).mean()
        self.wdb_logged_metrics['train_acc'] = self.exact_accuracy(self.pred_train_words, self.true_train_words)
        self.wdb_logged_metrics['tf_ratio'] = self.cur_tf_ratio
        self.wdb_logged_metrics['epoch'] = self.current_epoch
        self.train_losses.clear()

        # note that on train_epoch_end is actually executed after valid epoch; so we log onto wandb here
        if wandb.run is not None:
            wandb.log(self.wdb_logged_metrics)

        # for display bar
        self.log('train_loss', self.wdb_logged_metrics['train_loss'], on_epoch=True, prog_bar=True)
        self.log('train_acc', self.wdb_logged_metrics['train_acc'], on_epoch=True, prog_bar=True)
        self.pred_train_words.clear(); self.true_train_words.clear() # clear to save memory and for next epoch

        # for first 12 epochs, we dont change the tf ratio. Then we decrease it by 0.1 every epoch till
        # min_tf_ratio is reached. This is also logged.
        if (self.current_epoch >= 11):
            self.cur_tf_ratio -= 0.1
            self.cur_tf_ratio = max(self.cur_tf_ratio, self.min_tf_ratio)

    def validation_step(self, valid_batch, batch_idx):
        batch_X, batch_y, X_lens = valid_batch
        # get the logits, preds for the current batch
        logits, preds = self.model(batch_X, batch_y, X_lens) # no teacher forcing
        # ignore loss for the first time step
        targets = batch_y[:, 1:]; logits = logits[:, 1:, :]
        logits = logits.swapaxes(1, 2) # make class logits the second dimension as needed
        loss = self.loss_criterion(logits, targets)
        # for epoch-level metrics[accuracy], log all the required data
        self.true_valid_words += self.tar_lang.convert_to_words(batch_y)
        self.pred_valid_words += self.tar_lang.convert_to_words(preds)
        self.valid_losses.append(loss) # to get val loss for epoch
    
    def on_validation_epoch_end(self):
        # for wandb logging
        self.wdb_logged_metrics['val_loss'] = torch.stack(self.valid_losses).mean()
        self.wdb_logged_metrics['val_acc'] = self.exact_accuracy(self.true_valid_words, self.pred_valid_words)
        self.valid_losses.clear()
        
        # for display bar
        self.log('val_loss', self.wdb_logged_metrics['val_loss'], on_epoch=True, prog_bar=True)
        self.log('val_acc', self.wdb_logged_metrics['val_acc'], on_epoch=True, prog_bar=True)

        self.true_valid_words.clear(); self.pred_valid_words.clear() # clear to free memory and for next epoch
    
    # in a test step(for a batch), we just find the predictions and keep track of test loss
    def test_step(self, test_batch, batch_idx):
        batch_X, batch_y, X_lens = test_batch
        logits, pred_word, attn_matrix = self.model.greedy_inference(batch_X, X_lens)
        # update all the global lists
        self.pred_test_words += pred_word
        self.true_test_words += self.tar_lang.convert_to_words(batch_y)
        self.test_X_words += self.src_lang.convert_to_words(batch_X)
        # if there is attention, update the attention list also
        if (self.attention):
            self.attn_matrices += [attn_matrix]
        # ignore loss for the first time step
        targets = batch_y[:, 1:]; logits = logits[1:, :]
        # we shrink the logits to the true decoded sequence length for loss computation alone
        true_dec_len = targets.size(1)
        logits = (logits[:true_dec_len, :]).swapaxes(0,1).unsqueeze(0)
        # squeeze and swapping of dimensions is to meet condition needed by nn.CrossEntopyLoss()
        loss = self.loss_criterion(logits, targets)
        self.log('test_loss', loss, prog_bar=True, on_epoch=True, on_step=False)

    # on test epoch end, we just log the accuracy
    def on_test_epoch_end(self):
        self.log('test_acc', self.exact_accuracy(self.pred_test_words, self.true_test_words), 
                 on_epoch=True, prog_bar=True)
    
    # here, we will return the test predictions and attention matrices for logging
    def get_test_results(self):
        # if attention layer is present, we return attention matrices as well.
        if self.attention:
            ret_info = (self.test_X_words.copy(), self.true_test_words.copy(), self.pred_test_words.copy(), self.attn_matrices.copy())
        else:
            ret_info = (self.test_X_words.copy(), self.true_test_words.copy(), self.pred_test_words.copy())
        # clear before return to save memory 
        self.pred_test_words.clear(); self.true_test_words.clear(); self.test_X_words.clear(); self.attn_matrices.clear()
        return ret_info