import torch
import torch.nn as nn
from language import *
import lightning as lt

'''
    Class for the Encoder Network
'''
class EncoderNet(lt.LightningModule):
    def __init__(self, vocab_size, embed_size, num_layers, hid_size, cell_type, 
                 bidirect=False, dropout=0):
        '''
            input_vocab_size (V) = number of tokens in the input language dictionary
            embed_size = dim of embedding for each input token
            num_layers = number of layers in the encoder network
            hidden_size = dim of hidden state of each cell
            cell_type = RNN/GRU/LSTM
            bidirect = True for bidirectional network and False otherwise
            dropout = dropout probability
        '''
        # save all the necessary arch information
        super(EncoderNet, self).__init__()
        self.arch = {
            'vocab_size' : vocab_size,
            'hid_size' : hid_size,
            'embed_size' : embed_size,
            'num_layers' : num_layers,
            'cell' : cell_type,
            'bidirect' : bidirect
        }

        # create the embedding layer and a dropout layer for it
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(p=dropout)

        # we create the kw args using the received parameters and use it to create network layer stack
        kwargs = {'input_size':embed_size, 'hidden_size':hid_size, 'num_layers':num_layers, 
                 'bidirectional':bidirect, 'batch_first':True}
        if num_layers > 1:
            kwargs['dropout'] = dropout
        if cell_type == 'RNN':
            self.network = nn.RNN(**kwargs)
        elif cell_type == 'LSTM':
            self.network = nn.LSTM(**kwargs)
        else:
            self.network = nn.GRU(**kwargs)
        
        # for combining the final layer's forward and reverse directions' final hidden state
        # we create linear layers to do this for each encoder layer
        if (bidirect):
            self.combine_forward_backward = [nn.Linear(2 * hid_size, hid_size) for _ in range(num_layers)]
            self.combine_forward_backward = nn.ModuleList(self.combine_forward_backward)

    def forward(self, batch_X, X_lens):
        '''
            batch_X - padded input batch of examples. shape = (batch_size, max_batch_seq_length)
                      [padding is already taken care of by collate function of DataLoader]
            X_lens - length of each input sequence. A python list of `batch_size` many integers
        '''
        # pass the batch through the embedding with dropout and pack the batch. packing is for efficiency 
        batch_X = self.embedding(batch_X)
        batch_X = self.dropout(batch_X)
        packed_batch_x = nn.utils.rnn.pack_padded_sequence(batch_X, lengths=X_lens.cpu(), batch_first=True, enforce_sorted=False)
        # send the batch through the network correctly based on cell type
        # packed_outputs = packed sequence of outputs from the final layer.
        # hidden_outputs = hidden outputs from every layer. shape = (D * num_layers, batch_size, hidden_size)
        # D = 2 if bidirectional; else 1
        if self.arch['cell'] == 'LSTM':
            packed_outputs, (hidden_outputs, _) = self.network(packed_batch_x)
        else:
            packed_outputs, hidden_outputs = self.network(packed_batch_x)
        
        # unpack the packed sequence. outputs has shape (batch_size, max_seq_len, D * hidden_size)
        # without attention, outputs is NOT USED.   
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)

        # shape of `hidden_state_all_layers`= (num_layers, batch_size, hidden_size)
        hidden_state_all_layers = hidden_outputs[:, :, :]
        if self.arch['bidirect']:
            # here, we need to process bidirectional final hidden states of each layer through a linear layer
            req_shape = (hidden_outputs.shape[0]//2, hidden_outputs.shape[1], hidden_outputs.shape[2])
            hidden_state_all_layers = torch.zeros(req_shape).to(self.device)
            for idx in range(self.arch['num_layers']):
                # concatenate the forward and reverse directions outputs along the hidden_size's dimension
                # shape = (batch_size, 2 * hidden_size) now
                concat_hidden_state_cur_layer = torch.cat([hidden_outputs[2*idx, :, :], hidden_outputs[2 * idx + 1, :, :]], dim=1).to(self.device)
                hidden_state_cur_layer = self.combine_forward_backward[idx](concat_hidden_state_cur_layer)
                hidden_state_cur_layer = torch.tanh(hidden_state_cur_layer)
                hidden_state_all_layers[idx, :, :] = hidden_state_cur_layer

        # outputs - shape (batch_size, max_seq_len, D * hidden_size) -> NOT USED without attention
        # hidden_state_all_layers - shape (num_layers, batch_size, hidden_size)
        return outputs, hidden_state_all_layers


'''
    Class for Attention
'''
class Attention(lt.LightningModule):
    def __init__(self, hidden_dim, bidirect = False):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.bidirect = bidirect
        self.attn_matrix_indim = 3 * hidden_dim if bidirect == True else 2 * hidden_dim
        self.U = nn.Linear(self.attn_matrix_indim, hidden_dim) # to be sent to tanh layer
        self.V = nn.Linear(self.hidden_dim, 1, bias=False) # dotted with tanh layer's output to get weights
        self.softmaxlayer = nn.Softmax(dim=1)

    def forward(self, prev_dec_hidden, padded_enc_outputs, mask):
        ''' 
            prev_dec_hidden -> shape (batch_size, hidden_size) - first decoder layer's hidden state
            padded_enc_outputs -> shape (batch_size, max_seq_len, hidden_size)
            mask -> shape (batch_size, max_seq_len) with 1 in locations where pad token is present
            max_seq_len = max_seq_len in batch_X [that was processed by encoder]
        '''
        batch_size, max_seq_len, _ = padded_enc_outputs.shape
        hidden_extended = prev_dec_hidden.unsqueeze(1).repeat(1, max_seq_len, 1)
        # hidden_extended shape = (batch_size, max_seq_len, hidden_size)
        U_input = torch.cat([hidden_extended, padded_enc_outputs], dim=2)
        # U_input shape = (batch_size, max_seq_len, [2 or 3] * hidden_size)
        tanh_output = torch.tanh(self.U(U_input))
        # tanh_output shape = (batch_size, max_seq_len, hidden_size)
        attn_weights = self.V(tanh_output).squeeze(2)
        # attn_weights shape = (batch_size, max_seq_len)
        attn_weights = torch.masked_fill(attn_weights, mask==1, -1e12)
        # fill pad locations with very small values to be zeroed by softmax
        attn_weights = self.softmaxlayer(attn_weights)
        # convert weights to probabilities over max_seq_length dimension
        return attn_weights

'''
    Class for the Decoder Network
'''
class DecoderNet(lt.LightningModule):
    def __init__(self, vocab_size, embed_size, num_layers, hid_size, cell_type, 
                 attention=False, attn_layer=None, enc_bidirect=False, dropout=0):
        super(DecoderNet, self).__init__()
        # store all the network arch information
        self.arch = {
            'vocab_size' : vocab_size,
            'hid_size' : hid_size,
            'embed_size' : embed_size,
            'num_layers' : num_layers,
            'cell' : cell_type,
            'enc_bidirect' : enc_bidirect,
            'attention' : attention
        }

        # create the embedding layer with a dropout layer for it
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(p=dropout)

        # create the linear layer for producing output logits (need to be sent through softmax to get 
        # char probabilities). But, we can use CrossEntropyLoss() on this logits directly
        self.out_layer = nn.Linear(hid_size, vocab_size)

       # we create the required architecture using the received parameters
       # now, input_size will be [embed_size + 2 * hidden_size] if enc_bidirect
       #                    and [embed_size + hidden_size] otherwise
        kwargs = {'hidden_size':hid_size, 'num_layers':num_layers, 
                 'batch_first':True}
        if attention:
            kwargs['input_size'] = embed_size + 2 * hid_size if enc_bidirect else embed_size + hid_size
        else:
            kwargs['input_size'] = embed_size
        
        # create the network using kwargs
        if num_layers > 1:
            kwargs['dropout'] = dropout
        if cell_type == 'RNN':
            self.network = nn.RNN(**kwargs)
        elif cell_type == 'LSTM':
            self.network = nn.LSTM(**kwargs)
        else:
            self.network = nn.GRU(**kwargs)
        
        if attention:
            # save the attention object
            self.attn_layer = attn_layer

    # will always go 1 step forward in time (seqlen = L = 1)
    def forward(self, batch_y, prev_decoder_state, padded_enc_outputs=None, mask=None):
        ''' 
            batch_y -> shape (batch_size) = decoder input with vocabulary indices from target language
            prev_decoder_state(RNN, GRU) -> shape (num_layers, batch_size, hidden_size) 
            prev_decoder_state(LSTM) -> tuple of prev_hidden_state, prev_cell_state
                                        both have shape (num_layers, batch_size, hidden_size) ()
        
        -- optional (only when arch['attention'] = True)
            padded_enc_outputs -> shape (batch_size, max_enc_seq_len, (2 or 1) * hid_size)
                                        2 if encoder is bidirectional
            mask -> shape (batch_size, max_enc_seq_len) -> 1 where pad token is present
        '''
        # we add a dummy dimension for seqlen = 1. new shape of batch_y is (batch_size, 1)
        batch_y = batch_y.unsqueeze(1) 
        # pass through embedding and dropout layers. new shape of batch_y is (batch_size, 1, hidden_size)
        embedded_batch_y = self.embedding(batch_y)
        embedded_batch_y = self.dropout(embedded_batch_y)

        if self.arch['attention']:
            # unpack the decoder state and compute attention
            if self.arch['cell'] == 'LSTM':
                decoder_hidden_state, _ = prev_decoder_state
            else:
                decoder_hidden_state = prev_decoder_state

            attn_weights = self.attn_layer(decoder_hidden_state[0, :, :], padded_enc_outputs, mask)
            # recall attn_weights shape = (batch_size, max_enc_seq_len)
            attn_weights = attn_weights.unsqueeze(1)
            attn_weighted_enc_outputs = torch.bmm(attn_weights, padded_enc_outputs)
            # attn_weighted_enc_outputs shape = (batch_size, 1, (2 or 1) * hid_size)
            dec_input = torch.cat([attn_weighted_enc_outputs, embedded_batch_y], dim=2)
        else:
            dec_input = embedded_batch_y
            attn_weights = None

        # pass dec_input through the network
        outputs, new_decoder_state = self.network(dec_input, prev_decoder_state)
        outputs = outputs.squeeze(1) # remove seqlen dimension. shape = (batch_size, hidden_size)
        # pass outputs through the linear layer to get the logits (shape = (batch_size, out_vocab_size))
        logits = self.out_layer(outputs)
        # attention weights is required for visualization
        return logits, new_decoder_state, attn_weights

''' 
    Class for encapsulating the encoder and decoder networks
'''
class EncoderDecoder(lt.LightningModule):
    def __init__(self, encoder :EncoderNet, decoder : DecoderNet, src_lang : Language, tar_lang : Language) -> None:
        super(EncoderDecoder, self).__init__()
        # store the encoder and decoder along with language objects in the class
        self.enc_model = encoder
        self.dec_model = decoder
        self.src_lang = src_lang
        self.tar_lang = tar_lang
        self.cell_type = self.dec_model.arch['cell']
        # we require num of enc layers == num of dec layers as we connect encoder and decoder
        # layer by layer
        assert(self.enc_model.arch['num_layers'] == self.dec_model.arch['num_layers'])
        self.num_layers = self.enc_model.arch['num_layers']
        assert(self.enc_model.arch['hid_size'] == self.dec_model.arch['hid_size'])
        self.attention = self.dec_model.arch['attention']
    
    # function to make mask for batch_X where mask == 1 iff pad token is present in that location
    def make_mask(self, batch_X):
        return torch.where(batch_X == self.tar_lang.sym2index[PAD_SYM], 1, 0).to(self.device)
    
    def forward(self, batch_X, batch_y, X_lens, tf_ratio=None):
        ''' 
            batch_X -> shape (batch_size, max_batch_X_seq_len) - padded input to encoder
            batch_y -> shape (batch_size, max_batch_y_seq_len) - padded input to decoder
            X_lens -> list of true (unpadded) lengths of the sequences in batch_X
        '''
        # compute batch_size and send batch_X through the encoder
        batch_size = batch_X.size(0)
        enc_outputs, final_enc_hidden_state = self.enc_model(batch_X, X_lens)
        # recall final_enc_hidden_state -> shape (num_layers, batch_size, hidden_size)
        
        # make padding mask for batch_X
        pad_mask = self.make_mask(batch_X)
        tarlength = batch_y.size(1) # max seq length of batch_y

        # outlogits -> tensor for storing the output logits (softmax to get post prob.)
        outlogits = torch.zeros(batch_size, tarlength, self.dec_model.arch['vocab_size']).to(self.device)
        # preds -> tensor for storing argmax(logits) over the target vocab for each example and each time step
        preds = torch.zeros(batch_size, tarlength).to(self.device)

        dec_input = batch_y[:,0] # get initial input for decoder -> SOS tokens with shape (batch_size)
        decoder_state = final_enc_hidden_state # initially decoder hidden state = final_enc_hidden_state

        if (self.cell_type == 'LSTM'):
            # for LSTM, cell state is initialized to zero and is added to decoder state
            init_dec_cell_state = torch.zeros_like(decoder_state).to(self.device)
            decoder_state = (decoder_state, init_dec_cell_state)
        
        # for each timestep
        for tstep in range(1, tarlength):
            # send the dec_input through the decoder. we ignore the attn_weights here.
            if self.attention:
                curlogits, decoder_state, _ = self.dec_model(dec_input, decoder_state, enc_outputs, pad_mask)
            else:
                curlogits, decoder_state, _ = self.dec_model(dec_input, decoder_state)
            # recall curlogits -> shape (batch_size, out_vocab_size); decoder_state -> shape invariant.
            tf_input = batch_y[:, tstep] # dec input for next time step if teacher forcing is chosen
            # pred -> argmax along vocab_size (dim = 1) to get class labels. shape = (batch_size)
            pred = torch.argmax(curlogits, dim=1).to(self.device)
            # greedy dec input is whatever set of words was predicted previously. shape = (batch_size)
            dec_input = pred
            # change dec input to tf input with prob = tf_ratio
            if tf_ratio != None and torch.rand(1)[0] < tf_ratio:
                dec_input = tf_input
            # store curlogits (for loss backprop) and pred (for predicted word construction)
            # for the current timestep
            outlogits[:, tstep, :] = curlogits 
            preds[:, tstep] = pred
        # NOTE - outlogits[:, 0, :] -> is a dummy tensor. should be discarded in loss computation
        # Similarly preds[:, 0] -> is also to be ignored. It has only 0s (=SOS_SYM).
        return outlogits, preds
    
    # word by word inference is done as it is easier.
    def greedy_inference(self, X, max_dec_length=25):
        ''' 
            X -> shape (1, X_len) - single word in 1D tensor (generated by sym2index)
            max_dec_length -> length beyond which decoding is stopped
        '''
        # convert X should be a batch with size 1
        _, final_enc_hidden_state = self.enc_model(X, [X.size(1)])
        # recall final_enc_hidden_state -> shape (num_layers, batch_size, hidden_size)
        
        # outlogits -> tensor for storing the output logits (softmax to get post prob.)
        outlogits = torch.zeros(max_dec_length, self.dec_model.arch['vocab_size']).to(self.device)
        
        # preds -> tensor for storing argmax(logits) over the target vocab for each example at each time step
        preds = torch.zeros(max_dec_length).to(self.device)
        dec_input = torch.tensor(self.tar_lang.sym2index[SOS_SYM]).to(self.device)
        # get initial input for decoder -> SOS tokens with shape (batch_size)
        decoder_state = final_enc_hidden_state # initially decoder hidden state = final_enc_hidden_state

        if (self.cell_type == 'LSTM'):
            # for LSTM, cell state is initialized to zero and is added to decoder state
            init_dec_cell_state = torch.zeros_like(decoder_state).to(self.device)
            decoder_state = (decoder_state, init_dec_cell_state)

        # for each timestep
        for tstep in range(1, max_dec_length):
            # send the dec_input through the decoder.
            curlogits, decoder_state = self.dec_model(dec_input, decoder_state)
            # recall curlogits -> shape (batch_size, out_vocab_size); decoder_state -> shape invariant.
            # pred -> argmax along vocab_size (dim = 1) to get class labels. shape = (batch_size)
            pred = torch.argmax(curlogits, dim=1).to(self.device)
            # greedy dec input is whatever set of words was predicted previously. shape = (batch_size)
            dec_input = pred
            # store curlogits and pred (for predicted word construction) for the current timestep
            outlogits[tstep, :] = curlogits 
            preds[tstep] = pred

        # generate predicted words using preds tensor and return it.
        pred_words = self.tar_lang.convert_to_words(preds.unsqueeze(0).cpu().numpy())
        return pred_words