# The main file of the folder. Contains code to load data, create the model with the given hyperparameters
# train the model and finally, display the metrics of the model on test data. Also, the best model (the model with the highest 
# validation accuracy) is saved in a new subfolder (name = "checkpoints/model.ckpt"). Finally, the predictions on the test data are saved in
# "predictions.csv".  We also use early stopping with patience of 5 here. A few other hyperparameters like max_epochs 
# and min_epochs are also fixed. These hyperparameters are hardcoded but are easy to change.
# Also, no wandb logging is done here. See code in sweep_agent where logging is done. Also, default values for the hyperparameters
# are those of the best model from the wandb sweeps.
import lightning as lt
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
import torch
from language import *
from dataset_dataloader import *
from encoder_decoder import *
from plotting_code_and_helpers import *
from runner import Runner
import argparse as ap
import os, shutil

# we will ignore num_workers suggestions/warnings from pytorch-lightning
import warnings
warnings.filterwarnings("ignore")

# function to create a parser that parses the commandline arguments. Default values are the values from
# the best hyperparameter combination found from wandb sweeps. 
# NOTE - change the default values to that of the best model
def gen_parser():
    parser = ap.ArgumentParser(description='Recurrent network based encoder-decoder model for transliteration task. Ensure that "aksharantar_sampled" is present in this directory as downloaded without any change.')
    parser.add_argument('-src', '--source', dest='source', default='eng', help='source language (english) sub-folder name.')
    parser.add_argument('-tar', '--target', dest='target', default='tam', help='target language sub-folder name.')
    parser.add_argument('-em', '--embedding_size', dest='embedding_size', default=64, type=int, help='common embedding size of encoder and decoder')
    parser.add_argument('-nl', '--number_of_layers', dest='number_of_layers', default=1, type=int, help='common number of layers in encoder and decoder')
    parser.add_argument('-hs', '--hidden_size', dest='hidden_size', default=64, type=int, help='common hidden state size in encoder and decoder')
    parser.add_argument('-cl', '--cell', dest='cell', default='LSTM', choices=['RNN', 'GRU', 'LSTM'], help='Type of RNN-based cell to be used in the model')
    parser.add_argument('-bi', '--bidirectional', dest='bidirectional', default='True', choices=['True', 'False'], help='True iff encoder is bidirectional')
    parser.add_argument('-dr', '--dropout', dest='dropout', default=0.05, type=float, help='dropout probability in dropout layers')
    parser.add_argument('-itr', '--initial_tf_ratio', dest='initial_tf_ratio', default=0.8, type=float, help='Initial value of teacher-forcing ratio to be used in our tf ratio scheduler')
    parser.add_argument('-bs', '--batch_size', dest='batch_size', default=64, type=int, help='Batch size to be used during training and validation')
    parser.add_argument('-at', '--attention', dest='attention', default='True', choices=['True', 'False'], help='True iff decoder has an attention layer')
    parser.add_argument('-op', '--optimizer', dest='optimizer', default='Adam', choices=['Adam'], help='Optimizer to be used in the model. Currently, only ADAM is present but it is easy to add more.')
    parser.add_argument('-lr', '--learning_rate', dest='learning_rate', default=2e-3, type=float, help='Learning rate to be used in the model')
    parser.add_argument('-max_ep', '--max_epochs', dest='max_epochs', default=35,type=int, help='Maximum number of training epochs allowed.')
    parser.add_argument('-pa', '--patience', dest='patience', default=5,type=int, help='Patience for early stopping callback')
    parser.add_argument('-min_ep', '--min_epochs', dest='min_epochs', default=12,type=int, help='Mandatory minimum number of training epochs to be done. Necessary to override early stopping in the initial epochs.')
    parser.add_argument('-min_imp', '--min_delta_imp', dest='min_delta_imp', default=1e-3,type=float, help='Minimum increase that is counted as "improvement" in early stopping.')
    return parser

if __name__ == '__main__':
    parser = gen_parser()
    args = parser.parse_args()

    # fix type of bidirectional and attention arguments
    if args.bidirectional == 'True':
        args.bidirectional = True
    else:
        args.bidirectional = False
    if args.attention == 'True':
        args.attention = True
    else:
        args.attention = False

    # print the accelerator available
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # convert all cmd line arguments to a dictionary
    args = vars(args)

    # load all the available data and print sample counts for each set
    x_train, y_train = load_data(args['target'], 'train')
    x_valid, y_valid = load_data(args['target'], 'valid')
    x_test, y_test = load_data(args['target'], 'test')

    print(f'Number of train samples = {len(x_train)}')
    print(f'Number of valid samples = {len(x_valid)}')
    print(f'Number of test samples = {len(x_test)}')

    # create language objects for storing vocabulary, index2sym and sym2index
    SRC_LANG = Language(args['source'])
    TAR_LANG = Language(args['target'])

    # creating vocabulary using train data only
    SRC_LANG.create_vocabulary(*(x_train))
    TAR_LANG.create_vocabulary(*(y_train))

    # generate mappings from characters to numbers and vice versa
    SRC_LANG.generate_mappings()
    TAR_LANG.generate_mappings()

    # print the source and target vocabularies
    print(f'Source Vocabulary Size = {len(SRC_LANG.symbols)}')
    print(f'Source Vocabulary = {SRC_LANG.symbols}')
    print(f'Source Mapping {SRC_LANG.index2sym}')
    print(f'Target Vocabulary Size = {len(TAR_LANG.symbols)}')
    print(f'Target Vocabulary = {TAR_LANG.symbols}')
    print(f'Target Mapping {TAR_LANG.index2sym}')

    # dictionary to pass to a model (instance of Runner Class)
    rdict = dict(
                SOURCE=args['source'],
                TARGET=args['target'],
                src_lang=SRC_LANG,
                tar_lang=TAR_LANG,
                common_embed_size=args['embedding_size'],
                common_num_layers=args['number_of_layers'],
                common_hidden_size=args['hidden_size'],
                common_cell_type=args['cell'],
                init_tf_ratio= args['initial_tf_ratio'],
                enc_bidirect=args['bidirectional'],
                attention=args['attention'],
                dropout=args['dropout'],
                opt_name=args['optimizer'],
                learning_rate=args['learning_rate'],
                batch_size=args['batch_size'] 
    )

    # create a new checkpoints folder in working directory
    if os.path.exists('./checkpoints/'):
        shutil.rmtree('./checkpoints/')
    os.mkdir('./checkpoints/')

    #### TRAINING SECTION ####

    # create the model using the arguments in rdict(sources from cmdline)
    runner = Runner(**rdict)
    # early stop if val_acc does not improve by min_delta for patience many epochs
    early_stop_callback = EarlyStopping(monitor="val_acc", min_delta=args['min_delta_imp'], patience=args['patience'], verbose=True, mode="max")
    # we checkpoint the model when val_acc improves in the working directory.
    chkCallback = ModelCheckpoint(dirpath='./checkpoints/', filename=f'model', monitor='val_acc', mode='max')
    trainer = lt.Trainer(min_epochs=args['min_epochs'], max_epochs=args['max_epochs'], callbacks=[chkCallback, early_stop_callback])
    # train the model using pytorch lightning
    trainer.fit(runner)


    #### TESTING SECTION ####
    
    # load the best model (saved locally in 'model.ckpt')
    runner = Runner.load_from_checkpoint('./checkpoints/model.ckpt', **rdict)
    trainer.test(runner)

    # get the test results and unpack it (acc. to presence of attention layer)
    ret_info = runner.get_test_results()
    if args['attention']:
        src_list, tar_true_list, tar_pred_list, _ = ret_info
    else:
        src_list, tar_true_list, tar_pred_list = ret_info

    # save the predictions in predictions.csv
    save_predictions_file(src_list, tar_true_list, tar_pred_list, 'predictions')