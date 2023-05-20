# CS6910 Assignment 3
Third Assignment of the Deep Learning Course (CS6910), Summer 2023. Task is to perform Transliteration on a sampled subset of [aksharantar dataset](https://drive.google.com/file/d/1uRKU4as2NlS9i8sdLRS1e326vQRdhvfw/view) using recurrent neural network based seq2seq models. [Lightning](https://github.com/Lightning-AI/lightning) (previously called pytorch-lightning) has been used for this assignment. It provides tensorboard support and contains library callback methods for custom checkpointing and early stopping. Most importantly, it eliminates boilerplate code.

**Wandb Report** : [link](https://wandb.ai/cs19b021/cs6910-assignment3/reports/CS6910-DL-Assignment-3--Vmlldzo0MTY1NjU3)

# Code Details

## Source Files
1. **language.py** - Contains a function to load the data from [directory](./aksharantar_sampled) and a class **Language** which generates the sym2index and index2sym mappings for all characters of a language using training data. The special SOS, EOS, PAD and UNK tokens are defined here. 
2. **dataset_dataloader.py** - Contains the definitions of a pytorch class for the dataset and also the collation function to be used for postprocessing a batch in the dataloader. The collation function is used to pad all data in a batch to the same length to allow for batch training and batch validation. Later, the padded sequences in a batch are ['packed'](https://stackoverflow.com/questions/51030782/why-do-we-pack-the-sequences-in-pytorch) to avoid computation wastage.
3. **encoder_decoder.py** - Contains classes for defining the encoder, attention mechanism and the decoder. Also, these classes are encapsulated in a container class **EncoderDecoder** which represents our seq2seq model. All these classes are inherit from lt.LightningModule as we use lightning.
4. **runner.py** - Contains a class called Runner which is a subclass of lt.LightningModule. This class encapsulates and initializes an encoder_decoder model. It contains the methods needed by lightning to support loading data (setup, train_dataloader, val_dataloader, test_dataloader), training (training_step, validation_step, on_validation_epoch_end, on_train_epoch_end) and testing (test_step, on_test_epoch_end). All required metrics like train loss, train acc, val loss, val acc, test loss, test acc, test predictions and attention heatmaps are tracked manually or by using lightning support. Additional methods to retrieve test predictions, initialize weights, and compute accuracy(a prediction is correct iff it is equal to the target) are present.
5. **plotting_code_and_helpers.py** - Contains code to plot the attention heatmaps in a 3 x 3 plotly grid, sample and plot the predictions with color-coding (based on edit distance from target) in a plotly table and save all predictions to a .csv file.

## Notebooks 
1. **best_with_attention.ipynb** - Here, we load the [model](./best_checkpoints/attention/emb=192_layers=1_hid=256_cell=LSTM_bidirectional=True_dr=0_itfr=0.7_bsize=128_att=True_opt=Adam_lr=0.002.ckpt) logged by the best hyperparameter combination in the [wandb sweep](https://wandb.ai/cs19b021/cs6910-assignment3/sweeps/fbk84w2d) where an attention mechanism is present. Then, there is code to perform 1 validation epoch, test the model on test data, display test metrics, log some predictions and attention heatmaps onto wandb. Also, this notebook will save all predictions on the test data in a .csv file within [predictions_attention](./predictions_attention).
2. **best_without_attention.ipynb** - Here, we load the [model](best_checkpoints/no_attention/emb=64_layers=3_hid=256_cell=LSTM_bidirectional=False_dr=0.2_itfr=0.8_bsize=32_att=False_opt=Adam_lr=0.002.ckpt) logged by the best hyperparameter combination in the [wandb sweep](https://wandb.ai/cs19b021/cs6910-assignment3/sweeps/rlqfx0nb) without an attention mechanism. Again, we do validation, testing, display test metrics and log some predictions onto wandb. Also, this notebook will save all predictions on the test data in a .csv file within [predictions_vanilla](./predictions_vanilla).
3. **bug_testing.ipynb** - Here, the functionality of the code is tested to  identify any bugs. 24 models with hyperparameters over (num_layers={1,3}, cell_type={RNN, LSTM, GRU}, bidirectional={True, False}, attention={True, False}) are trained for 2 train steps to perform this bug detection.
4. **start_sweep.ipynb** - Code to start a sweep with a given hyperparameter search space and search strategy. It is easy to change this search space and strategy if desired. Requires an user to be logged into wandb via terminal beforehand.
5. **sweep_agent.ipynb** - Code to start a new sweep agent to run sweep configurations received from the wandb sweep server. Necessary to provide a sweep_id for this purpose. The agent will test 10 combinations (changeable - we use 10 to avoid overshooting execution time limits on kaggle). Early stopping and wandb model logging is also done. Also, by default, source language is ENGLISH and target language is TAMIL. This is changeable.

## Main Code File  
**main.py** - A python script that parses its commandline arguments into hyperparameters, generates a model, trains the model with early stopping + checkpointing, runs inference on test data, displays test metrics and saves all test predictions in a new local file './predictions.csv'.

## Requirements
**colab_extra_requirements.txt** - Contains the extra libraries needed by the code to run on colab. To run locally, these libraries should be installed and in case of any import errors, the missing libraries should be installed. No special library other than lightning, wandb, plotly, pytorch and standard libraries like numpy, pandas and argparse is used.

## Directories
1. **aksharantar_sampled** - Contains the [subset of aksharantar dataset](https://drive.google.com/file/d/1uRKU4as2NlS9i8sdLRS1e326vQRdhvfw/view) used for this assignment.
2. **best_checkpoints** - Contains subfolders that have the checkpoint artifacts logged by the best runs from the **attention sweep** and **no attention sweep**. The .ckpt files are properly named to indicate the hyperparameter combination used by the corresponding runs that generated them.
3. **predictions_vanilla** - Contains the predictions made by the best model **without any attention mechanism.**
4. **predictions_attention** - Contains the predictions made by the best model **with attention mechanism.**

# Usage

The main code file is **main.py**. To get detailed help instruction with all the features, run 

```bash
python3 main.py -h
```

The default hyperparameter values are those of the best model for tamil data. The default target language is also TAMIL. To use the default commandline args, run

```bash
python3 main.py
```

The complete set of commandline arguments supported are

```bash
usage: main.py [-h] [-src {eng}] [-tar TARGET] [-em EMBEDDING_SIZE] [-nl NUMBER_OF_LAYERS] [-hs HIDDEN_SIZE] [-cl {RNN,GRU,LSTM}]
               [-bi {True,False}] [-dr DROPOUT] [-itr INITIAL_TF_RATIO] [-bs BATCH_SIZE] [-at {True,False}] [-op {Adam}] [-lr LEARNING_RATE]
               [-max_ep MAX_EPOCHS] [-pa PATIENCE] [-min_ep MIN_EPOCHS] [-min_imp MIN_DELTA_IMP]
```
Other hyperparameter combinations may be tested using the details from the help menu.

# Implementation Details
- We will require that number of encoder layers must be equal to number of decoder layers in our architecture. This is required because we will follow the popular technique wherein **the final hidden state of encoder layer i is the initial hidden state of decoder layer i** where 
$1 \leq i \leq \text{NUMBER OF LAYERS}$.
- Also, we use the same embedding size, same hidden state size, same cell {rnn/gru/lstm} for both the encoder and decoder. This is a design choice and may be easily changed. Note that supporting different hidden state sizes between encoder and decoder may require the use of linear layers for connecting them.
- Only ADAM optimizer is currently allowed. 1 line modifications can be made to support other optimizers.
- When the encoder is bidirectional, the final hidden state of left2right, right2left components of each encoder layer are concatenated together and passed through a linear layer to get initial hidden state of decoder layer i. Each encoder layer has a separate linear layer for this purpose. 
- Teacher forcing is required to make the model learn during the initial training epoches. We use a schedule to do this, the teacher forcing ratio remains a high constant initially, then undergoes a constant decrease-per-epoch for a few epoches and finally, becomes constant at a very small value 0.01. This is done to ensure stability and smoothly transition from **heavy-teacher-forcing** to **negligible-teacher-forcing**.
- Pytorch RNN-based models have known non-determinism issues. Read warning section [here](https://pytorch.org/docs/stable/generated/torch.nn.RNN.html). This is the reason why we chose to log the best(highest-val-acc) model of every run onto wandb as an artifact. The models of the best runs from each of the 2 wandb sweeps are downloaded and saved in [best_checkpoints](./best_checkpoints).
