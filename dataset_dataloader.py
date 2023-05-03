from torch.utils.data import DataLoader, Dataset
from language import *

# A Dataset class to help with creating the dataset. Converts the data into numbers
# using the source and target languages' sym2index and index2sym dictionaries
class TransliterateDataset(Dataset):
    def __init__(self, x_data, y_data, src_lang : Language, tar_lang : Language):
        # save all the data points and the language objects
        self.x_data = x_data
        self.y_data = y_data
        self.src_lang = src_lang
        self.tar_lang = tar_lang
        
    def __len__(self):
        # needs to be implemented for a pytorch `Dataset`
        return len(self.y_data)

    def __getitem__(self, idx):
        # gives the data point (X, y) at index = idx
        # we convert them to a tensor of numbers using the Language objects
        # also returns the word y for ease of computing accuracy later
        x_enc, y_enc = self.x_data[idx], self.y_data[idx]
        x_enc = self.src_lang.convert_to_numbers(x_enc)
        y_enc = self.tar_lang.convert_to_numbers(y_enc) 
        return torch.tensor(x_enc, dtype=int), torch.tensor(y_enc,dtype=int)

# This is a collation function for post-processing a batch in a DataLoader. We sort the instances (X,y) in a batch
# based on seq length of X in desc order and create a padded batch to help with batch-processing in recurrent
# networks
class CollationFunction:
    def __init__(self, src_lang : Language, tar_lang : Language):
        self.src_lang = src_lang
        self.tar_lang = tar_lang
    
    def __call__(self, batch):
        # reasoning : https://stackoverflow.com/questions/51030782/why-do-we-pack-the-sequences-in-pytorch
        src, tar = zip(*batch)
        src_lens = torch.tensor([len(x) for x in src], dtype=int)
        # pad both the X part(src) and y part(tar) with PAD_SYM
        src = nn.utils.rnn.pad_sequence(list(src), batch_first=True, padding_value=self.src_lang.get_index(PAD_SYM))
        tar = nn.utils.rnn.pad_sequence(list(tar), batch_first=True, padding_value=self.tar_lang.get_index(PAD_SYM))
        # return padded batch_X (src), padded batch_Y (tar), X_lens (needed for unpacking) and y words(tar_words)
        # each entry in tar_words is a string
        return src, tar, src_lens