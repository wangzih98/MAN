from torch.utils.data import Dataset
#import Pickle as pickle
import pickle
import numpy as np
import h5py

from IPython import embed

AUDIO = 'covarep'
VISUAL = 'facet'
TEXT = 'glove'
LABEL = 'label'

def total(params):
    '''
    count the total number of hyperparameter settings
    '''
    settings = 1
    for k, v in params.items():
        settings *= len(v)
    return settings



def load_mosi_3d_(data_path):

    # parse the input args
    class MOSI(Dataset):
        '''
        PyTorch Dataset for MOSI, don't need to change this
        '''

        def __init__(self, audio, visual, text, labels):
            self.audio = audio
            self.visual = visual
            self.text = text
            self.labels = labels

        def __getitem__(self, idx):
            return [self.audio[idx, :], self.visual[idx, :], self.text[idx, :, :], self.labels[idx]]

        def __len__(self):
            return self.audio.shape[0]
        
    
    
    root = 'cmumosi/'
    
    mosi_data_train = np.array(h5py.File(root +'X_train.h5','r')['data'])
    mosi_data_val = np.array(h5py.File(root + 'X_valid.h5','r')['data'])
    mosi_data_test = np.array(h5py.File(root + 'X_test.h5','r')['data'])
    
    mosi_train_label = np.array(h5py.File(root + 'y_train.h5','r')['data'])
    mosi_valid_label = np.array(h5py.File(root + 'y_valid.h5','r')['data'])
    mosi_test_label = np.array(h5py.File(root + 'y_test.h5','r')['data'])
    
    mosi_train_audio = mosi_data_train[:,:,300:305]
    mosi_valid_audio = mosi_data_val[:,:,300:305]
    mosi_test_audio = mosi_data_test[:,:,300:305]
        
    mosi_train_visual = mosi_data_train[:,:,305:] 
    mosi_valid_visual = mosi_data_val[:,:,305:]
    mosi_test_visual = mosi_data_test[:,:,305:]
    
    mosi_train_text = mosi_data_train[:,:,:300]
    mosi_valid_text = mosi_data_val[:,:,:300]
    mosi_test_text = mosi_data_test[:,:,:300]
    
    

    train_audio, train_visual, train_text, train_labels \
        = mosi_train_audio, mosi_train_visual, mosi_train_text, mosi_train_label
    valid_audio, valid_visual, valid_text, valid_labels \
        = mosi_valid_audio, mosi_valid_visual, mosi_valid_text, mosi_valid_label
    test_audio, test_visual, test_text, test_labels \
        = mosi_test_audio, mosi_test_visual, mosi_test_text, mosi_test_label

    
    
    train_audio[1263] = np.zeros([train_audio.shape[1],train_audio.shape[2]])
    train_audio[844] = np.zeros([train_audio.shape[1],train_audio.shape[2]])
    train_audio[602] = np.zeros([train_audio.shape[1],train_audio.shape[2]])
    #train_audio[133] = np.zeros([train_audio.shape[1],train_audio.shape[2]])
    valid_audio[149] = np.zeros([valid_audio.shape[1],valid_audio.shape[2]])
    

    ### avoiding NAN

    test_audio[654] = np.zeros([test_audio.shape[1],test_audio.shape[2]])
    test_audio[674] = np.zeros([test_audio.shape[1],test_audio.shape[2]])
    test_audio[465] = np.zeros([test_audio.shape[1],test_audio.shape[2]])
    test_audio[459] = np.zeros([test_audio.shape[1],test_audio.shape[2]])
    test_audio[443] = np.zeros([test_audio.shape[1],test_audio.shape[2]])
    test_audio[361] = np.zeros([test_audio.shape[1],test_audio.shape[2]])
    test_audio[310] = np.zeros([test_audio.shape[1],test_audio.shape[2]])
    test_audio[301] = np.zeros([test_audio.shape[1],test_audio.shape[2]])
    test_audio[237] = np.zeros([test_audio.shape[1],test_audio.shape[2]])
    test_audio[114] = np.zeros([test_audio.shape[1],test_audio.shape[2]])
    test_audio[63] = np.zeros([test_audio.shape[1],test_audio.shape[2]])
    test_audio[11] = np.zeros([test_audio.shape[1],test_audio.shape[2]])
    
    
    print(train_visual.shape)
    print(train_text.shape)
    print(train_labels.shape)

    # code that instantiates the Dataset objects
    train_set = MOSI(train_audio, train_visual, train_text, train_labels)
    valid_set = MOSI(valid_audio, valid_visual, valid_text, valid_labels)
    test_set = MOSI(test_audio, test_visual, test_text, test_labels)


    audio_dim = train_set[0][0].shape[1]
    audio_lens = train_set[0][0].shape[0]
    print("Audio feature dimension is: {}".format(audio_dim))
    visual_dim = train_set[0][1].shape[1]
    visual_lens = train_set[0][1].shape[0]
    print("Visual feature dimension is: {}".format(visual_dim))
    text_dim = train_set[0][2].shape[1]
    text_lens = train_set[0][2].shape[0]
    print("Text feature dimension is: {}".format(text_dim))
    input_dims = (audio_dim, visual_dim, text_dim)
    input_lens = (audio_lens, visual_lens, text_lens)
    
    
    # remove possible NaN values
    train_set.visual[train_set.visual != train_set.visual] = 0
    valid_set.visual[valid_set.visual != valid_set.visual] = 0
    test_set.visual[test_set.visual != test_set.visual] = 0

    train_set.audio[train_set.audio != train_set.audio] = 0
    valid_set.audio[valid_set.audio != valid_set.audio] = 0
    test_set.audio[test_set.audio != test_set.audio] = 0


    return train_set, valid_set, test_set, input_dims, input_lens







