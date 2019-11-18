import numpy as np
import os, sys
import pandas as pd
from scipy.signal import savgol_filter

import torch
import torch.nn as nn

import pickle
from random import shuffle
import copy
# interpret tactile sensor data on iCub.
# ref: http://wiki.icub.org/wiki/Tactile_sensors_(aka_Skin)

# name of the object
object_names = ['bathTowel',
                 'cardBoard',
                 'carpetNet',
                 'cork',
                 'cotton',
                 'cuisionFoam',
                 'denim',
                 'eva',
                 'fakeLeather',
                 'felt',
                 'fiberBoard',
                 'metal',
                 'paper1',
                 'paper2',
                 'polypropileno',
                 'polypropilenoSmooth',
                 'softMaterial1',
                 'softMaterial2',
                 'spongeWhiteSmall',
                 'styrofoam',
                 'thinPolypropylene',
                 'woodHard',
                 'yogaMat']


# select 5 representative samples (random selection for now)
# random selection on Nov 11
# sub_obj_names = ['bathTowel', 'denim', 'metal', 'paper1', 'yogaMat', 'woodHard', 'styrofoam', 'felt']

# specify the initial classes
# # 8 similar classes
# init_obj_names = ['polypropileno', 'polypropilenoSmooth', 'eva', 'thinPolypropylene', 'cuisionFoam', 'fakeLeather', 'styrofoam', 'spongeWhiteSmall']
# # 5 different classes
init_obj_names = ['bathTowel', 'cardBoard', 'eva', 'paper1', 'styrofoam', 'metal', 'yogaMat', 'cotton']

# shuffle the remaining classes that are to be added later
obj_names = copy.deepcopy(object_names)
for obj in init_obj_names:
    try:
        obj_names.remove(obj)
    except ValueError:
        print(obj, "not found")
# print("obj_names")
# print(obj_names)
# print("init_obj_names")
# print(init_obj_names)
shuffle(obj_names)
# concatenate to obtain the full list
object_names = init_obj_names + obj_names 

# TODO: select representative types
# shuffle(object_names)
# print(object_names)
# selected window size
DEPTH_SIZE = 75

# acceleration and decelaration angle size (in degrees)
epsilon_lim = 1

lower_lim = 30+epsilon_lim
upper_lim = 90-epsilon_lim

# change name conventions
new_names = {}
for i in range(1,386):
    new_names[i] = i-1
    
# these taxels are meaningless: every 7 and 11 (out 12)
meaningless_taxels = []
for i in range(0,193, 12):
    meaningless_taxels.append(i+7)
    meaningless_taxels.append(i+11)
    
useful_triangle = [4, 5, 6, 7, 8, 10]

# drop columns
drop_columns = range(193, 385)

dir_folder = 'data/'

# READ TACTILE
def read_tactile(obj_name):
    
    # read object data file
    df = pd.read_csv(dir_folder + obj_name + '/slide_raw/right_forearm_comp/data.log', sep = ' ', header=None).drop([0, 1], axis=1)
    df = df.rename(columns=new_names)
    df = df.drop(drop_columns, axis=1)
    #df = df.rename(columns={0:'t'})
    
    drop_taxels = []
    for taxel in df.columns:
        if (taxel in meaningless_taxels) or ( int(np.ceil(taxel/12)) not in useful_triangle ):
            drop_taxels.append(taxel)
    df = df.drop(drop_taxels, axis=1)
    return df

def filter_picks(ll):
    # choose picks
    picks = []
    for i in ll.index:
        add_it = True
        for j in picks:
            if i-j <= 40:
                add_it = False
                break
        if(add_it):
            picks.append(i)
    return picks

def read_encoder(_filename):
    # read encoder data
    _file_dir_encoder =dir_folder +  _filename + '/slide_raw/right_arm_encoders/data.log'
    _df_encoder = pd.read_csv(_file_dir_encoder, sep = ' ', header=None).drop([0], axis=1)
    _df_encoder = _df_encoder.rename(columns={1:'t', 5:'elbow'})
    _df_encoder = _df_encoder.loc[:, ['t', 'elbow']]
    _df_encoder.loc[ :, 'elbow'] = savgol_filter(_df_encoder.loc[:,'elbow'].values, 41, 3)

    # select the region by index
    _mask = (_df_encoder.elbow >= lower_lim) & (_df_encoder.elbow <= upper_lim)
    pre_peaks = pd.Series(0,index=_df_encoder.index)
    pre_peaks[_mask] = 1
    pre_peaks = pre_peaks.diff()
    pre_peaks = pre_peaks[pre_peaks != 0]
    pre_peaks = pre_peaks.drop(0)
    
    # filter peaks
    _peaks = filter_picks(pre_peaks)
    return _df_encoder, _peaks

def generate_slide_data(names, test_inds, fout):
    train_labels = {}
    test_labels = {}

    #test_inds = list(np.linspace(1,13, 13, dtype='int')*4)

    train_count= 0
    test_count = 0

    data_dir = 'slide_6_10/'
    train_ids = []
    test_ids = []
    
    print("generate slide data for: ")
    print(names)
    for obj_name in names:

        # read data
        df_tactile = read_tactile(obj_name)
        df_encoder, peaks = read_encoder(obj_name)

        # read each slide
        for i in range(62):
            df_chunk = df_tactile.loc[peaks[2*i] : peaks[2*i] + DEPTH_SIZE -1, :].values.reshape(6,DEPTH_SIZE,10)
            df_chunk = np.swapaxes(df_chunk, 1,2).astype('float64')

            name_convention = obj_name + '_' + str(i)
            torch.save(torch.from_numpy(df_chunk), data_dir + name_convention + '.pt')

            if i in test_inds:
                test_ids.append(name_convention)
                test_labels[name_convention] = names.index(obj_name)
                test_count += 1
            else:
                train_ids.append(name_convention)
                train_labels[name_convention] = names.index(obj_name)
                train_count += 1
    pickle.dump([train_ids, train_labels, test_ids, test_labels], open(fout,'wb'))

def generate_classwise_slide_data(names, test_inds, fout, num_class=23, C=1, W=6, H=10):
    num_sample_tot = 62
    num_sample_test = len(test_inds)
    num_sample_train = num_sample_tot - num_sample_test
    # images_train = np.zeros((num_class, num_sample_train, C, W, H), dtype=np.uint8)
    # labels_train = np.zeros((num_class, num_sample_train), dtype=int)
    # images_val = 0
    # labels_val = 0
    # images_test = np.zeros((num_class, num_sample_test, C, W, H), dtype=np.uint8)
    # labels_test = np.zeros((num_class, num_sample_test), dtype=int)


    train_labels = {}
    test_labels = {}

    train_count= 0
    test_count = 0

    data_dir = 'slide_6_10/'
    train_classes_ids = []
    test_classes_ids = []
    train_classes_labels = []
    test_classes_labels = []

    print("generate classwise slide data for: ")
    print(names)
    
    for obj_name in names:
        # initialize empty arrays to store ids per class
        train_ids = []
        test_ids = []
        train_labels = {}
        test_labels = {}
        # read data
        df_tactile = read_tactile(obj_name)
        df_encoder, peaks = read_encoder(obj_name)

        # read each slide
        for i in range(num_sample_tot):
            df_chunk = df_tactile.loc[peaks[2*i] : peaks[2*i] + DEPTH_SIZE -1, :].values.reshape(6,DEPTH_SIZE,10)
            df_chunk = np.swapaxes(df_chunk, 1,2).astype('float64')

            name_convention = obj_name + '_' + str(i)
            torch.save(torch.from_numpy(df_chunk), data_dir + name_convention + '.pt')

            if i in test_inds:
                test_ids.append(name_convention)
                test_labels[name_convention] = names.index(obj_name)
                test_count += 1
            else:
                train_ids.append(name_convention)
                train_labels[name_convention] = names.index(obj_name)
                train_count += 1
            
            
        # list of list
        test_classes_ids.append(test_ids) 
        train_classes_ids.append(train_ids)
        # list of dict
        test_classes_labels.append(test_labels)
        train_classes_labels.append(train_labels) 
            
    pickle.dump([train_classes_ids, train_classes_labels, test_classes_ids, test_classes_labels], open(fout,'wb'))

#fout = 'slide_6_10_c8.pkl'
#generate_slide_data(sub_obj_names, np.random.choice([i for i in range(62)], int(62 * 0.2)), fout) # put train and test together (N, C, W, H)

fout = 'slide_6_10_diff_init_classwise.pkl'
generate_classwise_slide_data(object_names, np.random.choice([i for i in range(62)], int(62 * 0.2)), fout) # separate by classes (num_class, N, C, W, H) 
