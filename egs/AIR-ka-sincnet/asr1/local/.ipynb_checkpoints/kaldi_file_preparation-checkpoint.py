import pandas as pd
#from pydub import AudioSegment
import os
from random import shuffle
from sklearn.model_selection import train_test_split
import sys

AIR_dir = sys.argv[1]
Recipe_dir = sys.argv[2]
data_files = sys.argv[3]

# Indiividual speaker list
train_spk_list = []
test_spk_list = []
bulk_spk_list = []

# Experiment type either 'PER' or 'WER'
experiment_type = 'PER'

text_files_dir = AIR_dir+'/Transcript/'
audio_files_dir = AIR_dir+'/Audio/'

split_cat = ['train','test','dev']


# # Text
# ## Before running from here, make sure dev.uttids is copied from dataset folder to 'Bulk Data Kaldi Files' folder
for x in split_cat:
    ref_file = open(Recipe_dir+'/'+data_files+'/'+x+'.uttids','r')
    ref_file_content = ref_file.read().split('\n')
    ref_file_content.remove('')

    temp_list = []
    temp_list_phoneme = []
    for item in ref_file_content:
        if x == 'train':
            train_spk_list.append(item[:2])
        elif x == 'test':
            test_spk_list.append(item[:2])
        else:
            bulk_spk_list.append(item[:2])

        prefix = item[-1]
        file_name = item[:6]
        folder_name = item[7:-2]

        if prefix == 'C':
            type_of_data = 'clean'
        elif prefix == 'N':
            type_of_data = 'noisy'
        else:
            type_of_data = 'bulk'

        split_file = open(text_files_dir+type_of_data+'/'+folder_name+'/'+file_name+'.txt','r')
        s_f_content = split_file.read().split('\n')
        s_f_content = list(filter(lambda a: a != '', s_f_content))

        temp_str = item+' '
        temp_str_phoneme = item+';'
        for line in s_f_content:
            line = line.replace("32 ","")
            line = line.replace(" 32 ","")
            line = line.replace("32","")

            line = line.replace("34 ","")
            line = line.replace(" 34","")
            line = line.replace("34","")

            line = line.replace("IE","I E")
            line = line.replace("IPA","I PA")
            line = line.replace("VAl","VA")
            line = line.replace("LAll","LAl")
            if line != s_f_content[-1]:
                temp_str = temp_str + line + ' '
                temp_str_phoneme = temp_str_phoneme + line + ' '
            else:
                temp_str = temp_str + line
                temp_str_phoneme = temp_str_phoneme + line

        temp_str = ' '.join(temp_str.split())
        temp_str_phoneme = ' '.join(temp_str_phoneme.split())

        temp_list.append(temp_str)
        temp_list_phoneme.append(temp_str_phoneme)

        split_file.close()
    temp_list.sort()
    temp_list_phoneme.sort()


    file = open(Recipe_dir+'/'+data_files+'/'+x+'.text','w')
    for item in temp_list:
        file.write(item+'\n')
    file.close()
#     file = open(Recipe_dir+'/'+data_files+'/'+x+'_phoneme.text','w')
#     for item in temp_list_phoneme:
#         file.write(item+'\n')
#     file.close()

    ref_file.close()

train_spk_list = list(set(train_spk_list))
train_spk_list.sort()

test_spk_list = list(set(test_spk_list))
test_spk_list.sort()

bulk_spk_list = list(set(bulk_spk_list))
bulk_spk_list.sort()

# # utt2spk
for x in split_cat:
    ref_file = open(Recipe_dir+'/'+data_files+'/'+x+'.uttids','r')
    ref_file_content = ref_file.read().split('\n')
    ref_file_content.remove('')

    temp_list = []
    for item in ref_file_content:
        spk = item[:2]
        temp_str = item+' '+spk
        temp_list.append(temp_str)
    temp_list.sort()

    file = open(Recipe_dir+'/'+data_files+'/'+x+'.utt2spk','w')
    for item in temp_list:
        file.write(item+'\n')
    file.close()
    ref_file.close()


# # spk2utt
for x in split_cat:
    if x == 'dev':
        spk_list = bulk_spk_list
    elif x == 'train':
        spk_list = train_spk_list
    else:
        spk_list = test_spk_list

    ref_file = open(Recipe_dir+'/'+data_files+'/'+x+'.utt2spk','r')
    ref_file_content = ref_file.read().split('\n')
    ref_file_content.remove('')

    spk2utt_dict = {}
    for spk in spk_list:
        spk2utt_dict[spk] = []

    for item in ref_file_content:
        spk = item.split(' ')[1]
        spk2utt_dict[spk].append(item.split(' ')[0])

    file = open(Recipe_dir+'/'+data_files+'/'+x+'.spk2utt','w')
    for spk in spk_list:
        utt_list = spk2utt_dict[spk]

        temp_str = spk
        for utt in utt_list:
            temp_str = temp_str+' '+utt
        temp_str = temp_str+'\n'

        file.write(temp_str)
    file.close()
    ref_file.close()


# # spk2gender
for x in split_cat:
    if x == 'dev':
        spk_list = bulk_spk_list
    elif x == 'train':
        spk_list = train_spk_list
    else:
        spk_list = test_spk_list

    temp_list = []
    for spk in spk_list:
        gender = spk[0].lower()
        temp_str = spk+' '+gender+'\n'
        temp_list.append(temp_str)
    temp_list.sort()

    file = open(Recipe_dir+'/'+data_files+'/'+x+'.spk2gender','w')
    for item in temp_list:
        file.write(item)
    file.close()


# # wav.scp
for x in split_cat:
    ref_file = open(Recipe_dir+'/'+data_files+'/'+x+'.uttids','r')
    ref_file_content = ref_file.read().split('\n')
    ref_file_content.remove('')

    temp_list = []
    for item in ref_file_content:
        prefix = item[-1]
        file_name = item[:6]
        folder_name = item[7:-2]

        if prefix == 'C' or prefix == 'N':
            type_of_data = 'seed'
        else:
            type_of_data = 'bulk'

        temp_str = item+' '+audio_files_dir+type_of_data+'/'+folder_name+'/'+file_name+'.wav\n'

        temp_list.append(temp_str)
    temp_list.sort()

    file = open(Recipe_dir+'/'+data_files+'/'+x+'_wav.scp','w')
    for item in temp_list:
        file.write(item)
    file.close()
    ref_file.close()

