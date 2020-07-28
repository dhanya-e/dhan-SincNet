import os

#
# Please, modify the PATHs accordingly to your setup!
#

train_base = "/home/dhanya/espnet/egs/AIR_kannada/asr1/data/train_raw_nodev_25ms"
dev_base   = "/home/dhanya/espnet/egs/AIR_kannada/asr1/data/train_raw_dev_25ms"
test_base  = "/home/dhanya/espnet/egs/AIR_kannada/asr1/data/test_raw_25ms"

print("Creating data dirs...")
if not os.path.exists(train_base):
    os.makedirs(train_base)

if not os.path.exists(dev_base):
    os.makedirs(dev_base)

if not os.path.exists(test_base):
    os.makedirs(test_base)

train_lst     = open("/mnt/disk/Dhanya/AIR_kannada/AIR_kannada_uttids_new/train_wav.scp","r").readlines()
dev_lst       = open("/mnt/disk/Dhanya/AIR_kannada/AIR_kannada_uttids_new/dev_wav.scp","r").readlines()
test_lst      = open("/mnt/disk/Dhanya/AIR_kannada/AIR_kannada_uttids_new/test_wav.scp","r").readlines()

out_train_lst = open(train_base+"/wav.lst", "w")
out_dev_lst   = open(dev_base+"/wav.lst", "w")
out_test_lst  = open(test_base+"/wav.lst", "w")

audio_dir = "/mnt/disk/Dhanya/AIR_kannada/AIR_Data_Final/Audio/"
print("Converting audio files...")
for line in train_lst:
    file_path = line.split(" ")[1]
    file_path_split = file_path.split("/")[-3:]
    file_path_join = '/'.join(file_path_split) 
    #os.system("sox "+file_path+" "+file_path.split(".")[0]+".wav")
    out_train_lst.write(line.split(" ")[0]+ " " + audio_dir + file_path_join)

for line in dev_lst:
    file_path = line.split(" ")[1]
    file_path_split = file_path.split("/")[-3:]
    file_path_join = '/'.join(file_path_split)
    #os.system("sox "+file_path+" "+file_path.split(".")[0]+".wav")
    out_dev_lst.write(line.split(" ")[0]+" "+ audio_dir + file_path_join)

for line in test_lst:
    file_path = line.split(" ")[1]
    file_path_split = file_path.split("/")[-3:]
    file_path_join = '/'.join(file_path_split)
    #os.system("sox "+file_path+" "+file_path.split(".")[0]+".wav")
    out_test_lst.write(line.split(" ")[0]+" "+ audio_dir + file_path_join)

print("Done.")

