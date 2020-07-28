iimport os

#
# Please, modify the PATHs accordingly to your setup!
#

train_base = "/home/dhanya/espnet/egs/timit1/asr1/data/train_raw_nodev_25ms"
dev_base   = "/home/dhanya/espnet/egs/timit1/asr1/data/train_raw_dev_25ms"
test_base  = "/home/dhanya/espnet/egs/timit1/asr1/data/test_raw_25ms"

print("Creating data dirs...")
if not os.path.exists(train_base):
    os.makedirs(train_base)

if not os.path.exists(dev_base):
    os.makedirs(dev_base)

if not os.path.exists(test_base):
    os.makedirs(test_base)

train_lst     = open("/home/dhanya/kaldi/egs/timit1/s5/data/train/wav.scp","r").readlines()
dev_lst       = open("/home/dhanya/kaldi/egs/timit1/s5/data/dev/wav.scp","r").readlines()
test_lst      = open("/home/dhanya/kaldi/egs/timit1/s5/data/test/wav.scp","r").readlines()

out_train_lst = open(train_base+"/wav.lst", "w")
out_dev_lst   = open(dev_base+"/wav.lst", "w")
out_test_lst  = open(test_base+"/wav.lst", "w")

print("Converting audio files...")
for line in train_lst:
    file_path = line.split(" ")[4]
    os.system("sox "+file_path+" "+file_path.split(".")[0]+".wav")
    out_train_lst.write(line.split(" ")[0]+" "+file_path.split(".")[0]+".wav\n")

for line in dev_lst:
    file_path = line.split(" ")[4]
    os.system("sox "+file_path+" "+file_path.split(".")[0]+".wav")
    out_dev_lst.write(line.split(" ")[0]+" "+file_path.split(".")[0]+".wav\n")

for line in test_lst:
    file_path = line.split(" ")[4]
    os.system("sox "+file_path+" "+file_path.split(".")[0]+".wav")
    out_test_lst.write(line.split(" ")[0]+" "+file_path.split(".")[0]+".wav\n")

print("Done.")
