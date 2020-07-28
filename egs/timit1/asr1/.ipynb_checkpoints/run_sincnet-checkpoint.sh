#!/bin/bash

# Copyright 2019 IIIT-Bangalore (Shreekantha Nadig)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch
stage=-1       # start from -1 if you need to start from data download
stop_stage=100
ngpu=1         # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
dumpdir=dump_raw_25ms   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=1      # verbose option
resume=        # Resume the training from snapshot

# feature configuration
do_delta=false

train_config=conf/train_sincLSTM_25ms_base.yaml
decode_config=conf/decode.yaml

# decoding parameter
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'

# data
#timit=/local_disk/galactica/laboinfo/parcollet/CORPUS/TIMIT/TIMIT/
timit=/mnt/disk/Dhanya/Datasets/TIMIT_E2ESincNet
trans_type=phn

# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_raw_nodev_25ms
train_dev=train_raw_dev_25ms
recog_set="train_raw_dev_25ms test_raw_25ms"

feat_tr_dir=${dumpdir}/${train_set}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}; mkdir -p ${feat_dt_dir}

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### copy the actual raw features
    echo "stage 1: Feature processing"
    bash local/timit_data_prep.sh ${timit} ${trans_type} || exit 1
    bash local/timit_format_data.sh || exit 1

    # compute global CMVN for normalization
    compute-cmvn-stats scp:data/${train_set}/feats_raw.scp data/${train_set}/cmvn.ark

    # dump features
    bash dump.sh --cmd "$train_cmd" --nj 8 --do_delta ${do_delta} \
    data/${train_set}/feats_raw.scp data/${train_set}/cmvn.ark exp/${dumpdir}/train ${feat_tr_dir}

    bash dump.sh --cmd "$train_cmd" --nj 8 --do_delta ${do_delta} \
    data/${train_dev}/feats_raw.scp data/${train_set}/cmvn.ark exp/${dumpdir}/dev ${feat_dt_dir}

    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}; mkdir -p ${feat_recog_dir}
        bash dump.sh --cmd "$train_cmd" --nj 8 --do_delta ${do_delta} \
        data/${rtask}/feats_raw.scp data/${train_set}/cmvn.ark exp/${dumpdir}/recog/${rtask} \
        ${feat_recog_dir}
    done
fi


dict=data/lang_1char/train_nodev_units.txt
echo "dictionary: ${dict}"
recog_set="train_raw_dev_25ms test_raw_25ms"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"

    mkdir -p data/lang_1char/
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    text2token.py -s 1 -n 1 data/train/text --trans_type ${trans_type} | cut -f 2- -d" " | tr " " "\n" \
    | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}

    # make json labels
    data2json.sh --feat ${feat_tr_dir}/feats.scp --trans_type ${trans_type} \
    data/train ${dict} > ${feat_tr_dir}/data.json
    data2json.sh --feat ${feat_dt_dir}/feats.scp --trans_type ${trans_type} \
    data/dev ${dict} > ${feat_dt_dir}/data.json


    feat_recog_dir=${dumpdir}/train_raw_dev_25ms
    data2json.sh --feat ${feat_recog_dir}/feats.scp --trans_type ${trans_type} \
    data/dev ${dict} > ${feat_recog_dir}/data.json

    feat_recog_dir=${dumpdir}/test_raw_25ms
    data2json.sh --feat ${feat_recog_dir}/feats.scp --trans_type ${trans_type} \
    data/test ${dict} > ${feat_recog_dir}/data.json
fi

if [ -z ${tag} ]; then
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
else
    expname=${train_set}_${backend}_${tag}
fi


expdir=exp/${expname}
mkdir -p ${expdir}
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: Network Training"
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
    asr_train.py \
    --config ${train_config} \
    --ngpu ${ngpu} \
    --backend ${backend} \
    --outdir ${expdir}/results \
    --tensorboard-dir tensorboard/${expname} \
    --debugmode ${debugmode} \
    --dict ${dict} \
    --debugdir ${expdir} \
    --minibatches ${N} \
    --verbose ${verbose} \
    --resume ${resume} \
    --train-json ${feat_tr_dir}/data.json \
    --valid-json ${feat_dt_dir}/data.json
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Decoding"
    nj=8
    for rtask in ${recog_set}; do
        (
            decode_dir=decode_${rtask}_$(basename ${decode_config%.*})
            feat_recog_dir=${dumpdir}/${rtask}

            # split data
            splitjson.py --parts ${nj} ${feat_recog_dir}/data.json

            #### use CPU for decoding
            ngpu=0

            ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --debugmode ${debugmode} \
            --verbose ${verbose} \
            --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model} \
			--batchsize 0
            score_sclite.sh ${expdir}/${decode_dir} ${dict}

        ) &
    done
    wait
    echo "Finished"
fi
