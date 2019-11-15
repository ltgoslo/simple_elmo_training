# simple_elmo_training
Minimal code to train ELMo models in TensorFlow.

Heavily based on https://github.com/allenai/bilm-tf .

Most changes are simplifications and updating the code to the recent versions of TensorFlow 1.
See also our repository with [simple code to infer contextualized word vectors from pre-trained ELMo models](https://github.com/ltgoslo/simple_elmo).

# Training

`python3 bilm/train_elmo.py --train_prefix $DATA --size $SIZE --vocab_file $VOCAB --save_dir $OUT`

where

`$DATA` is a path to the directory containing 2 or more of (possibly gzipped) plain text files: your training corpus.

`$SIZE` if the number of word tokens in $DATA (necessary to properly construct and log batches).

`$VOCAB` is a (possibly gzipped) one-word-per-line vocabulary file to be used for language modeling; it should always contain at least <S>, </S> and <UNK>.

`$OUT` is a directory where the TensorFlow checkpoints will be saved.


Before training, please review the settings in `bilm/train_elmo.py`. The most important are:
- batch_size (default 128)
- n_gpus (default 2; if no GPU, all available CPU cores are used)
- LSTM dimensionality (default 2048; the original paper used 4096)
- n_epochs (default 3; optimal value depends on the size of your corpus)
- n_negative_samples_batch (default 4096; the original paper used 8192)

# Converting to HDF5

After the training, use the `bilm/dump_weights.py` script to convert the checkpoints to and HDF5 model.
Save your vocabulary in the same directory. Change the `n_characters` in the `options.json` file to 262.

More details at https://github.com/allenai/bilm-tf

