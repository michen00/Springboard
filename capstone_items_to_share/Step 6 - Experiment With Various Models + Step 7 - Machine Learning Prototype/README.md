# Outline

1. 1.0-mic-divide_data_by_duration
> Samples are divided by duration into short, medium, and long sets.

1. 2.0-mic-compare_rescale_pre_and_post_spectrogram_with_initial_pad
> I compared the effects of rescaling (stretching or shrinking) before and after spectrogram extraction. The results were moot since the rescaling operations introduced a lot of distortion. Padding but not stretching/shrinking will be considered henceforth.

1. 3.0-mic-pad_data_short
> The samples of short data are prepadded for downstream analyses.

1. 4.0-mic-explore-initial-models_medium
> I tried two time series classification techniques (RISE and MINIROCKET) on the raw audio of a subsample of the medium-length data.

1. 5.0-mic-extract_spectrograms_and_MFCCs_short
> Mel-frequency spectrograms and cepstral coefficients are extracted from the short data for downstream analysis.

1. 6.0-mic-explore_models_with_spectrogram_short
> I tried a CNN and MINIROCKET on the spectrograms.

1. 7.0-mic-explore_models_with_MFCCs_short
> I tried a CNN and MINIROCKET on the mel-frequence cepstral coefficients.

1. 8.0-mic-explore_models_with_FRILL_short
> The FRILL embeddings are introduced and piloted with a number of classifiers. This feature set confers several advantages that are discussed, establishing the FRILL embeddings as a focal point of subsequent development. The test harness is compared on ragged and prepadded samples of short duration, but results are inconclusive due to leakage in the padding operation.

1. 9.0-mic-compare_torchaudio_vs_librosa_load
> Since audio loading time was a drag on development, a mini-experiment is conducted to compare the loading times of librosa and torchaudio. Torchaudio wins by a small margin.

1. 10.0-mic-compare_ragged_and_padded_with_FRILL_short_plus_long
> The short and long samples are combined (without the medium samples) to assess the effects of padding. The test harness is revised for stratified grouped k-fold cross validation with in-fold padding. The pool of candidate models is reduced to omit the worst-performing models from 8.0 above. Metrics more suited to imbalanced classification (F1 and balanced accuracy) are introduced. Several models show promising performance without tuning.

# Get the data

[Unified Multilingual Dataset of Emotional Human Utterances](#https://github.com/michen00/unified_multilingual_dataset_of_emotional_human_utterances)

# Third-party libraries used

I included an environment.yml file, but only after a recent transition from full anaconda to miniconda resetting my environments in the process, so I can't guarantee everything is aligned.

* findspark
* ipython
* ipython-autotime
* keras
* librosa
* lightgbm
* matplotlib
* nb_black
* numpy
* opencv
* pandas
* pyspark
* seaborn
* skimage
* sklearn
* sktime
* swifter
* tensorflow
* tensorflow_hub
* torchaudio
* tqdm
* tsai
* xgboost
