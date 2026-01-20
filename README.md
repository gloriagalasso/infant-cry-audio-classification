# Infant Cry Classification Using Audio Features and Convolutional Models

This repository implements experiments for automatic classification of infant cry types using audio feature extraction and convolutional neural network models. The work focuses on discriminating five clinically relevant cry categories (belly pain, burping, discomfort, hungry, tired) using the Donate-a-Cry corpus.

**Summary**

- **Objective:** Develop and evaluate machine learning pipelines that classify infant cry audio into need-based categories using MFCC features and convolutional sequence models.
- **Approach:** Extract MFCC features with `librosa`, normalize with `scikit-learn`, and train 1D convolutional neural networks implemented with `tensorflow.keras`.

**Dataset**

- **Source:** Donate-a-Cry corpus, located at `data/donateacry_corpus` in this repository.
- **Labels:** `belly_pain`, `burping`, `discomfort`, `hungry`, `tired`.
- **Layout:** Audio files are organized into class-specific subfolders under the dataset directory.

**Methodology**

- **Preprocessing:** Audio is loaded with a fixed duration (5 seconds in example notebooks) and converted to MFCCs (40 coefficients). MFCC sequences are padded or trimmed to a fixed length (examples use 862 time steps) for model input.
- **Normalization:** Features are flattened and standardized with `StandardScaler`, then reshaped to `(time_steps, n_mfcc)` for model consumption.
- **Modeling:** Example models use stacked `Conv1D` layers with Batch Normalization, pooling or GlobalAveragePooling, dropout, and a final dense softmax layer for five-class classification.
- **Training:** Models are compiled with the Adam optimizer and categorical cross-entropy loss. Notebooks show training with a validation split and print test accuracy after evaluation.

**Audio-to-Spectrogram Processing**

- **Motivation:** Time-domain waveforms can be transformed into time–frequency representations that make spectral and temporal patterns explicit for convolutional and image-based models.
- **Procedure:** Compute short-time Fourier transform (STFT) or mel-scaled spectrograms from raw audio (parameters: `n_fft`, `hop_length`, window length). Convert magnitude to a log-scale (e.g., dB) or use log-mel spectrograms to compress dynamic range and emphasize perceptually relevant bands.
- **Shapes & sizing:** Spectrograms are resized, padded or cropped to a fixed time-frequency size so they can be consumed by 2D convolutional architectures or treated as channels for 1D/2D models.
- **Variants:** Use linear-magnitude spectrograms, log-mel, or Mel-frequency cepstral representations depending on model choice. Time–frequency augmentation (see below) is often applied on spectrograms.

**Data Augmentation**

- **Purpose:** Augmentation increases effective dataset size, reduces overfitting, and exposes models to realistic acoustic variability.
- **Common waveform-level augmentations:** additive background noise, random time-shifting (temporal translations), pitch shifting, time-stretching, random gain/amplitude scaling, and room impulse response (reverberation) simulation.
- **Spectrogram-level augmentations:** frequency/time masking (SpecAugment), random cropping of time frames, and frequency warping. These augmentations are label-preserving and effective for time–frequency models.
- **Mixing strategies:** `mixup` or audio-level mixing (overlaying two audio signals) can regularize training but require careful label handling (soft labels or proportional labels).
- **Implementation choices:** Apply augmentations offline to expand the stored dataset or apply them on-the-fly during training (recommended for variety and storage efficiency). Ensure augmentation parameters maintain label validity (e.g., small pitch shifts that do not change the semantic label).
- **Evaluation consideration:** When using augmentation, keep a clean validation/test set without augmentation to measure generalization to authentic recordings.

**Repository Contents**

- Notebooks:
	- `augmented_audio_approach.ipynb` — end-to-end feature extraction, Conv1D model, training, evaluation, and prediction inspection.
	- `classifying-infant-cry-type.ipynb` — alternative experiments and analyses.
	- `audio_approach.ipynb` — baseline audio preprocessing experiments.
- Scripts:
	- `20audioeach.py` — auxiliary audio processing and dataset utilities.
- Data:
	- `data/donateacry_corpus` — raw audio files arranged by label.
- Project metadata:
	- `LICENSE` — licensing terms for the repository.

**Usage**

1. Create a Python virtual environment and install dependencies. Recommended packages include `numpy`, `librosa`, `scikit-learn`, and `tensorflow`.

	 ```bash
	 python -m venv .venv
	 source .venv/bin/activate
	 pip install numpy librosa scikit-learn tensorflow
	 ```

2. Open the desired notebook in JupyterLab or Jupyter Notebook and run cells sequentially:

	 ```bash
	 jupyter lab
	 ```

3. Typical workflow: extract features, normalize, train Conv1D model, and evaluate on a held-out test set (see `augmented_audio_approach.ipynb`).

**Reproducibility**

- Notebooks set `random_state` in `train_test_split` for reproducible splits. To reproduce experiments, run notebooks end-to-end in the same environment and record package versions. Saving trained model weights and training histories is recommended.

**Limitations and Ethical Considerations**

- Generalizability is limited by dataset size and recording conditions. Models trained here are experimental and not intended for clinical decision-making.
- Labels reflect dataset annotations and may include subjective judgments. Ensure informed consent and compliance with privacy regulations before sharing or deploying models.

**Future Work**

- Increase dataset diversity and perform cross-environment validation.
- Evaluate alternative features (spectrogram, log-mel), advanced model architectures (transformers, attention), and data augmentation strategies.
- Add interpretability analyses (saliency maps) and infant-level cross-validation to reduce subject bias.

**Citation**

When using these experiments in publications, cite the Donate-a-Cry corpus and the audio-processing and deep-learning methods employed (e.g., MFCC literature, convolutional models for sequential data).
