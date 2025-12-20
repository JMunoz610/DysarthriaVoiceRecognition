# Dysarthric Speech Recognition System (Jan 2025 - May 2025)

## Motivation
Individuals with motor speech disorders often struggle with traditional speech recognition systems, creating barriers to independence and mobility. This project addresses that challenge by fine‑tuning a **Wav2Vec2 model** on dysarthric speech datasets (UASpeech, TORGO) to improve recognition accuracy. By combining optimized training with a real‑time, wake‑word activated pipeline, the system enables reliable **voice‑controlled wheelchair navigation** — all offline and with minimal latency.

---

## Project Contents
- **Trainer.py** → fine‑tunes Wav2Vec2 on dysarthric speech data
- **ASR.py** → real‑time speech recognition system for live voice control
- **environment.yml** → Conda environment file for dependencies

---

## FOLDER STRUCTURE

📁 UASpeech/
    ├── uaspeech_train.csv
    ├── uaspeech_test.csv
    └── wav files (referenced in the CSVs)

📁 TORGO/
    ├── metadata.csv
    └── wav files (referenced in the CSV)

📁 final-dysarthria-model/
    ├── pytorch_model.bin
    ├── config.json
    ├── vocab.json
    ├── tokenizer_config.json
    └── other Hugging Face model files

📄 Trainer.py        → training script
📄 ASR.py            → real-time speech recognition system
📄 environment.yml   → dependencies for Conda environment

To get model parameters download from this google drive: [Model Files](https://drive.google.com/drive/folders/1V5cxaTxbSmFlER3M-AAEdEKh2gls5-Hp)

---

## SETUP INSTRUCTIONS

1. INSTALL ANACONDA
   → https://www.anaconda.com/products/distribution

2. CREATE THE ENVIRONMENT

   Open a terminal in the project folder and run:

   conda env create -f environment.yml
   conda activate dysarthria-asr

3. CHECK YOUR FILE PATHS

- Edit the following paths inside Trainer.py to reflect your folder layout:

  UASPEECH_TRAIN_PATH = "UASpeech/uaspeech_train.csv"
  UASPEECH_TEST_PATH  = "UASpeech/uaspeech_test.csv"
  TORGO_PATH          = "TORGO/metadata.csv"

- Make sure the paths in the CSV files (under the 'Wav_path' column) are either:
  → Relative to the project folder, or
  → Full absolute paths pointing to valid .wav files

---

## TRAINING THE MODEL

To fine-tune the model:

   python Trainer.py

This will:
- Load and preprocess both UASpeech and TORGO datasets
- Train the Wav2Vec2 model
- Save it to: /final-dysarthria-model/

Training parameters like learning rate, batch size, and epochs can be changed at the top of Trainer.py.

---

## RUNNING THE ASR SYSTEM

To start the real-time voice command system:

   python ASR.py

It will:
- Continuously listen using your microphone
- Wait for the wake word ("kip" by default)
- Transcribe speech using your fine-tuned model
- Recognize commands using phonetic similarity
- Reply with text-to-speech confirmation

You can customize:
- The wake word
- Command list and thresholds
- Chunk size, overlap ratio, and cooldown settings

These settings are defined in the ASRConfig class in ASR.py.

**Note:** The ASR system supports stereo microphone input.
You must specify how many microphones ie: channels=1 or channels=2

---

### DEPENDENCIES

Installed via environment.yml. Includes:
- torch
- torchaudio
- transformers
- librosa
- pyaudio
- jellyfish
- pyttsx3

---

### NOTES

- All training and inference audio must be 16 kHz .wav files
- Inference supports both mono and stereo input (converted to mono automatically)
- No internet is required after installation
- Use a USB or headset mic for best results

---------------------------------------------------
