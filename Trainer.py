import torch
import torchaudio
import numpy as np
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Trainer, TrainingArguments, TrainerCallback, EarlyStoppingCallback
from datasets import load_dataset, Features, Value, concatenate_datasets, DatasetDict
import evaluate
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from transformers.trainer import logger

#-------------------------------------------------------------------------
# CUSTOMIZABLE PARAMETERS
#-------------------------------------------------------------------------
# Base pretrained model to use
MODEL_NAME = "facebook/wav2vec2-large-960h-lv60-self"

# File paths for datasets 
UASPEECH_TRAIN_PATH = "UASpeech/uaspeech_train.csv"
UASPEECH_TEST_PATH = "UASpeech/uaspeech_test.csv"
TORGO_PATH = "TORGO/metadata.csv"

# Output directory for saving the model
OUTPUT_DIR = "./wav2vec2-dysarthria-finetuned-fullrun"

# Training parameters
NUM_TRAIN_EPOCHS = 4
LEARNING_RATE = 3e-5
BATCH_SIZE = 2  # per device
GRADIENT_ACCUMULATION_STEPS = 2
WARMUP_STEPS = 500
SAVE_STEPS = 400
EVAL_STEPS = 400
LOGGING_STEPS = 25

# Model dropout settings - can improve generalization for dysarthric speech
ACTIVATION_DROPOUT = 0.1
ATTENTION_DROPOUT = 0.1
HIDDEN_DROPOUT = 0.1
FEAT_PROJ_DROPOUT = 0.0

# Audio preprocessing parameters
VAD_TOP_DB = 20  # Voice activity detection threshold (more lenient for dysarthric speech)
MIN_AUDIO_LENGTH = 32000  # Minimum samples (2 seconds at 16kHz)

#-------------------------------------------------------------------------
# VISUALIZATION AND MONITORING
#-------------------------------------------------------------------------
class TrainingMonitorCallback(TrainerCallback):
    """Tracks training metrics and generates plots at the end of training."""
    
    def __init__(self):
        self.train_steps = []
        self.train_losses = []
        self.eval_steps = []
        self.eval_wers = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        step = state.global_step
        if "loss" in logs:
            self.train_steps.append(step)
            self.train_losses.append(logs["loss"])
        if "eval_wer" in logs:
            self.eval_steps.append(step)
            self.eval_wers.append(logs["eval_wer"])

    def on_train_end(self, args, state, control, **kwargs):
        if self.train_steps:
            plt.figure(figsize=(10, 4))
            plt.plot(self.train_steps, self.train_losses, label="Train Loss")
            if self.eval_steps:
                plt.plot(self.eval_steps, self.eval_wers, label="Eval WER")
            plt.xlabel("Step")
            plt.title("Training Loss & Evaluation WER")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

#-------------------------------------------------------------------------
# DATA PROCESSING
#-------------------------------------------------------------------------
@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator for batching samples with variable length.
    Handles padding of audio input values and labels for CTC loss.
    """
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = "longest"
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Get input values and labels
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        # Pad input values
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Pad labels
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt"
        )

        # Replace padding with -100 for loss calculation
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # Add only essential information
        batch["labels"] = labels

        return batch


def process_dysarthric_speech(speech_array, sampling_rate):
    """
    Apply specific processing for dysarthric speech:
    - Normalization
    - Pre-emphasis to enhance speech clarity
    - Voice activity detection with lenient threshold
    - Padding to ensure minimum length
    """
    try:
        # Normalize audio
        speech_array = librosa.util.normalize(speech_array)

        # Apply pre-emphasis to enhance speech clarity
        speech_array = librosa.effects.preemphasis(speech_array)

        # Set minimum FFT size and hop length
        min_fft_size = 512  # Minimum FFT size
        n_fft = max(min_fft_size, min(2048, len(speech_array)))
        if n_fft % 2 != 0:  # Ensure n_fft is even
            n_fft -= 1

        # Ensure hop_length is valid (minimum 128)
        hop_length = max(128, n_fft // 4)

        # Voice activity detection with adjusted parameters
        if len(speech_array) > n_fft:  # Only apply VAD if signal is long enough
            intervals = librosa.effects.split(
                speech_array,
                top_db=VAD_TOP_DB,  # Customizable - more lenient threshold for dysarthric speech
                frame_length=n_fft,
                hop_length=hop_length
            )

            if len(intervals) > 0:
                speech_array = np.concatenate([speech_array[start:end] for start, end in intervals])

        # Ensure minimum length for wav2vec2 masking (at least 32000 samples = 2 seconds at 16kHz)
        if len(speech_array) < MIN_AUDIO_LENGTH:
            # Pad with silence if too short
            pad_length = MIN_AUDIO_LENGTH - len(speech_array)
            speech_array = np.pad(speech_array, (0, pad_length), mode='constant')

        return speech_array
    except Exception as e:
        print(f"Error in audio processing: {str(e)}")
        return speech_array  # Return original array if processing fails


def create_preprocessing_function(proc):
    """
    Creates a preprocessing function to transform raw audio files and transcriptions
    into the format needed by the model. Handles dysarthric speech specifically.
    """
    def _preprocessing_function(batch):
        try:
            # Clean transcription - remove newlines and extra whitespace
            transcription = batch["transcription"].replace('\n', ' ').strip()
            transcription = ' '.join(transcription.split())  # Remove extra whitespace

            # Load audio file
            audio_array, sampling_rate = torchaudio.load(batch["Wav_path"])

            # Convert to mono if necessary
            if audio_array.shape[0] > 1:
                audio_array = torch.mean(audio_array, dim=0)
            elif len(audio_array.shape) == 1:
                audio_array = audio_array.unsqueeze(0)

            # Resample to 16kHz if needed
            if sampling_rate != 16000:
                resampler = torchaudio.transforms.Resample(sampling_rate, 16000)
                audio_array = resampler(audio_array)
                sampling_rate = 16000

            # Convert to numpy for processing
            speech_array = audio_array.numpy().squeeze()

            # Apply preprocessing specific to dysarthric speech
            speech_array = process_dysarthric_speech(speech_array, sampling_rate)

            # Ensure speech_array is 1D
            if len(speech_array.shape) > 1:
                speech_array = speech_array.squeeze()

            # Check if the processed audio is too short
            if len(speech_array) < MIN_AUDIO_LENGTH:  # 2 seconds at 16kHz
                print(f"Warning: Audio file {batch['Wav_path']} is too short after processing")
                return None

            # Get input values
            inputs = proc(
                speech_array,
                sampling_rate=sampling_rate,
                padding=True,
                return_attention_mask=True,
                return_tensors=None  # Don't return tensors yet
            )

            # Handle input values - take first element if it's a list
            input_values = inputs.input_values
            if isinstance(input_values, list):
                input_values = input_values[0]

            # Process transcription
            label_features = proc(text=transcription.lower())
            labels = label_features.input_ids

            # Ensure labels is always a 1D array
            if isinstance(labels, (int, np.int32, np.int64)):
                labels = np.array([labels], dtype=np.int64)
            elif isinstance(labels, list):
                labels = np.array(labels, dtype=np.int64)
            elif isinstance(labels, np.ndarray) and labels.ndim == 0:
                labels = np.array([labels.item()], dtype=np.int64)

            # Convert input_values to numpy array if it isn't already
            if not isinstance(input_values, np.ndarray):
                input_values = np.array(input_values)

            # Return dictionary with proper keys
            return {
                "input_values": input_values,  # Should be 1D numpy array
                "labels": labels,  # Should be 1D numpy array
            }

        except Exception as e:
            print(f"Error processing file: {batch['Wav_path']} - {str(e)}")
            print(f"Transcription was: '{batch['transcription']}'")
            if 'audio_array' in locals():
                print(f"Audio shape: {audio_array.shape}")
                if 'speech_array' in locals():
                    print(f"Processed audio shape: {speech_array.shape}")
                if 'input_values' in locals():
                    print(f"Input values type: {type(input_values)}")
                    if hasattr(input_values, 'shape'):
                        print(f"Input values shape: {input_values.shape}")
                if 'labels' in locals():
                    print(f"Labels type: {type(labels)}")
                    if hasattr(labels, 'shape'):
                        print(f"Labels shape: {labels.shape}")
                    print(f"Labels value: {labels}")
            return None

    return _preprocessing_function


def compute_metrics(pred, processor):
    """
    Compute Word Error Rate (WER) to evaluate model performance.
    Lower WER is better.
    """
    wer_metric = evaluate.load("wer")

    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    # Decode predictions
    pred_str = processor.batch_decode(pred_ids)
    pred_str = [text.lower().strip() for text in pred_str]

    # Decode labels
    labels = pred.label_ids
    labels[labels == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels, group_tokens=False)
    label_str = [text.lower().strip() for text in label_str]

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {
        "wer": wer,
    }

#-------------------------------------------------------------------------
# CUSTOM TRAINERS
#-------------------------------------------------------------------------
class CTCTrainer(Trainer):
    """
    Custom trainer for CTC loss calculation.
    Implements detailed CTC loss computation appropriate for speech recognition.
    """
    def compute_loss(self, model, inputs, return_outputs=False):
        # Forward pass
        outputs = model(**inputs)

        # Get log probabilities
        logits = outputs.logits
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        # Get target lengths
        target_lengths = (inputs["labels"] != -100).sum(-1)

        # Get input lengths accounting for wav2vec2's downsampling
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            # Calculate lengths based on attention mask
            input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1))
        else:
            # If no attention mask, use sequence length
            input_lengths = self._get_feat_extract_output_lengths(
                torch.full((logits.shape[0],), logits.shape[1], device=logits.device)
            )

        # Get labels without padding
        labels = inputs["labels"].clone()
        labels[labels == -100] = 0  # Replace padding with valid index

        # Calculate CTC loss
        loss = torch.nn.functional.ctc_loss(
            log_probs.transpose(0, 1),  # (T, N, C)
            labels,
            input_lengths,
            target_lengths,
            blank=model.config.pad_token_id,
            reduction='mean',
            zero_infinity=True
        )

        return (loss, outputs) if return_outputs else loss

    def _get_feat_extract_output_lengths(self, input_lengths):
        """
        Computes the output length of the convolutional layers in wav2vec2
        """
        def _conv_out_length(input_length, kernel_size, stride):
            # Length after convolution layer
            return (input_length - kernel_size) // stride + 1

        # wav2vec2's convolutional feature encoder has multiple layers
        # we need to account for all of them
        for kernel_size, stride in [(10, 5), (3, 2), (3, 2), (3, 2), (3, 2), (2, 2), (2, 2)]:
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        return input_lengths


class CustomCTCTrainer(CTCTrainer):
    """
    Extends the CTCTrainer with custom checkpoint handling.
    This avoids issues with RNG state restoration which can cause problems when resuming training.
    """
    def _load_rng_state(self, checkpoint):
        if checkpoint is None:
            return
        
        # Skip RNG state restoration but keep all other checkpoint data
        print(f"\nContinuing training from checkpoint {checkpoint}, but using fresh random state")
        return


#-------------------------------------------------------------------------
# MAIN TRAINING FUNCTION
#-------------------------------------------------------------------------
def main():
    # Check CUDA availability
    print("✅ CUDA Available:", torch.cuda.is_available())
    print("🖥️  Using device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

    # Initialize model and processor
    processor = Wav2Vec2Processor.from_pretrained(
        MODEL_NAME,
        do_lower_case=True
    )
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)

    # Uncomment to load from checkpoint instead
    # model_path = "PATH_TO_CHECKPOINT_DIR"
    # processor = Wav2Vec2Processor.from_pretrained(model_path)
    # model = Wav2Vec2ForCTC.from_pretrained(model_path)

    # Apply dropout settings for dysarthric speech
    model.config.activation_dropout = ACTIVATION_DROPOUT
    model.config.attention_dropout = ATTENTION_DROPOUT
    model.config.hidden_dropout = HIDDEN_DROPOUT
    model.config.feat_proj_dropout = FEAT_PROJ_DROPOUT

    # Load UASpeech dataset
    UASpeech_dataset = load_dataset(
       "csv",
       data_files={
           "train": UASPEECH_TRAIN_PATH,
           "test": UASPEECH_TEST_PATH
       },
       features=Features({
            'Wav_path': Value('string'),
            'transcription': Value('string')
        }) 
    )

    # Load TORGO dataset
    TORGO_dataset = load_dataset(
        "csv",
        data_files=TORGO_PATH,
        features=Features({
            'Wav_path': Value('string'),
            'transcription': Value('string')
        })
    )

    # Combine datasets and create train/validation/test splits
    combined_train = concatenate_datasets([UASpeech_dataset["train"], TORGO_dataset["train"]])
    split_dataset = combined_train.train_test_split(test_size=0.1, seed=42)

    dataset = DatasetDict({
        "train": split_dataset["train"],
        "validation": split_dataset["test"],
        "test": UASpeech_dataset["test"].select(range(500))  # Use subset of test data for faster evaluation
    })

    # Print dataset statistics
    print(f"📊 TORGO train samples (original): {len(TORGO_dataset['train'])}")
    print(f"📊 UASpeech train samples (original): {len(UASpeech_dataset['train'])}")
    print(f"🔀 Combined training samples (before split): {len(combined_train)}")
    print(f"✅ Training samples (after split): {len(dataset['train'])}")
    print(f"📈 Validation samples: {len(dataset['validation'])}")
    print(f"🧪 Final test samples: {len(dataset['test'])}")

    # Create preprocessing function
    preprocessing_function = create_preprocessing_function(processor)

    # Preprocess dataset
    dataset = dataset.map(
        preprocessing_function,
        remove_columns=dataset["train"].column_names,
        num_proc=4  # Adjust based on CPU cores available
    )

    # Filter out None values from preprocessing
    dataset = dataset.filter(lambda x: x is not None)

    # Print dataset info after preprocessing
    print("\nDataset after preprocessing:")
    print("Train dataset size:", len(dataset["train"]))
    print("Test dataset size:", len(dataset["test"]))
    if len(dataset["train"]) > 0:
        sample = dataset["train"][0]
        print("\nSample processed item:")
        for key, value in sample.items():
            if isinstance(value, np.ndarray):
                print(f"{key} shape:", value.shape)
            elif isinstance(value, (list, str)):
                print(f"{key} length:", len(value))
            else:
                print(f"{key} value:", value)

    # Initialize data collator
    data_collator = DataCollatorCTCWithPadding(
        processor=processor,
        padding=True
    )

    # Create compute_metrics function with processor
    compute_metrics_with_processor = lambda pred: compute_metrics(pred, processor)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        group_by_length=True,  # Group similar length sequences together to speed up training
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        eval_strategy="steps",  # Run evaluations regularly during training
        num_train_epochs=NUM_TRAIN_EPOCHS,
        fp16=True,  # Use mixed precision training
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        save_steps=SAVE_STEPS,
        eval_steps=EVAL_STEPS,
        logging_steps=LOGGING_STEPS,
        save_total_limit=3,  # Keep 3 most recent checkpoints
        load_best_model_at_end=True,  # Load checkpoint with lowest WER after training
        metric_for_best_model="wer",
        greater_is_better=False,  # Lower WER is better
        max_grad_norm=3.0,  # Clip gradients for stability
        weight_decay=0.01,  # L2 regularization
        remove_unused_columns=False,
        dataloader_num_workers=4,  # Adjust based on CPU cores
        lr_scheduler_type="cosine"  # Cosine learning rate schedule
    )

    # Initialize trainer with custom CTC trainer
    trainer = CustomCTCTrainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics_with_processor,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=processor.feature_extractor,
        callbacks=[
            TrainingMonitorCallback(),
            EarlyStoppingCallback(early_stopping_patience=5)  # Stop if no improvement for 5 evaluations
        ]
    )

    # Train model
    # Uncomment to resume from checkpoint:
    # checkpoint_path = "PATH_TO_CHECKPOINT"
    # trainer.train(resume_from_checkpoint=checkpoint_path)
    
    trainer.train()

    # Save final model
    trainer.save_model("./final-dysarthria-model")
    processor.save_pretrained("./final-dysarthria-model")


if __name__ == "__main__":
    # Set environment variables to reduce verbosity
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    main()