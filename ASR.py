import time
import torch
import pyaudio
import numpy as np
import pyttsx3
import jellyfish
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import os


@dataclass
class ASRConfig:
    """Configuration for the Automatic Speech Recognition system.
    
    This class holds all customizable parameters for the speech recognition system.
    Modify these values to adjust the behavior of the system.
    """
    # Model paths - CUSTOMIZE THESE FOR YOUR SETUP
    base_model_path: str = "final-dysarthria-model"  # Pre-trained model path
    checkpoint_path: str = "final-dysarthria-model"  # Fine-tuned model path
    
    # Audio configuration - CUSTOMIZE AS NEEDED
    chunk_size: int = 1024  # Size of audio chunks to process
    audio_format: int = pyaudio.paInt16  # Audio format (16-bit)
    channels: int = 1  # Mono audio
    rate: int = 16000  # Sample rate in Hz
    overlap_ratio: float = 0.5  # Overlap between audio chunks (0.0-1.0)
    
    # Command recognition settings - CUSTOMIZE THESE FOR YOUR NEEDS
    default_similarity_threshold: float = 0.5  # Default threshold for command matching
    wake_word: str = "kip"  # Wake word to activate command mode
    command_timeout: int = 15  # Seconds to listen for commands after wake word
    command_cooldown: float = 1.0  # Seconds between allowed command detections
    fingerprint_history_size: int = 5  # Number of recent audio fingerprints to store
    
    # Voice settings - CUSTOMIZE FOR YOUR PREFERENCE
    voice_id: int = 1  # Voice ID for text-to-speech (0: usually male, 1: usually female)
    
    # Command-specific thresholds - CUSTOMIZE FOR CRITICAL COMMANDS
    command_thresholds: Dict[str, float] = None
    
    def __post_init__(self):
        # Initialize command thresholds with defaults if not provided
        if self.command_thresholds is None:
            self.command_thresholds = {
                "stop": 0.6,  # Higher threshold for important commands
                "halt": 0.6,
                "cease": 0.6,
                self.wake_word: 0.65  # Wake word usually needs higher confidence
            }


class PhoneticCommandProcessor:
    """Handles command recognition using phonetic matching algorithms."""
    
    def __init__(self, predefined_commands: List[str], config: ASRConfig):
        self.predefined_commands = predefined_commands
        self.config = config
        # Commands that bypass the wake word requirement (safety-critical)
        self.critical_commands = ["stop", "halt", "cease"]
        
        # Precompute phonetic representations of commands for faster matching
        self.phonetic_representations = {}
        for command in predefined_commands:
            self.phonetic_representations[command] = {
                'soundex': jellyfish.soundex(command),
                'metaphone': jellyfish.metaphone(command),
                'nysiis': jellyfish.nysiis(command)
            }
    
    def compute_similarity(self, recognized_text: str, command: str) -> float:
        """Compute weighted similarity between recognized text and a command."""
        if not recognized_text:
            return 0.0
        
        # Normalize text
        recognized_text = recognized_text.lower().strip()
        
        # Get phonetic representations of the recognized text
        recognized_soundex = jellyfish.soundex(recognized_text)
        recognized_metaphone = jellyfish.metaphone(recognized_text)
        recognized_nysiis = jellyfish.nysiis(recognized_text)
        
        # Get command phonetic representations
        command_soundex = self.phonetic_representations[command]['soundex']
        command_metaphone = self.phonetic_representations[command]['metaphone']
        command_nysiis = self.phonetic_representations[command]['nysiis']
        
        # Calculate similarity metrics
        levenshtein_distance = jellyfish.levenshtein_distance(recognized_text, command)
        max_length = max(len(recognized_text), len(command))
        normalized_levenshtein = 1 - (levenshtein_distance / max_length if max_length > 0 else 0)
        
        jaro_similarity = jellyfish.jaro_similarity(recognized_text, command)
        jaro_winkler_similarity = jellyfish.jaro_winkler_similarity(recognized_text, command)
        
        soundex_match = 1.0 if recognized_soundex == command_soundex else 0.0
        metaphone_match = 1.0 if recognized_metaphone == command_metaphone else 0.0
        nysiis_match = 1.0 if recognized_nysiis == command_nysiis else 0.0
        
        # Apply weighted scoring - adjust weights if needed
        similarity_score = (
            0.10 * normalized_levenshtein +
            0.10 * jaro_similarity +
            0.10 * jaro_winkler_similarity +
            0.30 * soundex_match +
            0.20 * metaphone_match +
            0.20 * nysiis_match
        )
        
        return similarity_score
    
    def find_most_similar_command(self, recognized_text: str) -> Tuple[Optional[str], Optional[float]]:
        """Find the most similar command using phonetic matching."""
        if not recognized_text:
            return None, None
            
        best_command = None
        highest_similarity = 0.0
        
        # Check each command for similarity
        for command in self.predefined_commands:
            similarity = self.compute_similarity(recognized_text, command)
            
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_command = command
        
        # Determine if the similarity passes the threshold for this command
        if best_command:
            threshold = self.config.command_thresholds.get(
                best_command, self.config.default_similarity_threshold
            )
            if highest_similarity >= threshold:
                return best_command, highest_similarity
            
        return None, None
    
    def is_critical_command(self, command: str) -> bool:
        """Check if the command is considered critical (bypass wake word)."""
        return command in self.critical_commands


class TextToSpeech:
    """Handles text-to-speech synthesis."""
    
    def __init__(self, voice_id: int = 1):
        self.engine = pyttsx3.init()
        voices = self.engine.getProperty('voices')
        self.engine.setProperty('voice', voices[voice_id].id)
        
    def speak(self, text: str) -> None:
        """Speak the given text."""
        self.engine.say(text)
        self.engine.runAndWait()


class AudioCapture:
    """Handles audio capture from microphone."""
    
    def __init__(self, config: ASRConfig):
        self.config = config
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=config.audio_format,
            channels=config.channels,
            rate=config.rate,
            input=True,
            frames_per_buffer=config.chunk_size
        )
        self.frames = []
        self.previous_batch = None  # Store previous batch for overlap
        
    def read_audio_chunk(self) -> None:
        """Read a chunk of audio data and add to frames."""
        data = self.stream.read(self.config.chunk_size)
        self.frames.append(data)
        
    def get_audio_data(self) -> np.ndarray:
        """Convert frames to audio data with overlap and reset frames."""
        # Convert current frames to audio data
        current_audio_data = np.frombuffer(b''.join(self.frames), dtype=np.int16)
        
        # Convert stereo to mono if necessary
        if self.config.channels == 2:
            # Reshape the data to have two columns (left and right channels)
            stereo_data = current_audio_data.reshape(-1, 2)
            # Average the two channels to create mono
            current_audio_data = np.mean(stereo_data, axis=1).astype(np.int16)
            print("Stereo input detected: Converting to mono for processing")
        
        # Add overlap with previous batch to improve recognition continuity
        if self.previous_batch is not None:
            # Calculate the amount of data to keep from previous batch
            overlap_size = int(len(self.previous_batch) * self.config.overlap_ratio)
            
            # Combine the latter part of previous batch with current data
            combined_audio = np.concatenate([
                self.previous_batch[-overlap_size:],
                current_audio_data
            ])
        else:
            combined_audio = current_audio_data
        
        # Store current batch as previous for next processing
        self.previous_batch = current_audio_data
        
        # Reset current frames
        self.frames = []
        
        return combined_audio
    
    def compute_audio_fingerprint(self, audio_data: np.ndarray) -> int:
        """Compute a simple fingerprint for the audio data to detect duplicates."""
        if len(audio_data) == 0:
            return 0
            
        # Use a simple hash of the audio data as fingerprint
        # Take a sample of the audio to make it more efficient
        sample_size = min(1000, len(audio_data))
        sample_step = max(1, len(audio_data) // sample_size)
        sampled_data = audio_data[::sample_step][:sample_size]
        
        # Convert to tuple for hashing and normalize to reduce sensitivity to volume
        if np.std(sampled_data) > 0:
            normalized = (sampled_data - np.mean(sampled_data)) / np.std(sampled_data)
            # Quantize to reduce sensitivity
            quantized = (normalized * 10).astype(int)
            return hash(tuple(quantized))
        return hash(tuple(sampled_data[:100]))
        
    def has_enough_audio(self) -> bool:
        """Check if enough audio has been collected for processing."""
        return len(self.frames) >= self.config.rate // self.config.chunk_size
        
    def close(self) -> None:
        """Close the audio stream."""
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()


class SpeechRecognitionEngine:
    """Main ASR engine combining all components."""
    
    def __init__(self, config: ASRConfig = ASRConfig()):
        self.config = config

        # Set device to GPU if available, otherwise fallback to CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize predefined commands and responses
        # CUSTOMIZE: Add your own commands and responses here
        self.commands = [
            "back", "start", "stop", "slow", "fast",
            "forward", "backward", "halt", "cease", "help", 
            "cafeteria", "pool", "move", self.config.wake_word
        ]
        
        self.responses = {
            "back": "Moving backwards",
            "start": "Starting movement",
            "stop": "Stopping now",
            "slow": "Reducing speed",
            "fast": "Going faster",
            "forward": "Moving forward",
            "backward": "Moving backwards",
            "halt": "Coming to a stop",
            "cease": "Stopping all movement",
            "help": "Assistance is on the way",
            "cafeteria": "Heading to the cafeteria",
            "pool": "Going to the pool area",
            "move": "Starting to move",
            self.config.wake_word: "What's up?"
        }
        
        # Initialize components
        print(f"Loading processor from {config.base_model_path}...")
        print(f"Loading model from {config.checkpoint_path}...")
        self.processor = Wav2Vec2Processor.from_pretrained(config.base_model_path)
        self.model = Wav2Vec2ForCTC.from_pretrained(config.checkpoint_path).to(self.device)
        
        self.command_processor = PhoneticCommandProcessor(
            self.commands, 
            config
        )
        
        self.tts = TextToSpeech(config.voice_id)
        self.audio_capture = AudioCapture(config)
        
        # State variables
        self.wake_word_active = False  # Whether the wake word has been activated
        self.wake_word_detection_time = 0  # When the wake word was last detected
        self.last_timeout_check = 0  # Last time we checked for timeout
        self.timeout_check_interval = 1  # Check timeout every second
        
        # Command deduplication variables
        self.last_command_times = {cmd: 0 for cmd in self.commands}
        self.recent_audio_fingerprints = []
        
    def is_duplicate_audio(self, audio_data: np.ndarray) -> bool:
        """Check if the audio data is a duplicate of recently processed audio."""
        fingerprint = self.audio_capture.compute_audio_fingerprint(audio_data)
        
        # Check if fingerprint exists in recent history
        if fingerprint in self.recent_audio_fingerprints:
            return True
            
        # Add fingerprint to history and maintain history size
        self.recent_audio_fingerprints.append(fingerprint)
        if len(self.recent_audio_fingerprints) > self.config.fingerprint_history_size:
            self.recent_audio_fingerprints.pop(0)
            
        return False
    
    def is_command_in_cooldown(self, command: str) -> bool:
        """Check if a command is currently in cooldown period to prevent duplicates."""
        current_time = time.time()
        last_time = self.last_command_times.get(command, 0)
        
        # Check if enough time has passed since last detection
        return (current_time - last_time) < self.config.command_cooldown
    
    def process_audio(self) -> Tuple[str, bool]:
        """Process captured audio and return transcription and duplicate status."""
        audio_data = self.audio_capture.get_audio_data()
        
        # Check for duplicate audio to avoid processing the same sound twice
        is_duplicate = self.is_duplicate_audio(audio_data)
        if is_duplicate:
            return "", True
        
        # Convert audio to model input format
        input_values = self.processor(
            audio_data, 
            return_tensors="pt", 
            sampling_rate=self.config.rate
        ).input_values.float().to(self.device)
        
        # Run inference
        with torch.no_grad():
            logits = self.model(input_values).logits
        
        # Decode results
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.decode(predicted_ids[0])
        
        return transcription, False
    
    def check_wake_word_timeout(self) -> bool:
        """Check if the wake word has timed out and update state if needed."""
        if not self.wake_word_active:
            return False
            
        current_time = time.time()
        if current_time - self.wake_word_detection_time > self.config.command_timeout:
            self.wake_word_active = False
            self.tts.speak("Command mode deactivated.")
            print("Command listening timeout, say wake word again to restart.")
            return True
            
        # Calculate and print remaining time if this is an interval check
        if current_time - self.last_timeout_check >= self.timeout_check_interval:
            self.last_timeout_check = current_time
            remaining_time = int(self.config.command_timeout - (current_time - self.wake_word_detection_time))
            if remaining_time % 5 == 0 and remaining_time > 0:  # Report every 5 seconds
                print(f"Command mode active: {remaining_time} seconds remaining")
                
        return False
    
    def handle_command(self, transcription: str) -> None:
        """Process transcription and handle any recognized commands."""
        most_similar_command, similarity = self.command_processor.find_most_similar_command(transcription)
        
        if most_similar_command:
            # Check for command cooldown to prevent duplicate detections
            if self.is_command_in_cooldown(most_similar_command):
                print(f"Command '{most_similar_command}' ignored (cooldown active)")
                return
                
            # Update the last detection time for this command
            self.last_command_times[most_similar_command] = time.time()
            
            is_critical = self.command_processor.is_critical_command(most_similar_command)
            
            # Handle wake word detection
            if most_similar_command == self.config.wake_word:
                self.wake_word_active = True
                self.wake_word_detection_time = time.time()
                self.last_timeout_check = time.time()
                print(f"Wake word '{self.config.wake_word}' detected, listening for commands...")
                self.tts.speak(self.responses[self.config.wake_word])
                return
                
            # Handle critical commands - they work even without wake word activation
            if is_critical:
                self.wake_word_active = True  # Critical commands also activate listening mode
                self.wake_word_detection_time = time.time()
                self.last_timeout_check = time.time()
                print(f"Critical command '{most_similar_command}' detected!")
                self.tts.speak(self.responses[most_similar_command])
                return
                
            # Handle regular commands only if wake word is active
            if self.wake_word_active:
                # Reset the timeout when a command is recognized
                self.wake_word_detection_time = time.time()
                self.last_timeout_check = time.time()
                
                print(f"Command recognized: \033[1m|||||{most_similar_command}||||\033[0m (similarity: {similarity:.2f})")
                self.tts.speak(self.responses[most_similar_command])
            else:
                print(f"Command '{most_similar_command}' recognized but wake word not active.")
        else:
            if self.wake_word_active:
                print("No command recognized during active listening period.")
    
    def run(self):
        """Run the speech recognition system continuously."""
        print("Speech recognition system is now listening...")
        print(f"Wake word is '{self.config.wake_word}'")
        print(f"Critical commands that bypass wake word: {', '.join(self.command_processor.critical_commands)}")
        print(f"Command timeout is set to {self.config.command_timeout} seconds")
        print(f"Using {self.config.overlap_ratio * 100}% audio overlap for improved recognition")
        print(f"Command cooldown period: {self.config.command_cooldown} seconds")
        print(f"Input channels: {self.config.channels} {'(stereo input will be converted to mono)' if self.config.channels == 2 else ''}")
        
        try:
            while True:
                # Check for timeout independently of speech recognition
                self.check_wake_word_timeout()
                
                # Read audio chunk
                self.audio_capture.read_audio_chunk()
                
                if self.audio_capture.has_enough_audio():
                    transcription, is_duplicate = self.process_audio()
                    
                    # Skip processing if it's duplicate audio
                    if is_duplicate:
                        continue
                        
                    # Only print recognized text if it's not empty
                    if transcription.strip():
                        print(f"Recognized text: {transcription}")
                        self.handle_command(transcription)
                    
        except KeyboardInterrupt:
            print("Stopping speech recognition system...")
        finally:
            self.audio_capture.close()


if __name__ == "__main__":
    # CUSTOMIZE: Create a custom config with your preferred settings
    config = ASRConfig(
        # Model paths - Set these to your model locations
        checkpoint_path="final-dysarthria-model",
        
        # Recognition settings
        default_similarity_threshold=0.5,  # Lower = more sensitive, higher = more specific
        wake_word="kip",                   # Change to your preferred wake word
        command_timeout=8,                 # Seconds to wait for commands after wake word
        
        # Audio processing
        overlap_ratio=0.7,                 # Higher = more overlap between audio chunks
        chunk_size=2048,                   # Larger = more latency but better accuracy
        channels=1,                        # Set to 2 for stereo input (will be converted to mono)
        
        # Command deduplication
        command_cooldown=4.0,              # Seconds between allowed command detections
        fingerprint_history_size=5,        # Number of audio fingerprints to store
        
        # Custom thresholds for specific commands
        command_thresholds={
            "stop": 0.6,                   # Safety commands have higher thresholds
            "halt": 0.6, 
            "cease": 0.6,
            "kip": 0.65                    # Wake word requires higher confidence
        }
    )
    
    # Initialize and run the speech recognition engine
    engine = SpeechRecognitionEngine(config)
    engine.run()