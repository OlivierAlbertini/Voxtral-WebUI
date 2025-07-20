import os
import time
import torch
import numpy as np
from typing import BinaryIO, Union, Tuple, List, Callable, Optional
import gradio as gr
import tempfile
import librosa

try:
    from transformers import VoxtralForConditionalGeneration, AutoProcessor
    VOXTRAL_AVAILABLE = True
except ImportError:
    print("Warning: VoxtralForConditionalGeneration not available. Please install latest transformers:")
    print("pip uninstall transformers -y")
    print("pip install git+https://github.com/huggingface/transformers.git")
    VOXTRAL_AVAILABLE = False
    VoxtralForConditionalGeneration = None
    AutoProcessor = None

from modules.utils.paths import (VOXTRAL_MODELS_DIR, DIARIZATION_MODELS_DIR, UVR_MODELS_DIR, OUTPUT_DIR)
from modules.whisper.data_classes import *
from modules.whisper.base_transcription_pipeline import BaseTranscriptionPipeline


class VoxtralWhisperInference(BaseTranscriptionPipeline):
    def __init__(self,
                 model_dir: str = VOXTRAL_MODELS_DIR,
                 diarization_model_dir: str = DIARIZATION_MODELS_DIR,
                 uvr_model_dir: str = UVR_MODELS_DIR,
                 output_dir: str = OUTPUT_DIR,
                 ):
        if not VOXTRAL_AVAILABLE:
            raise ImportError(
                "Voxtral is not available. Please install the latest transformers:\n"
                "pip uninstall transformers -y\n"
                "pip install git+https://github.com/huggingface/transformers.git"
            )
        super().__init__(
            model_dir=model_dir,
            diarization_model_dir=diarization_model_dir,
            uvr_model_dir=uvr_model_dir,
            output_dir=output_dir
        )
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.processor = None
        self.device = self.get_device()
        self.repo_id = "mistralai/Voxtral-Mini-3B-2507"
        self.available_models = ["voxtral-mini-3b"]
        
    def _segment_audio(self, audio_path: str, chunk_length: int = 1500, chunk_overlap: int = 2) -> List[Tuple[float, float, str]]:
        """
        Segment audio into chunks for processing.
        
        Parameters
        ----------
        audio_path: str
            Path to the audio file
        chunk_length: int
            Length of each chunk in seconds (default: 1500 = 25 minutes)
        chunk_overlap: int
            Overlap between chunks in seconds (default: 2)
            
        Returns
        -------
        List[Tuple[float, float, str]]
            List of (start_time, end_time, chunk_audio_path) tuples
        """
        audio_data, sr = librosa.load(audio_path, sr=None)
        audio_duration = len(audio_data) / sr
        
        chunks = []
        chunk_start = 0.0
        
        while chunk_start < audio_duration:
            chunk_end = min(chunk_start + chunk_length, audio_duration)
            
            # Extract chunk audio data
            start_sample = int(chunk_start * sr)
            end_sample = int(chunk_end * sr)
            chunk_data = audio_data[start_sample:end_sample]
            
            # Save chunk to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                chunk_path = temp_file.name
                import soundfile as sf
                sf.write(chunk_path, chunk_data, sr)
            
            chunks.append((chunk_start, chunk_end, chunk_path))
            
            # Move to next chunk with overlap
            chunk_start = chunk_end - chunk_overlap
            
            # If we're at the end, don't create a tiny chunk
            if chunk_start >= audio_duration - chunk_overlap:
                break
                
        return chunks
    
    def transcribe(self,
                   audio: Union[str, BinaryIO, np.ndarray],
                   progress: gr.Progress = gr.Progress(),
                   progress_callback: Optional[Callable] = None,
                   *whisper_params,
                   ) -> Tuple[List[Segment], float]:
        """
        Transcribe method for voxtral-mini.

        Parameters
        ----------
        audio: Union[str, BinaryIO, np.ndarray]
            Audio path or file binary or Audio numpy array
        progress: gr.Progress
            Indicator to show progress directly in gradio.
        progress_callback: Optional[Callable]
            callback function to show progress. Can be used to update progress in the backend.
        *whisper_params: tuple
            Parameters related with whisper. This will be dealt with "WhisperParameters" data class

        Returns
        ----------
        segments_result: List[Segment]
            list of Segment that includes start, end timestamps and transcribed text
        elapsed_time: float
            elapsed time for transcription
        """
        start_time = time.time()
        
        params = WhisperParams.from_list(list(whisper_params))
        
        if params.model_size != self.current_model_size or self.model is None:
            self.update_model(params.model_size, params.compute_type, progress)
        
        progress(0.1, desc="Processing audio...")
        
        # Convert audio to the format expected by voxtral
        audio_path = self._prepare_audio(audio)
        temp_files_to_cleanup = []
        
        try:
            # Get audio duration and determine if we need to segment
            audio_duration = self._get_audio_duration(audio_path)
            chunk_length = getattr(params, 'chunk_length', 1500)  # Default 25 minutes
            chunk_overlap = 2  # 2 seconds overlap
            
            segments_result = []
            
            # Check if audio is longer than chunk_length
            if audio_duration > chunk_length:
                progress(0.2, desc="Segmenting audio for long file processing...")
                
                # Segment the audio
                audio_chunks = self._segment_audio(audio_path, chunk_length, chunk_overlap)
                temp_files_to_cleanup.extend([chunk[2] for chunk in audio_chunks])
                
                total_chunks = len(audio_chunks)
                
                for i, (chunk_start, chunk_end, chunk_path) in enumerate(audio_chunks):
                    chunk_progress = 0.2 + (0.7 * i / total_chunks)
                    progress(chunk_progress, desc=f"Transcribing chunk {i+1}/{total_chunks} ({chunk_start:.0f}s - {chunk_end:.0f}s)")
                    
                    # Determine language for first chunk or use provided language
                    language = params.lang if params.lang else "en"
                    
                    # Apply transcription request for this chunk
                    inputs = self.processor.apply_transcrition_request(
                        language=language, 
                        audio=chunk_path, 
                        model_id=self.repo_id
                    )
                    inputs = inputs.to(self.device, dtype=torch.bfloat16)
                    
                    # Generate transcription for this chunk
                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs, 
                            max_new_tokens=params.max_new_tokens or 32000,
                            temperature=params.temperature if params.temperature > 0 else 0.0,
                            do_sample=params.temperature > 0
                        )
                    
                    # Decode outputs
                    decoded_outputs = self.processor.batch_decode(
                        outputs[:, inputs.input_ids.shape[1]:], 
                        skip_special_tokens=True
                    )
                    
                    # Create segment from transcription with adjusted timestamps
                    transcription_text = decoded_outputs[0].strip() if decoded_outputs else ""
                    
                    if transcription_text:  # Only add non-empty segments
                        segment = Segment(
                            id=i,
                            text=transcription_text,
                            start=chunk_start,
                            end=chunk_end,
                            seek=int(chunk_start),
                            tokens=None,
                            temperature=params.temperature,
                            avg_logprob=None,
                            compression_ratio=None,
                            no_speech_prob=None,
                            words=None
                        )
                        segments_result.append(segment)
                    
                    # Free GPU memory between chunks if needed
                    if i < total_chunks - 1:
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            else:
                # Process as single segment for short audio
                progress(0.3, desc="Transcribing with Voxtral...")
                
                language = params.lang if params.lang else "en"
                
                inputs = self.processor.apply_transcrition_request(
                    language=language, 
                    audio=audio_path, 
                    model_id=self.repo_id
                )
                inputs = inputs.to(self.device, dtype=torch.bfloat16)
                
                progress(0.6, desc="Generating transcription...")
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs, 
                        max_new_tokens=params.max_new_tokens or 32000,
                        temperature=params.temperature if params.temperature > 0 else 0.0,
                        do_sample=params.temperature > 0
                    )
                
                progress(0.8, desc="Processing results...")
                
                decoded_outputs = self.processor.batch_decode(
                    outputs[:, inputs.input_ids.shape[1]:], 
                    skip_special_tokens=True
                )
                
                transcription_text = decoded_outputs[0].strip() if decoded_outputs else ""
                
                segments_result = [
                    Segment(
                        id=0,
                        text=transcription_text,
                        start=0.0,
                        end=audio_duration,
                        seek=0,
                        tokens=None,
                        temperature=params.temperature,
                        avg_logprob=None,
                        compression_ratio=None,
                        no_speech_prob=None,
                        words=None
                    )
                ]
            
            progress(1.0, desc="Transcription completed!")
            
        finally:
            # Clean up temporary files
            if isinstance(audio, np.ndarray) or (isinstance(audio, str) and audio_path != audio):
                if os.path.exists(audio_path):
                    os.unlink(audio_path)
            
            # Clean up chunk files
            for temp_file in temp_files_to_cleanup:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
        
        elapsed_time = time.time() - start_time
        return segments_result, elapsed_time

    def update_model(self,
                     model_size: str,
                     compute_type: str,
                     progress: gr.Progress = gr.Progress()
                     ):
        """
        Update current model setting for voxtral-mini

        Parameters
        ----------
        model_size: str
            Model size identifier (for voxtral, this will be "voxtral-mini-3b")
        compute_type: str
            Compute type for transcription (handled by device selection)
        progress: gr.Progress
            Indicator to show progress directly in gradio.
        """
        progress(0, desc="Initializing Voxtral Model...")
        
        if self.model is not None and self.current_model_size == model_size:
            return
            
        # Load processor
        progress(0.3, desc="Loading processor...")
        self.processor = AutoProcessor.from_pretrained(self.repo_id)
        
        # Load model
        progress(0.6, desc="Loading model...")
        self.model = VoxtralForConditionalGeneration.from_pretrained(
            self.repo_id,
            torch_dtype=torch.bfloat16,
            device_map=self.device
        )
        
        self.current_model_size = model_size
        self.current_compute_type = compute_type
        
        progress(1.0, desc="Model loaded successfully!")

    def _prepare_audio(self, audio: Union[str, BinaryIO, np.ndarray]) -> str:
        """
        Prepare audio for voxtral processing.
        Voxtral expects audio file paths, so we need to convert other formats.
        
        Returns
        -------
        str: Path to audio file
        """
        if isinstance(audio, str):
            # Already a file path
            return audio
        elif isinstance(audio, np.ndarray):
            # Convert numpy array to temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            temp_path = temp_file.name
            temp_file.close()
            
            # Use librosa to save the numpy array as audio file
            import soundfile as sf
            sf.write(temp_path, audio, 16000)
            return temp_path
        else:
            # Binary IO - save to temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            temp_path = temp_file.name
            
            audio.seek(0)
            temp_file.write(audio.read())
            temp_file.close()
            
            return temp_path

    def _get_audio_duration(self, audio_path: str) -> float:
        """Get audio duration in seconds"""
        try:
            audio_data, sr = librosa.load(audio_path, sr=None)
            return len(audio_data) / sr
        except:
            # Fallback duration if we can't determine it
            return 30.0

    def get_available_compute_type(self):
        """Return available compute types for voxtral"""
        if self.device == "cuda":
            return ["float16", "bfloat16"]
        else:
            return ["float32"]

    def get_compute_type(self):
        """Get default compute type for voxtral"""
        if self.device == "cuda":
            return "bfloat16"
        else:
            return "float32"