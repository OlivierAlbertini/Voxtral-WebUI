import functools
import uuid
import numpy as np
from fastapi import (
    File,
    UploadFile,
)
import gradio as gr
from fastapi import APIRouter, BackgroundTasks, Depends, Response, status
from typing import List, Dict, Optional
from sqlalchemy.orm import Session
from datetime import datetime
from modules.whisper.data_classes import *
from modules.utils.paths import BACKEND_CACHE_DIR
from modules.whisper.whisper_factory import WhisperFactory
from modules.whisper.base_transcription_pipeline import BaseTranscriptionPipeline
from backend.common.audio import read_audio
from backend.common.models import QueueResponse
from backend.common.config_loader import load_server_config
from backend.db.task.dao import (
    add_task_to_db,
    get_db_session,
    update_task_status_in_db
)
from backend.db.task.models import TaskStatus, TaskType
from modules.utils.logger import get_logger
import traceback
import torch
import gc

logger = get_logger()

transcription_router = APIRouter(prefix="/transcription", tags=["Transcription"])


class MockProgress:
    """Mock Progress class for FastAPI context to replace gr.Progress()"""
    def __init__(self, identifier: Optional[str] = None):
        self.identifier = identifier
        
    def __call__(self, progress: float, desc: str = ""):
        """Update progress - in FastAPI context, we log instead of updating UI"""
        if self.identifier:
            logger.debug(f"Task {self.identifier}: {desc} ({progress*100:.1f}%)")
        return self


def create_progress_callback(identifier: str):
    def progress_callback(progress_value: float):
        try:
            update_task_status_in_db(
                identifier=identifier,
                update_data={
                    "uuid": identifier,
                    "status": TaskStatus.IN_PROGRESS,
                    "progress": round(progress_value, 2),
                    "updated_at": datetime.utcnow()
                },
            )
            logger.debug(f"Task {identifier} progress updated: {progress_value*100:.1f}%")
        except Exception as e:
            logger.error(f"Error updating progress for task {identifier}: {str(e)}")
            # Don't propagate the error - we don't want to fail the whole task due to progress update failure
    return progress_callback


@functools.lru_cache
def get_pipeline() -> 'BaseTranscriptionPipeline':
    config = load_server_config()["whisper"]
    
    # Get whisper type from config or determine based on model_size
    whisper_type = config.get("whisper_type", None)
    if not whisper_type:
        # Fallback to determining type from model_size
        model_size = config["model_size"]
        if model_size == "voxtral-mini-3b":
            whisper_type = WhisperImpl.VOXTRAL_MINI.value
        else:
            whisper_type = WhisperImpl.FASTER_WHISPER.value
    
    logger.info(f"Creating pipeline with whisper_type: {whisper_type}, model_size: {config['model_size']}")
    
    try:
        inferencer = WhisperFactory.create_whisper_inference(
            whisper_type=whisper_type,
            output_dir=BACKEND_CACHE_DIR
        )
        
        # Use a mock progress for model initialization in FastAPI context
        mock_progress = MockProgress()
        inferencer.update_model(
            model_size=config["model_size"],
            compute_type=config["compute_type"],
            progress=mock_progress
        )
        
        logger.info(f"Pipeline created successfully with model: {config['model_size']}")
        return inferencer
        
    except Exception as e:
        logger.error(f"Error creating pipeline: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise


def run_transcription(
    audio: np.ndarray,
    params: TranscriptionPipelineParams,
    identifier: str,
) -> List[Segment]:
    try:
        logger.info(f"Starting transcription for task {identifier}")
        
        update_task_status_in_db(
            identifier=identifier,
            update_data={
                "uuid": identifier,
                "status": TaskStatus.IN_PROGRESS,
                "updated_at": datetime.utcnow(),
                "progress": 0.0,
            },
        )

        progress_callback = create_progress_callback(identifier)
        
        # Use MockProgress instead of gr.Progress() for FastAPI context
        mock_progress = MockProgress(identifier)
        
        logger.debug(f"Running pipeline for task {identifier}")
        segments, elapsed_time = get_pipeline().run(
            audio,
            mock_progress,  # Use mock progress instead of gr.Progress()
            "SRT",
            False,
            progress_callback,  
            *params.to_list()
        )
        
        logger.debug(f"Pipeline completed for task {identifier}, converting segments")
        segments = [seg.model_dump() for seg in segments]

        update_task_status_in_db(
            identifier=identifier,
            update_data={
                "uuid": identifier,
                "status": TaskStatus.COMPLETED,
                "result": segments,
                "updated_at": datetime.utcnow(),
                "duration": elapsed_time,
                "progress": 1.0,
            },
        )
        
        logger.info(f"Task {identifier} completed successfully in {elapsed_time:.2f}s")
        
        # Clean up memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return segments
        
    except Exception as e:
        error_msg = f"Error in transcription task {identifier}: {str(e)}"
        logger.error(error_msg)
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Update task status to FAILED
        update_task_status_in_db(
            identifier=identifier,
            update_data={
                "uuid": identifier,
                "status": TaskStatus.FAILED,
                "error": error_msg,
                "updated_at": datetime.utcnow(),
            },
        )
        
        # Clean up memory even on failure
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # Re-raise the exception to let FastAPI handle it
        raise


@transcription_router.post(
    "/",
    response_model=QueueResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Transcribe Audio",
    description="Process the provided audio or video file to generate a transcription.",
)
async def transcription(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Audio or video file to transcribe."),
    whisper_params: WhisperParams = Depends(),
    vad_params: VadParams = Depends(),
    bgm_separation_params: BGMSeparationParams = Depends(),
    diarization_params: DiarizationParams = Depends(),
) -> QueueResponse:
    try:
        logger.info(f"Received transcription request for file: {file.filename if hasattr(file, 'filename') else 'unknown'}")
        
        if not isinstance(file, np.ndarray):
            audio, info = await read_audio(file=file)
        else:
            audio, info = file, None

        params = TranscriptionPipelineParams(
            whisper=whisper_params,
            vad=vad_params,
            bgm_separation=bgm_separation_params,
            diarization=diarization_params
        )

        identifier = add_task_to_db(
            status=TaskStatus.QUEUED,
            file_name=file.filename if hasattr(file, 'filename') else None,
            audio_duration=info.duration if info else None,
            language=params.whisper.lang,
            task_type=TaskType.TRANSCRIPTION,
            task_params=params.to_dict(),
        )
        
        logger.info(f"Created task {identifier} for file: {file.filename if hasattr(file, 'filename') else 'unknown'}")

        background_tasks.add_task(
            run_transcription,
            audio=audio,
            params=params,
            identifier=identifier,
        )

        return QueueResponse(identifier=identifier, status=TaskStatus.QUEUED, message="Transcription task has queued")
        
    except Exception as e:
        logger.error(f"Error creating transcription task: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise


