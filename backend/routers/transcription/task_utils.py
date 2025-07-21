"""Utilities for handling transcription tasks"""
import asyncio
from datetime import datetime, timedelta
from typing import Optional
from backend.db.task.dao import get_all_tasks_status_from_db, update_task_status_in_db
from backend.db.task.models import TaskStatus
from modules.utils.logger import get_logger

logger = get_logger()


async def check_stale_tasks(timeout_minutes: int = 30):
    """
    Check for tasks that have been in IN_PROGRESS state for too long
    and mark them as FAILED.
    
    Args:
        timeout_minutes: Number of minutes after which a task is considered stale
    """
    try:
        all_tasks = get_all_tasks_status_from_db()
        if not all_tasks or not all_tasks.tasks:
            return
            
        current_time = datetime.utcnow()
        timeout_delta = timedelta(minutes=timeout_minutes)
        
        for task in all_tasks.tasks:
            if task.status == TaskStatus.IN_PROGRESS:
                # Check if task has been in progress for too long
                if hasattr(task, 'updated_at') and task.updated_at:
                    time_diff = current_time - task.updated_at
                    if time_diff > timeout_delta:
                        logger.warning(f"Task {task.uuid} has been in progress for {time_diff}. Marking as failed.")
                        update_task_status_in_db(
                            identifier=task.uuid,
                            update_data={
                                "uuid": task.uuid,
                                "status": TaskStatus.FAILED,
                                "error": f"Task timed out after {timeout_minutes} minutes",
                                "updated_at": current_time
                            }
                        )
                        
    except Exception as e:
        logger.error(f"Error checking stale tasks: {str(e)}")


async def periodic_stale_task_checker(interval_minutes: int = 5, timeout_minutes: int = 30):
    """
    Periodically check for stale tasks.
    
    Args:
        interval_minutes: How often to check for stale tasks
        timeout_minutes: Number of minutes after which a task is considered stale
    """
    while True:
        await check_stale_tasks(timeout_minutes)
        await asyncio.sleep(interval_minutes * 60)