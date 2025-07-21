#!/usr/bin/env python3
"""Test script to verify the backend fixes for status tracking"""

import asyncio
import requests
import time
import json
from pathlib import Path

# Configuration
API_URL = "http://localhost:8787"
TEST_AUDIO_FILE = "test_audio.wav"  # Change this to your test audio file


def create_test_audio():
    """Create a simple test audio file if it doesn't exist"""
    import numpy as np
    import scipy.io.wavfile as wav
    
    if not Path(TEST_AUDIO_FILE).exists():
        print("Creating test audio file...")
        # Generate 3 seconds of sine wave
        sample_rate = 16000
        duration = 3
        frequency = 440  # A4 note
        
        t = np.linspace(0, duration, sample_rate * duration)
        audio_data = np.sin(2 * np.pi * frequency * t)
        
        # Convert to 16-bit PCM
        audio_data = (audio_data * 32767).astype(np.int16)
        
        wav.write(TEST_AUDIO_FILE, sample_rate, audio_data)
        print(f"Created test audio file: {TEST_AUDIO_FILE}")


def test_transcription():
    """Test the transcription endpoint"""
    print("\n=== Testing Transcription API ===")
    
    # Check if test audio exists
    if not Path(TEST_AUDIO_FILE).exists():
        create_test_audio()
    
    # Upload file for transcription
    with open(TEST_AUDIO_FILE, 'rb') as f:
        files = {'file': (TEST_AUDIO_FILE, f, 'audio/wav')}
        
        print(f"Uploading {TEST_AUDIO_FILE} for transcription...")
        response = requests.post(f"{API_URL}/transcription", files=files)
        
        if response.status_code == 201:
            result = response.json()
            task_id = result['identifier']
            print(f"Task created successfully! ID: {task_id}")
            print(f"Initial status: {result['status']}")
            
            # Poll for task completion
            print("\nPolling for task completion...")
            max_attempts = 30
            for i in range(max_attempts):
                time.sleep(2)
                
                status_response = requests.get(f"{API_URL}/task/{task_id}")
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    current_status = status_data['status']
                    progress = status_data.get('progress', 0)
                    
                    print(f"Attempt {i+1}/{max_attempts}: Status={current_status}, Progress={progress*100:.1f}%")
                    
                    if current_status == 'completed':
                        print("\n✅ Task completed successfully!")
                        print(f"Duration: {status_data.get('duration', 'N/A')} seconds")
                        if status_data.get('result'):
                            print(f"Result segments: {len(status_data['result'])}")
                        break
                    elif current_status == 'failed':
                        print("\n❌ Task failed!")
                        print(f"Error: {status_data.get('error', 'Unknown error')}")
                        break
                else:
                    print(f"Failed to get task status: {status_response.status_code}")
            else:
                print("\n⚠️  Task did not complete within timeout period")
                print("This might indicate the task is stuck in IN_PROGRESS state")
        else:
            print(f"Failed to create task: {response.status_code}")
            print(f"Response: {response.text}")


def test_all_tasks_status():
    """Test getting all tasks status"""
    print("\n=== Testing All Tasks Status ===")
    
    response = requests.get(f"{API_URL}/task/all")
    if response.status_code == 200:
        tasks = response.json()['tasks']
        print(f"Total tasks in database: {len(tasks)}")
        
        # Count by status
        status_counts = {}
        for task in tasks:
            status = task[1]  # status is second element in tuple
            status_counts[status] = status_counts.get(status, 0) + 1
        
        print("\nTasks by status:")
        for status, count in status_counts.items():
            print(f"  {status}: {count}")
            
        # Check for stuck tasks
        in_progress_tasks = [t for t in tasks if t[1] == 'in_progress']
        if in_progress_tasks:
            print(f"\n⚠️  Found {len(in_progress_tasks)} tasks stuck in IN_PROGRESS state")
    else:
        print(f"Failed to get tasks: {response.status_code}")


def main():
    """Run all tests"""
    print("Starting backend fix verification...")
    print(f"API URL: {API_URL}")
    
    # Check if backend is running
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code != 200:
            print("❌ Backend is not responding. Please start the backend first.")
            return
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to backend. Please start the backend first.")
        print("Run: python run_backend.py")
        return
    
    print("✅ Backend is running")
    
    # Run tests
    test_all_tasks_status()
    test_transcription()
    
    print("\n=== Test Complete ===")


if __name__ == "__main__":
    main()