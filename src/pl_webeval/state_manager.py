"""
State management for test execution - tracks progress without modifying input files.
Uses a separate JSON state file for robustness and reproducibility.
"""

import json
import os
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Set
import hashlib


class StateManager:
    """Manages test execution state in a separate file for robust, resumable testing."""
    
    def __init__(self, output_dir: Path, test_cases_hash: str = None):
        """
        Initialize state manager.
        
        Args:
            output_dir: Directory where results are stored
            test_cases_hash: Hash of test cases for validation
        """
        self.output_dir = output_dir
        self.state_file = output_dir / ".test_state.json"
        self.test_cases_hash = test_cases_hash
        self.state = self._load_state()
    
    def _load_state(self) -> Dict:
        """Load existing state or create new one."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                    # Validate state is for same test set
                    if state.get('test_cases_hash') != self.test_cases_hash:
                        print("⚠️ Test cases have changed, starting fresh run")
                        return self._create_new_state()
                    return state
            except Exception as e:
                print(f"⚠️ Could not load state file: {e}, starting fresh")
                return self._create_new_state()
        return self._create_new_state()
    
    def _create_new_state(self) -> Dict:
        """Create a new state structure."""
        return {
            'test_cases_hash': self.test_cases_hash,
            'started_at': datetime.now().isoformat(),
            'completed_tests': {},
            'failed_tests': {},
            'skipped_tests': {},
            'in_progress': None,
            'statistics': {
                'total': 0,
                'completed': 0,
                'failed': 0,
                'skipped': 0
            }
        }
    
    def _save_state(self):
        """Persist state to disk."""
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(self.state, f, indent=2, default=str)
        except Exception as e:
            print(f"⚠️ Could not save state: {e}")
    
    def is_completed(self, test_id: str) -> bool:
        """Check if a test has been completed."""
        return test_id in self.state['completed_tests']
    
    def is_failed(self, test_id: str) -> bool:
        """Check if a test has failed."""
        return test_id in self.state['failed_tests']
    
    def should_skip(self, test_id: str) -> bool:
        """Check if a test should be skipped (already completed or explicitly skipped)."""
        return self.is_completed(test_id) or test_id in self.state['skipped_tests']
    
    def mark_in_progress(self, test_id: str):
        """Mark a test as currently running."""
        self.state['in_progress'] = {
            'test_id': test_id,
            'started_at': datetime.now().isoformat()
        }
        self._save_state()
    
    def mark_completed(self, test_id: str, result_summary: Dict = None):
        """Mark a test as completed."""
        self.state['completed_tests'][test_id] = {
            'completed_at': datetime.now().isoformat(),
            'result': result_summary or {}
        }
        self.state['statistics']['completed'] += 1
        if self.state['in_progress'] and self.state['in_progress'].get('test_id') == test_id:
            self.state['in_progress'] = None
        self._save_state()
    
    def mark_failed(self, test_id: str, error: str):
        """Mark a test as failed."""
        self.state['failed_tests'][test_id] = {
            'failed_at': datetime.now().isoformat(),
            'error': error
        }
        self.state['statistics']['failed'] += 1
        if self.state['in_progress'] and self.state['in_progress'].get('test_id') == test_id:
            self.state['in_progress'] = None
        self._save_state()
    
    def mark_skipped(self, test_id: str, reason: str = "User requested"):
        """Mark a test as skipped."""
        self.state['skipped_tests'][test_id] = {
            'skipped_at': datetime.now().isoformat(),
            'reason': reason
        }
        self.state['statistics']['skipped'] += 1
        self._save_state()
    
    def get_progress_summary(self) -> Dict:
        """Get a summary of test progress."""
        return {
            'total': self.state['statistics']['total'],
            'completed': self.state['statistics']['completed'],
            'failed': self.state['statistics']['failed'],
            'skipped': self.state['statistics']['skipped'],
            'remaining': self.state['statistics']['total'] - 
                        (self.state['statistics']['completed'] + 
                         self.state['statistics']['failed'] + 
                         self.state['statistics']['skipped']),
            'in_progress': self.state['in_progress']
        }
    
    def set_total_tests(self, total: int):
        """Set the total number of tests."""
        self.state['statistics']['total'] = total
        self._save_state()
    
    def get_resume_point(self) -> Optional[str]:
        """Get the test ID that was in progress (for resuming after crash)."""
        if self.state['in_progress']:
            return self.state['in_progress'].get('test_id')
        return None
    
    @staticmethod
    def compute_test_cases_hash(test_cases) -> str:
        """Compute a hash of test cases to detect changes."""
        # Create a string representation of test cases
        test_str = ""
        for tc in test_cases:
            test_str += f"{tc.website}|{tc.ux_profile}|{tc.llm_model}|"
        return hashlib.md5(test_str.encode()).hexdigest()


class CheckpointManager:
    """Manages checkpoints for long-running tests to enable fine-grained resume."""
    
    @staticmethod
    def sanitize_id(test_id: str) -> str:
        """Sanitize test ID for use as directory name."""
        # Replace invalid characters with underscores
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', test_id)
        # Replace multiple underscores with single
        sanitized = re.sub(r'_+', '_', sanitized)
        # Limit length to avoid path issues
        if len(sanitized) > 200:
            # Use hash for very long IDs
            hash_suffix = hashlib.md5(test_id.encode()).hexdigest()[:8]
            sanitized = sanitized[:190] + "_" + hash_suffix
        return sanitized
    
    def __init__(self, output_dir: Path, test_id: str):
        """Initialize checkpoint manager for a specific test."""
        # Sanitize the test_id for filesystem compatibility
        safe_test_id = self.sanitize_id(test_id)
        self.checkpoint_dir = output_dir / ".checkpoints" / safe_test_id
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(self, phase: str, data: Dict):
        """Save a checkpoint for a test phase."""
        checkpoint_file = self.checkpoint_dir / f"{phase}.json"
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump({
                'phase': phase,
                'timestamp': datetime.now().isoformat(),
                'data': data
            }, f, indent=2, default=str)
    
    def load_checkpoint(self, phase: str) -> Optional[Dict]:
        """Load a checkpoint for a test phase."""
        checkpoint_file = self.checkpoint_dir / f"{phase}.json"
        if checkpoint_file.exists():
            try:
                with open(checkpoint_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return None
        return None
    
    def has_checkpoint(self, phase: str) -> bool:
        """Check if a checkpoint exists for a phase."""
        return (self.checkpoint_dir / f"{phase}.json").exists()
    
    def clear_checkpoints(self):
        """Clear all checkpoints for this test."""
        import shutil
        if self.checkpoint_dir.exists():
            shutil.rmtree(self.checkpoint_dir)