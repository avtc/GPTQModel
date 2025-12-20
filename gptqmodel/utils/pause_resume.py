# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Simple pause/resume functionality for GPTQModel quantization.

Provides thread-safe pause/resume capabilities with keyboard input handling.
"""

import threading
import logging
from enum import Enum
from typing import Optional, Callable
from contextlib import contextmanager
from pynput import keyboard
from logbar import LogBar

log = logging.getLogger(__name__)


class PauseResumeState(Enum):
    """Enumeration of possible pause/resume states."""
    RUNNING = "running"
    PAUSE_REQUESTED = "pause_requested"
    PAUSED = "paused"


class PauseResumeController:
    """
    Simple thread-safe controller for managing pause/resume during quantization.

    Provides:
    - Thread-safe state management
    - Keyboard input handling (Pause/Break key)
    - Integration with progress tracking
    """

    def __init__(self, enable_keyboard: bool = True):
        """
        Initialize the pause/resume controller.

        Args:
            enable_keyboard: Whether to enable keyboard input handling
        """
        self._state = PauseResumeState.RUNNING
        self._state_lock = threading.RLock()
        self._pause_event = threading.Event()
        self._resume_event = threading.Event()

        # Keyboard handling
        self._keyboard_enabled = enable_keyboard
        self._keyboard_active = False
        self._keyboard_listener = None

        # Callbacks for status updates
        self._status_callback: Optional[Callable[[PauseResumeState], None]] = None

        # Status bar for pause/resume
        self._status_bar = None
        self._setup_status_bar()

        # Initialize events
        self._resume_event.set()  # Allow execution to start

        if self._keyboard_enabled:
            self._setup_keyboard_handler()

    def _setup_status_bar(self):
        """Setup a dedicated status bar for pause/resume information."""
        try:
            logger = LogBar.shared()
            # Create a manual progress bar for status (using range(1) for static display)
            self._status_bar = logger.pb(range(1)).title("Pause/Resume Status").manual()
            self._update_status_bar()
        except Exception as e:
            log.warning(f"Failed to setup pause/resume status bar: {e}")
            self._status_bar = None

    def _update_status_bar(self):
        """Update the pause/resume status bar."""
        if not self._status_bar:
            return

        try:
            state = self.get_state()
            if state == PauseResumeState.RUNNING:
                status_msg = "Running (Press 'p' or Pause/Break to pause)"
            elif state == PauseResumeState.PAUSE_REQUESTED:
                status_msg = "Pause requested - will pause after current layer"
            elif state == PauseResumeState.PAUSED:
                status_msg = "Paused (Press 'p' or Pause/Break to resume)"
            else:
                status_msg = state.value

            self._status_bar.subtitle(status_msg).draw()
        except Exception as e:
            log.warning(f"Failed to update status bar: {e}")

    def _setup_keyboard_handler(self):
        """Setup keyboard event handlers for pause/break keys."""

        def on_key_press(key):
            try:
                # Convert pynput key to string representation
                key_str = str(key).lower()

                # Handle different key formats
                if hasattr(key, 'char') and key.char:
                    key_char = key.char.lower()
                    if key_char == 'p':
                        self.toggle_pause_resume()
                        return

                # Handle special keys
                if 'pause' in key_str or 'break' in key_str:
                    self.toggle_pause_resume()
            except Exception as e:
                log.warning(f"Keyboard handler error: {e}")

        try:
            self._keyboard_listener = keyboard.Listener(on_press=on_key_press)
            self._keyboard_listener.start()
            self._keyboard_active = True
            log.info("Keyboard pause/resume enabled (Press 'p' or Pause/Break to toggle)")
        except Exception as e:
            log.warning(f"Failed to setup keyboard handler: {e}")
            self._keyboard_active = False
            self._keyboard_listener = None

    def set_status_callback(self, callback: Callable[[PauseResumeState], None]):
        """Set callback for state changes."""
        with self._state_lock:
            self._status_callback = callback

    def _set_state(self, new_state: PauseResumeState):
        """Internal method to change state with notification."""
        with self._state_lock:
            old_state = self._state
            self._state = new_state

            # Update events based on state
            if new_state == PauseResumeState.PAUSED:
                self._pause_event.set()
                self._resume_event.clear()
            elif new_state == PauseResumeState.RUNNING:
                self._pause_event.clear()
                self._resume_event.set()

            # Notify callback
            if self._status_callback and old_state != new_state:
                try:
                    self._status_callback(new_state)
                except Exception as e:
                    log.warning(f"Status callback error: {e}")

            # Update status bar
            self._update_status_bar()

            log.info(f"Pause/Resume state: {old_state.value} -> {new_state.value}")

    def get_state(self) -> PauseResumeState:
        """Get current state."""
        with self._state_lock:
            return self._state

    def pause(self):
        """Request pause - will pause at next safe point."""
        with self._state_lock:
            if self._state == PauseResumeState.RUNNING:
                self._set_state(PauseResumeState.PAUSE_REQUESTED)
            elif self._state == PauseResumeState.PAUSED:
                log.info("Already paused")
            elif self._state == PauseResumeState.PAUSE_REQUESTED:
                log.info("Pause already requested")

    def resume(self):
        """Resume quantization."""
        with self._state_lock:
            if self._state == PauseResumeState.PAUSED:
                self._set_state(PauseResumeState.RUNNING)
            elif self._state == PauseResumeState.PAUSE_REQUESTED:
                log.info("Resuming from pause requested state")
                self._set_state(PauseResumeState.RUNNING)
            elif self._state == PauseResumeState.RUNNING:
                log.info("Already running")

    def toggle_pause_resume(self):
        """Toggle between pause and resume states."""
        current_state = self.get_state()
        if current_state == PauseResumeState.RUNNING:
            self.pause()
        elif current_state in [PauseResumeState.PAUSED, PauseResumeState.PAUSE_REQUESTED]:
            self.resume()

    def check_pause_point(self, layer_info: Optional[str] = None) -> bool:
        """
        Check if we should pause at this point.

        Called between layers or other safe points.

        Args:
            layer_info: Optional description of current layer for logging

        Returns:
            Always True (execution continues)
        """
        # Check if pause was requested
        with self._state_lock:
            if self._state == PauseResumeState.PAUSE_REQUESTED:
                self._set_state(PauseResumeState.PAUSED)
                layer_msg = f" after {layer_info}" if layer_info else ""
                log.info(f"⏸️  Quantization paused{layer_msg}. Press 'p' or Pause/Break to resume.")

        # Wait if paused
        if self._pause_event.is_set():
            while self._pause_event.is_set():
                if self._resume_event.wait(timeout=0.1):
                    with self._state_lock:
                        if self._state == PauseResumeState.PAUSED:
                            self._set_state(PauseResumeState.RUNNING)
                            layer_msg = f" after {layer_info}" if layer_info else ""
                            log.info(f"▶️  Quantization resumed{layer_msg}")
                        break

        return True

    def is_paused(self) -> bool:
        """Check if currently paused or pause requested."""
        with self._state_lock:
            return self._state in [PauseResumeState.PAUSED, PauseResumeState.PAUSE_REQUESTED]

    def is_running(self) -> bool:
        """Check if currently running."""
        with self._state_lock:
            return self._state == PauseResumeState.RUNNING

    @contextmanager
    def pause_context(self, operation_name: str):
        """
        Context manager for operations that should respect pause state.

        Args:
            operation_name: Name of the operation for logging
        """
        self.check_pause_point(operation_name)
        try:
            yield
        finally:
            # Check pause point again after operation
            self.check_pause_point(f"after {operation_name}")

    def cleanup(self):
        """Cleanup resources and keyboard handlers."""
        # Cleanup status bar
        if self._status_bar:
            try:
                self._status_bar.close()
                self._status_bar = None
            except Exception as e:
                log.warning(f"Error closing status bar: {e}")

        # Cleanup keyboard handlers
        if self._keyboard_active and self._keyboard_listener:
            try:
                self._keyboard_listener.stop()
                self._keyboard_listener = None
                self._keyboard_active = False
                log.info("Keyboard handlers cleaned up")
            except Exception as e:
                log.warning(f"Error cleaning up keyboard handlers: {e}")

        # Set final state to running for clean shutdown
        with self._state_lock:
            if self._state != PauseResumeState.RUNNING:
                self._set_state(PauseResumeState.RUNNING)


# Convenience function for quick setup
def create_pause_controller(enable_keyboard: bool = True) -> PauseResumeController:
    """
    Create a pause/resume controller with sensible defaults.

    Args:
        enable_keyboard: Whether to enable keyboard input handling

    Returns:
        Configured PauseResumeController instance
    """
    return PauseResumeController(enable_keyboard=enable_keyboard)