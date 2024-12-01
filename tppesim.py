
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from enum import Enum
import numpy as np
from collections import deque

@dataclass
class TPPEConfig:
    num_timesteps: int = 4
    bitmask_width: int = 128
    weight_buffer_size: int = 128 
    fifo_depth: int = 8

class PipelineStage(Enum):
    FAST_PREFIX = 0  # Fast prefix sum computation
    PSEUDO_ACC = 1   # Pseudo accumulation
    LOADING = 2      # Loading/waiting for laggy prefix
    CORRECTION = 3   # Correction phase
    DONE = 4

class FIFOBuffer:
    def __init__(self, depth: int, name: str):
        self.buffer = deque(maxlen=depth)
        self.depth = depth
        self.name = name

    def push(self, value) -> bool:
        if len(self.buffer) < self.depth:
            self.buffer.append(value)
            return True
        return False

    def pop(self) -> Optional[any]:
        return self.buffer.popleft() if self.buffer else None

    def is_empty(self) -> bool:
        return len(self.buffer) == 0

    def __len__(self):
        return len(self.buffer)

class TPPE:
    def __init__(self, config: TPPEConfig, debug=True):
        self.config = config
        self.debug = debug
        self.reset()

    def reset(self):
        # Data registers
        self.bitmask_a = None
        self.bitmask_b = None
        self.fiber_a_data = None
        self.fiber_b_data = None

        # Pipeline registers
        self.fifo_b = FIFOBuffer(self.config.fifo_depth, "Weight")
        self.fifo_mp = FIFOBuffer(self.config.fifo_depth, "Position")
        self.pseudo_acc = np.zeros(self.config.num_timesteps)

        # Pipeline state
        self.pipeline = {
            PipelineStage.FAST_PREFIX: None,  # Current position in this stage
            PipelineStage.PSEUDO_ACC: None,
            PipelineStage.LOADING: None,
            PipelineStage.CORRECTION: None
        }
        
        # Performance counters
        self.total_cycles = 0
        self.compute_cycles = 0
        
        # Control signals
        self.ready = False  # For pseudo accumulation
        self.gated = False  # For correction
        
        # Processing state
        self._matched_positions = []
        self._current_position_idx = 0
        
    def debug_print(self, message):
        if self.debug:
            print(message)

    def cycle(self) -> bool:
        """Execute one cycle of pipeline"""
        self.total_cycles += 1
        self.debug_print(f"\n=== Cycle {self.total_cycles} ===")

        # Initialize pipeline on first cycle
        if self.total_cycles == 1:
            matched = np.logical_and(self.bitmask_a, self.bitmask_b)
            self._matched_positions = np.where(matched)[0].tolist()
            if not self._matched_positions:
                return True
            self.pipeline[PipelineStage.FAST_PREFIX] = self._matched_positions[0]
            self._current_position_idx = 1

        # Pipeline stages execute in reverse order to prevent conflicts
        
        # Stage 4: Correction
        if self.pipeline[PipelineStage.CORRECTION] is not None:
            pos = self.pipeline[PipelineStage.CORRECTION]
            spike_pattern = self.fiber_a_data[pos]
            if not np.all(spike_pattern):
                weight = self.fiber_b_data[pos]
                # Apply correction where spikes are 0
                self.pseudo_acc[~spike_pattern] -= weight
            self.pipeline[PipelineStage.CORRECTION] = None

        # Stage 3: Loading
        if self.pipeline[PipelineStage.LOADING] is not None:
            self.pipeline[PipelineStage.CORRECTION] = self.pipeline[PipelineStage.LOADING]
            self.pipeline[PipelineStage.LOADING] = None

        # Stage 2: Pseudo Accumulation
        if self.pipeline[PipelineStage.PSEUDO_ACC] is not None:
            pos = self.pipeline[PipelineStage.PSEUDO_ACC]
            weight = self.fiber_b_data[pos]
            # Optimistically accumulate assuming all 1s
            self.pseudo_acc += weight
            self.compute_cycles += 1
            self.pipeline[PipelineStage.LOADING] = self.pipeline[PipelineStage.PSEUDO_ACC]
            self.pipeline[PipelineStage.PSEUDO_ACC] = None

        # Stage 1: Fast Prefix
        if self.pipeline[PipelineStage.FAST_PREFIX] is not None:
            self.pipeline[PipelineStage.PSEUDO_ACC] = self.pipeline[PipelineStage.FAST_PREFIX]
            self.pipeline[PipelineStage.FAST_PREFIX] = None
            
            # Load next position if available
            if self._current_position_idx < len(self._matched_positions):
                self.pipeline[PipelineStage.FAST_PREFIX] = self._matched_positions[self._current_position_idx]
                self._current_position_idx += 1

        # Check if pipeline is complete
        pipeline_active = any(stage is not None for stage in self.pipeline.values())
        if not pipeline_active and self._current_position_idx >= len(self._matched_positions):
            return True

        return False

    def start_processing(self, bitmask_a: np.ndarray, bitmask_b: np.ndarray, 
                        fiber_a_data: np.ndarray, fiber_b_data: np.ndarray):
        """Start processing new input data"""
        self.reset()
        self.bitmask_a = bitmask_a
        self.bitmask_b = bitmask_b
        self.fiber_a_data = fiber_a_data
        self.fiber_b_data = fiber_b_data

    def get_results(self) -> Tuple[np.ndarray, Dict]:
        """Get final results and statistics"""
        stats = {
            'total_cycles': self.total_cycles,
            'compute_cycles': self.compute_cycles,
        }
        return self.pseudo_acc, stats

def test_tppe():
    config = TPPEConfig()
    tppe = TPPE(config, debug=True)
    
    # Test data from paper example
    bitmask_a = np.array([1, 0, 1, 1, 0] + [0] * 123, dtype=bool)
    bitmask_b = np.array([1, 1, 1, 0, 0] + [0] * 123, dtype=bool)
    fiber_a_data = np.array([
        [1, 1, 1, 1],  # Position 0: all active
        [0, 0, 0, 0],  # Position 1: not used
        [1, 0, 1, 0],  # Position 2: alternating
        [0, 1, 0, 1],  # Position 3: not used
        [0, 0, 0, 0],  # Position 4: not used
    ] + [[0, 0, 0, 0]] * 123)
    fiber_b_data = np.array([1, 2, 3, 0, 0] + [0] * 123)
    
    tppe.start_processing(bitmask_a, bitmask_b, fiber_a_data, fiber_b_data)
    
    while not tppe.cycle():
        print(f"Pipeline state:")
        for stage, pos in tppe.pipeline.items():
            if pos is not None:
                print(f"  {stage.name}: Position {pos}")
        print(f"Accumulator: {tppe.pseudo_acc}")
    
    results, stats = tppe.get_results()
    print("\nFinal Results:")
    print(f"Accumulator values: {results}")
    print(f"Expected values: [4, 1, 4, 1]")
    print("\nStatistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    test_tppe()