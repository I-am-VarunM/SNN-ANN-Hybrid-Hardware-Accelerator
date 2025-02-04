from dataclasses import dataclass 
from typing import List, Dict, Optional, Tuple
from enum import Enum
import numpy as np
from collections import deque
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from typing import List, Tuple
import torch.nn as nn

@dataclass
class TPPEConfig:
    num_timesteps: int = 4 
    bitmask_width: int = 128 
    weight_buffer_size: int = 128
    fifo_depth: int = 8
    v_threshold: float = 3.0  
    tau: float = 1.0  # Leaky factor (set to 1 to ignore leakage)
    energy_accumulator: float = 0.16    # Accumulator energy per operation
    energy_fast_prefix: float = 1.46    # Fast prefix sum circuit
    energy_laggy_prefix: float = 0.32   # Laggy prefix sum circuit
    energy_others: float = 0.88         # Other components

class PipelineStage(Enum):
    FAST_PREFIX = 0  
    PSEUDO_ACC = 1   
    LOADING = 2      
    CORRECTION = 3
    LIF = 4         # Added LIF stage
    DONE = 5

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
class EnergyStats:
    def __init__(self):
        self.accumulator_ops = 0      # Number of accumulator operations
        self.fast_prefix_ops = 0      # Number of fast prefix sum operations
        self.laggy_prefix_ops = 0     # Number of laggy prefix sum operations
        self.other_ops = 0            # Other operations
        
    def calculate_energy(self, config: TPPEConfig) -> Dict[str, float]:
        """Calculate energy consumption for each component"""
        return {
            'accumulator': self.accumulator_ops * config.energy_accumulator,
            'fast_prefix': self.fast_prefix_ops * config.energy_fast_prefix,
            'laggy_prefix': self.laggy_prefix_ops * config.energy_laggy_prefix,
            'others': self.other_ops * config.energy_others,
            'total': (self.accumulator_ops * config.energy_accumulator +
                     self.fast_prefix_ops * config.energy_fast_prefix +
                     self.laggy_prefix_ops * config.energy_laggy_prefix +
                     self.other_ops * config.energy_others)
        }

class TPPE:
    def __init__(self, config: TPPEConfig, debug=True):
        self.config = config
        self.debug = debug
        self.energy_stats = EnergyStats()
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
        self.pseudo_acc = np.zeros(self.config.num_timesteps)  # Membrane potential
        self.output_spikes = np.zeros(self.config.num_timesteps, dtype=bool)  # Output spikes
        
        # Pipeline state tracking
        self.pipeline = {
            PipelineStage.FAST_PREFIX: None,
            PipelineStage.PSEUDO_ACC: None,  
            PipelineStage.LOADING: None,
            PipelineStage.CORRECTION: None,
            PipelineStage.LIF: None
        }
        
        # Performance counters
        self.total_cycles = 0
        self.compute_cycles = 0
        self.pipeline_cycles = 0
        # Control signals
        self.ready = False  
        self.gated = False

        # Processing state
        self._matched_positions = []
        self._current_position_idx = 0
        self.membrane_potential = np.zeros(self.config.num_timesteps)  # U[t]
        self.pseudo_acc = np.zeros(self.config.num_timesteps)  # X[t]
        self.output_spikes = np.zeros(self.config.num_timesteps, dtype=bool)  # C[t]
        
    def debug_print(self, message, indent=0):
        if self.debug:
            print("  " * indent + message)

    def print_pipeline_state(self):
        """Print detailed pipeline state information"""
        self.debug_print("\nPipeline Status:")
        self.debug_print("================")
        # Print current stage occupancy
        for stage in PipelineStage:
            if stage != PipelineStage.DONE:
                pos = self.pipeline[stage]
                if pos is not None:
                    self.debug_print(f"{stage.name:12}: Position {pos}")
                    if stage == PipelineStage.PSEUDO_ACC:
                        self.debug_print(f"            Weight: {self.fiber_b_data[pos]}")
                        self.debug_print(f"            Pattern: {self.fiber_a_data[pos].astype(int)}")
                else:
                    self.debug_print(f"{stage.name:12}: Empty")

        self.debug_print("\nNeuron State:")
        self.debug_print(f"Membrane Potential: {self.pseudo_acc}")
        self.debug_print(f"Output Spikes: {self.output_spikes.astype(int)}")

        # Print FIFO states
        self.debug_print(f"\nFIFO Status:")
        self.debug_print(f"Weight FIFO: {list(self.fifo_b.buffer)}")
        self.debug_print(f"Position FIFO: {list(self.fifo_mp.buffer)}")
        
        # Print next positions to be processed
        if self._current_position_idx < len(self._matched_positions):
            remaining = self._matched_positions[self._current_position_idx:]
            self.debug_print(f"\nRemaining positions to process: {remaining}")
        else:
            self.debug_print("\nNo remaining positions")
        self.debug_print("================\n")

    def apply_lif(self):
        """Apply LIF neuron dynamics"""
        if self.pipeline[PipelineStage.LIF] is not None:
            self.debug_print("LIF Stage:", indent=1)
            self.debug_print(f"  Input Current (X[t]): {self.pseudo_acc}", indent=1)
            self.debug_print(f"  Previous Membrane Potential (U[t-1]): {self.membrane_potential}", indent=1)
            
            for t in range(self.config.num_timesteps):
                # Get membrane potential from previous timestep (U[t-1])
                prev_potential = self.membrane_potential[t-1] if t > 0 else 0
                
                # Add current input to previous membrane potential
                total_potential = self.pseudo_acc[t] + prev_potential
                
                # Generate spike if threshold is exceeded
                self.output_spikes[t] = total_potential > self.config.v_threshold
                
                # Update membrane potential
                if self.output_spikes[t]:
                    # Reset if spike occurs
                    self.membrane_potential[t] = 0
                else:
                    # Keep accumulating if no spike
                    self.membrane_potential[t] = self.config.tau * total_potential
            
            self.debug_print(f"  Output Spikes: {self.output_spikes.astype(int)}", indent=1)
            self.debug_print(f"  Updated Membrane Potential (U[t]): {self.membrane_potential}", indent=1)
            
            self.pipeline[PipelineStage.LIF] = None
            
    def cycle(self) -> bool:
        """Execute one cycle of pipeline"""
        self.total_cycles += 1
        self.debug_print(f"\n=== Cycle {self.total_cycles} ===")
        
        # Print initial state
        self.print_pipeline_state()
        
        # Initialize pipeline on first cycle
        if self.total_cycles == 1:
            # Always count one "other" operation for bitmask AND
            self.energy_stats.other_ops += 1
            
            matched = np.logical_and(self.bitmask_a, self.bitmask_b)
            self._matched_positions = np.where(matched)[0].tolist()
            self.debug_print(f"\nMatched positions: {self._matched_positions}")
            if not self._matched_positions:
                return True
                
            self.debug_print("\nInitializing pipeline:")
            self.pipeline[PipelineStage.PSEUDO_ACC] = self._matched_positions[0]
            self.debug_print(f"  Loading position {self._matched_positions[0]} into PSEUDO_ACC")
            
            if len(self._matched_positions) > 1:
                self.pipeline[PipelineStage.FAST_PREFIX] = self._matched_positions[1]
                self.debug_print(f"  Loading position {self._matched_positions[1]} into FAST_PREFIX")
                self._current_position_idx = 2
            else:
                self._current_position_idx = 1
                
        self.debug_print("\nExecuting Pipeline Stages:")
        
        # Execute stages in reverse order to prevent conflicts
        
        # Stage 4: Correction
        if self.pipeline[PipelineStage.CORRECTION] is not None:
            pos = self.pipeline[PipelineStage.CORRECTION]
            spike_pattern = self.fiber_a_data[pos]
            weight = self.fiber_b_data[pos]
            self.debug_print(f"Correction Stage - Position {pos}:", indent=1)
            self.debug_print(f"  Spike Pattern: {spike_pattern.astype(int)}", indent=1)
            self.debug_print(f"  Weight: {weight}", indent=1)
            self.debug_print(f"  Before Correction: {self.pseudo_acc}", indent=1)

            # Only apply correction if there are any 0s in the spike pattern
            if not np.all(spike_pattern):
                # Subtract weight only from positions where spike is 0
                inactive_timesteps = np.logical_not(spike_pattern)
                self.debug_print(f"  Spike pattern: {spike_pattern}")
                self.debug_print(f"  Inactive timesteps: {inactive_timesteps}")
                self.pseudo_acc[inactive_timesteps] -= weight
                self.debug_print(f"  After Correction: {self.pseudo_acc}", indent=1)
            else:
                self.debug_print(f"  No correction needed (all spikes are 1)", indent=1)

            self.pipeline[PipelineStage.CORRECTION] = None
            
        # Stage 3: Loading 
        if self.pipeline[PipelineStage.LOADING] is not None:
            pos = self.pipeline[PipelineStage.LOADING]
            self.debug_print(f"Loading Stage - Moving Position {pos} to Correction", indent=1)
            self.pipeline[PipelineStage.CORRECTION] = pos
            self.pipeline[PipelineStage.LOADING] = None
            
        # Stage 2: Pseudo Accumulation
        if self.pipeline[PipelineStage.PSEUDO_ACC] is not None:
            pos = self.pipeline[PipelineStage.PSEUDO_ACC]
            weight = self.fiber_b_data[pos]
            self.debug_print(f"Pseudo Accumulation Stage - Position {pos}:", indent=1)
            self.debug_print(f"  Weight to Add: {weight}", indent=1)
            self.debug_print(f"  Before Accumulation: {self.pseudo_acc}", indent=1)
            
            # Accumulate assuming all 1s
            self.pseudo_acc += weight
            self.debug_print(f"  After Accumulation: {self.pseudo_acc}", indent=1)
            
            self.compute_cycles += 1
            self.pipeline[PipelineStage.LOADING] = pos
            self.pipeline[PipelineStage.PSEUDO_ACC] = None
            
        # Stage 1: Fast Prefix Sum
        if self.pipeline[PipelineStage.FAST_PREFIX] is not None:
            pos = self.pipeline[PipelineStage.FAST_PREFIX]
            self.debug_print(f"Fast Prefix Stage - Position {pos}:", indent=1)
            self.debug_print(f"  Moving to Pseudo Accumulation", indent=1)
            
            self.pipeline[PipelineStage.PSEUDO_ACC] = pos
            self.pipeline[PipelineStage.FAST_PREFIX] = None
            
            # Load next position if available
            if self._current_position_idx < len(self._matched_positions):
                next_pos = self._matched_positions[self._current_position_idx]
                self.debug_print(f"  Loading next position: {next_pos}", indent=1)
                self.pipeline[PipelineStage.FAST_PREFIX] = next_pos
                self._current_position_idx += 1

        # Move to LIF stage when other stages are complete
        all_positions_done = self._current_position_idx >= len(self._matched_positions)
        no_active_stages = not any(self.pipeline[stage] is not None for stage in 
            [PipelineStage.FAST_PREFIX, PipelineStage.PSEUDO_ACC, PipelineStage.LOADING, PipelineStage.CORRECTION])
            
        if all_positions_done and no_active_stages and self.pipeline[PipelineStage.LIF] is None:
            self.pipeline[PipelineStage.LIF] = True

        # Apply LIF after other stages
        self.apply_lif()

        # Count energy for pipeline operations
        pipeline_active = False

        if self.pipeline[PipelineStage.PSEUDO_ACC] is not None:
            self.energy_stats.accumulator_ops += 1
            pipeline_active = True
            
        if self.pipeline[PipelineStage.FAST_PREFIX] is not None:
            self.energy_stats.fast_prefix_ops += 1
            pipeline_active = True
            
        if self.pipeline[PipelineStage.CORRECTION] is not None:
            self.energy_stats.laggy_prefix_ops += 1
            pipeline_active = True

        if pipeline_active:
            self.pipeline_cycles += 1
            self.energy_stats.other_ops += 1
                
        # Print state after execution
        self.debug_print("\nPipeline state after execution:")
        self.print_pipeline_state()
                
        # Check if pipeline is complete        
        pipeline_active = any(stage is not None for stage in self.pipeline.values())
        if not pipeline_active and self._current_position_idx >= len(self._matched_positions):
            return True
            
        return False
    def get_results(self) -> Tuple[np.ndarray, Dict]:
        """Get final results with performance and energy statistics"""
        energy_consumption = self.energy_stats.calculate_energy(self.config)
        
        stats = {
            'total_cycles': self.total_cycles,
            'compute_cycles': self.compute_cycles,
            'pipeline_cycles': self.pipeline_cycles,
            'energy_stats': energy_consumption
        }
        return self.output_spikes, stats   
    def start_processing(self, bitmask_a: np.ndarray, bitmask_b: np.ndarray,
                        fiber_a_data: np.ndarray, fiber_b_data: np.ndarray):
        """Start processing new input data"""
        self.reset()
        self.bitmask_a = bitmask_a  
        self.bitmask_b = bitmask_b
        self.fiber_a_data = fiber_a_data
        self.fiber_b_data = fiber_b_data
def print_performance_stats(stats: Dict):
    """Print detailed performance and energy statistics"""
    print("\nPerformance Statistics:")
    print(f"Total Cycles: {stats['total_cycles']}")
    print(f"Compute Cycles: {stats['compute_cycles']}")
    print(f"Pipeline Active Cycles: {stats['pipeline_cycles']}")
    
    print("\nEnergy Consumption (pJ):")
    print(f"Accumulator: {stats['energy_stats']['accumulator']:.2f}")
    print(f"Fast Prefix: {stats['energy_stats']['fast_prefix']:.2f}")
    print(f"Laggy Prefix: {stats['energy_stats']['laggy_prefix']:.2f}")
    print(f"Others: {stats['energy_stats']['others']:.2f}")
    print(f"Total: {stats['energy_stats']['total']:.2f}")

def test_basic():
    """Basic functionality test"""
    config = TPPEConfig(v_threshold=3.0, tau=1.0)
    tppe = TPPE(config)
    
    # Test data from paper example
    bitmask_a = np.array([1,0,1,1,0] + [0]*123, dtype=bool) 
    bitmask_b = np.array([1,1,1,0,0] + [0]*123, dtype=bool)
    
    fiber_a_data = np.array([
        [1,1,1,1],  # Position 0: all active
        [0,0,0,0],  # Position 1: unused
        [1,0,1,0],  # Position 2: alternating
        [0,1,0,1],  # Position 3: unused
        [0,0,0,0]   # Position 4: unused
    ] + [[0,0,0,0]]*123)
    
    fiber_b_data = np.array([1,2,3,0,0] + [0]*123)
    
    print("\nStarting basic test with configuration:")
    print(f"Bitmask A: {bitmask_a[:5].astype(int)}")
    print(f"Bitmask B: {bitmask_b[:5].astype(int)}")
    print(f"Threshold: {config.v_threshold}")
    print("\nSpike Patterns:")
    for i in range(5):
        print(f"Position {i}: {fiber_a_data[i].astype(int)}")
    print("\nWeights:", fiber_b_data[:5])
    
    tppe.start_processing(bitmask_a, bitmask_b, fiber_a_data, fiber_b_data)
    
    while not tppe.cycle():
        pass
    
    results, stats = tppe.get_results()
    print("\nFinal Results:")
    print(f"Output Spikes: {results.astype(int)}")
    print(f"Final Membrane Potential: {tppe.membrane_potential}")
    print(f"Expected Spikes: [1, 0, 1, 0]")
    print(f"Expected Membrane Potential: [0, 1, 0, 1]")
    print("\nStatistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    print_performance_stats(stats)  # Added performance statistics print

def test_corner_cases():
    """Tests edge cases with LIF behavior"""
    config = TPPEConfig(v_threshold=3.0, tau=1.0)
    tppe = TPPE(config)
    
    # Case 1: All sparse (no spikes expected)
    print("\nCase 1: All Sparse")
    bitmask_a = np.zeros(128, dtype=bool)
    bitmask_b = np.zeros(128, dtype=bool)
    fiber_a = np.zeros((128,4))
    fiber_b = np.zeros(128)
    
    tppe.start_processing(bitmask_a, bitmask_b, fiber_a, fiber_b)
    while not tppe.cycle():
        pass
    results, stats = tppe.get_results()
    print_performance_stats(stats)  # Added performance statistics print
    assert np.all(results == 0), "All-zero case failed"
    assert np.all(tppe.membrane_potential == 0), "All-zero membrane potential failed"
    print("Case 1 passed")
    
    # Case 2: High temporal sparsity
    print("\nCase 2: High Temporal Sparsity")
    bitmask_a = np.array([1,1,0,0] + [0]*124, dtype=bool)
    bitmask_b = np.array([1,1,0,0] + [0]*124, dtype=bool)
    fiber_a = np.zeros((128,4))
    fiber_a[0] = [1,0,0,0]  # Only active in first timestep
    fiber_a[1] = [0,0,0,1]  # Only active in last timestep
    fiber_b = np.array([2,3] + [0]*126)  # Weights

    print("Configuration:")
    print(f"Bitmask A: {bitmask_a[:4].astype(int)}")
    print(f"Bitmask B: {bitmask_b[:4].astype(int)}")
    print(f"Position 0 pattern: {fiber_a[0].astype(int)}")
    print(f"Position 1 pattern: {fiber_a[1].astype(int)}")
    print(f"Weights: {fiber_b[:2]}")
    
    tppe.start_processing(bitmask_a, bitmask_b, fiber_a, fiber_b)
    while not tppe.cycle():
        pass
    results, stats = tppe.get_results()
    print_performance_stats(stats)  # Added performance statistics print
    # Calculations for expected outputs:
    # t0: 2 > threshold (spike=0, U[0]=2)
    # t1: 2 < threshold (spike=0, U[1]=2)
    # t2: 2 < threshold (spike=0, U[2]=2)
    # t3: 5 > threshold (spike=1, U[3]=0)
    expected_spikes = np.array([0,0,0,1])
    expected_membrane = np.array([2,2,2,0])
    
    print(f"\nOutput Spikes: {results.astype(int)}")
    print(f"Expected Spikes: {expected_spikes}")
    print(f"Membrane Potential: {tppe.membrane_potential}")
    print(f"Expected Membrane: {expected_membrane}")
    assert np.all(results == expected_spikes), "High temporal sparsity case failed"
    assert np.allclose(tppe.membrane_potential, expected_membrane), "Membrane potential mismatch"
    print("Case 2 passed")


class ActivationHook:
    def __init__(self):
        self.activations = {}
        
    def __call__(self, module, input_tensor, output_tensor):
        self.activations[module] = output_tensor.detach()

def convert_to_spikes(activation: torch.Tensor, threshold: float = 0.5, timesteps: int = 4) -> np.ndarray:
    """Convert activation values to spike trains
    Args:
        activation: Input tensor of shape [M, N]
        threshold: Base threshold for spike generation
        timesteps: Number of timesteps
    Returns:
        Spike tensor of shape [M, N, timesteps]
    """
    # Normalize activations to [0,1]
    act_min = activation.min()
    act_max = activation.max()
    if act_max > act_min:
        normalized = (activation - act_min) / (act_max - act_min)
    else:
        normalized = torch.zeros_like(activation)
    
    # Convert to spike trains across timesteps
    spikes = torch.zeros((*activation.shape, timesteps))
    for t in range(timesteps):
        spikes[..., t] = (normalized > (threshold * (t + 1) / timesteps)).float()
    
    return spikes.numpy()

def load_vgg16(image_path: str = None, timesteps: int = 4, threshold: float = 0.5) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """Load pretrained VGG16 and get activations for a single image
    
    Args:
        image_path: Path to input image (if None, will use random input)
        timesteps: Number of timesteps for spike encoding
        threshold: Threshold for spike conversion
    
    Returns:
        weights: List of weight matrices per layer
        activations: List of spike matrices per layer
        reference: List of reference outputs per layer
    """
    # 1. Load pretrained VGG16
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    model = model.eval()

    # 2. Create single random input with correct dimensions
    # VGG16 expects (batch_size, channels, height, width)
    input_tensor = torch.randn(1, 3, 224, 224)  # Single random image
    print("\nInput shape:", input_tensor.shape)

    # 3. Register hooks to capture activations
    activation_hook = ActivationHook()
    hooks = []
    conv_layers = []
    
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(activation_hook))
            conv_layers.append(module)
    
    # 4. Forward pass to get activations
    with torch.no_grad():
        output = model(input_tensor)
    
    # 5. Process weights and activations
    weights = []
    activations = []
    reference = []
    
    for i, layer in enumerate(conv_layers):
        print(f"\nProcessing Layer {i+1}:")
        
        # Get weights
        w = layer.weight.data.cpu().numpy()
        print(f"Original weight shape: {w.shape}")  # [out_channels, in_channels, kernel_h, kernel_w]
        
        # Reshape weights to [in_channels*kernel_h*kernel_w, out_channels]
        w_reshaped = w.reshape(w.shape[0], -1).T
        print(f"Reshaped weight: {w_reshaped.shape}")
        weights.append(w_reshaped)
        
        # Get activations
        act = activation_hook.activations[layer].cpu()
        print(f"Original activation shape: {act.shape}")  # [1, out_channels, height, width]
        
        # Reshape activations to [1, out_channels, height*width]
        N = act.shape[2] * act.shape[3]  # height * width
        act_reshaped = act.reshape(act.shape[0], act.shape[1], -1)
        print(f"Reshaped activation before spikes: {act_reshaped.shape}")
        
        # Convert to spikes [1, out_channels, height*width, timesteps]
        spikes = convert_to_spikes(act_reshaped.squeeze(0), threshold, timesteps)  # Remove batch dimension for spike conversion
        spikes = np.expand_dims(spikes, axis=0)  # Add batch dimension back
        print(f"Spike shape: {spikes.shape}")
        activations.append(spikes)
        
        # Store reference output [1, out_channels, height*width]
        reference.append(act_reshaped.numpy())
        print(f"Reference shape: {reference[-1].shape}")
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
        
    print("\nLayer Summary:")
    print("=============")
    for i, (w, a, r) in enumerate(zip(weights, activations, reference)):
        print(f"\nLayer {i+1}:")
        print(f"  Weight Matrix:  {w.shape}")
        print(f"  Activation Matrix: {a.shape}")
        print(f"  Reference Matrix: {r.shape}")
        print(f"  Sparsity:")
        print(f"    Weight: {100 * (1 - np.count_nonzero(w)/w.size):.2f}%")
        print(f"    Activation: {100 * (1 - np.count_nonzero(r)/r.size):.2f}%")
    
    return weights, activations, reference


class LayerStats:
    def __init__(self, layer_name: str):
        self.layer_name = layer_name
        self.effective_macs = 0  # Actual MACs performed by LoAS
        self.total_macs = 0      # Total MACs if dense
        self.energy_stats = None  # Energy consumption
        self.sparsity = {
            'weight': 0.0,       # Weight sparsity
            'activation': 0.0    # Activation sparsity
        }
        self.functional_check = True  # Verification flag

def analyze_vgg16_layer(layer_name: str, 
                       weights: np.ndarray, 
                       activations: np.ndarray, 
                       reference_output: np.ndarray,
                       config: TPPEConfig) -> LayerStats:
    """Analyze single VGG16 layer using LoAS"""
    stats = LayerStats(layer_name)
    
    print("\n" + "="*50)
    print(f"Detailed Analysis of {layer_name}")
    print("="*50)
    
    # Input Analysis
    print("\nInput Analysis:")
    print(f"Weights shape: {weights.shape}")
    print(f"- Number of output filters: {weights.shape[0]}")
    print(f"- Input dimension (in_channels * kernel_size^2): {weights.shape[1]}")
    
    print(f"\nActivations shape: {activations.shape}")
    print(f"- Number of output channels: {activations.shape[0]}")
    print(f"- Feature map size (height*width): {activations.shape[1]}")
    print(f"- Timesteps: {activations.shape[2]}")
    
    print(f"\nReference output shape: {reference_output.shape}")
    
    # Shape Analysis for Matrix Multiplication
    print("\nMatrix Multiplication Analysis:")
    print("For matmul C = AB:")
    print(f"A (activations) shape: ({activations.shape[0]}, {activations.shape[1]})")
    print(f"B (weights) shape: ({weights.shape[0]}, {weights.shape[1]})")
    print("Required shape transformation for proper matmul:")
    print(f"- Need to transpose weights from ({weights.shape[0]}, {weights.shape[1]}) to ({weights.shape[1]}, {weights.shape[0]})")
    
    # Transpose weights for proper multiplication
    weights_t = weights.T
    print(f"\nTransposed weights shape: {weights_t.shape}")

    # Calculate theoretical MACs
    stats.total_macs = activations.shape[0] * activations.shape[1] * weights.shape[1]
    
    # Process each timestep
    for t in range(activations.shape[2]):
        print(f"\nProcessing Timestep {t}:")
        
        # Get activations for current timestep
        act_t = activations[:, :, t]
        print(f"- Activation slice shape: {act_t.shape}")
        
        # Create bitmasks
        bitmask_a = (act_t != 0)
        bitmask_b = (weights_t != 0)
        print("\nBitmask Analysis:")
        print(f"bitmask_a shape: {bitmask_a.shape}")
        print(f"bitmask_b shape: {bitmask_b.shape}")
        print(f"bitmask_a sparsity: {100*(1-np.count_nonzero(bitmask_a)/bitmask_a.size):.2f}%")
        print(f"bitmask_b sparsity: {100*(1-np.count_nonzero(bitmask_b)/bitmask_b.size):.2f}%")
        
        # Track sparsity (only for first timestep)
        if t == 0:
            stats.sparsity['activation'] = 1.0 - np.count_nonzero(bitmask_a)/bitmask_a.size
            stats.sparsity['weight'] = 1.0 - np.count_nonzero(bitmask_b)/bitmask_b.size
        
        # Shape analysis before LoAS
        print("\nPre-LoAS Shape Check:")
        print(f"act_t shape for matmul: {act_t.shape}")
        print(f"weights_t shape for matmul: {weights_t.shape}")
        
        # Run LoAS
        try:
            tppe = TPPE(config)
            tppe.start_processing(bitmask_a, bitmask_b, act_t, weights_t)
            
            while not tppe.cycle():
                pass
                
            results, run_stats = tppe.get_results()
            print("\nLoAS Processing Successful:")
            print(f"- Results shape: {results.shape}")

            # Verify results
            if not np.allclose(results, reference_output, rtol=1e-3):
                stats.functional_check = False
                print(f"Functional check failed at timestep {t}")
            
            # Accumulate stats
            if t == 0:
                stats.energy_stats = run_stats['energy_stats']
                stats.effective_macs = run_stats['compute_cycles']
            else:
                for key in stats.energy_stats:
                    stats.energy_stats[key] += run_stats['energy_stats'][key]
                stats.effective_macs += run_stats['compute_cycles']
            
        except ValueError as e:
            print("\nERROR in LoAS Processing:")
            print(f"Shape mismatch during processing: {str(e)}")
            print("Required shapes for logical_and:")
            print(f"- bitmask_a: {bitmask_a.shape}")
            print(f"- bitmask_b: {bitmask_b.shape}")
            raise
            
    return stats

def analyze_vgg16(output_path: str = "vgg16_analysis.txt", image_path: str = None):
    """Analyze VGG16 model using LoAS"""
    print("Loading VGG16 pretrained on ImageNet...")
    weights, activations, reference = load_vgg16(image_path)
    
    config = TPPEConfig(v_threshold=3.0)
    layer_stats = []
    
    # Analyze each layer
    for i, (w, a, ref) in enumerate(zip(weights, activations, reference)):
        layer_name = f"Conv_Layer_{i+1}"
        print(f"\nAnalyzing {layer_name}...")
        
        stats = analyze_vgg16_layer(layer_name, w, a, ref, config)
        
        if not stats.functional_check:
            print(f"ERROR: Functional check failed for {layer_name}")
            quit()
            return
            
        layer_stats.append(stats)
    
    # Generate report
    with open(output_path, 'w') as f:
        f.write("VGG16 Analysis Report using LoAS\n")
        f.write("===============================\n\n")
        
        total_effective_macs = 0
        total_theoretical_macs = 0
        total_energy = 0
        
        for stats in layer_stats:
            f.write(f"\nLayer: {stats.layer_name}\n")
            f.write("-" * 20 + "\n")
            
            # Computation efficiency
            f.write(f"Theoretical MACs: {stats.total_macs:,}\n")
            f.write(f"Effective MACs: {stats.effective_macs:,}\n")
            f.write(f"Computation saved: {100*(1-stats.effective_macs/stats.total_macs):.2f}%\n")
            
            # Sparsity
            f.write(f"Weight Sparsity: {100*stats.sparsity['weight']:.2f}%\n")
            f.write(f"Activation Sparsity: {100*stats.sparsity['activation']:.2f}%\n")
            
            # Energy
            f.write("\nEnergy Consumption (pJ):\n")
            f.write(f"  Accumulator: {stats.energy_stats['accumulator']:.2f}\n")
            f.write(f"  Fast Prefix: {stats.energy_stats['fast_prefix']:.2f}\n")
            f.write(f"  Laggy Prefix: {stats.energy_stats['laggy_prefix']:.2f}\n")
            f.write(f"  Others: {stats.energy_stats['others']:.2f}\n")
            f.write(f"  Total: {stats.energy_stats['total']:.2f}\n")
            
            total_effective_macs += stats.effective_macs
            total_theoretical_macs += stats.total_macs
            total_energy += stats.energy_stats['total']
        
        # Overall statistics
        f.write("\nOverall Statistics\n")
        f.write("=================\n")
        f.write(f"Total Theoretical MACs: {total_theoretical_macs:,}\n")
        f.write(f"Total Effective MACs: {total_effective_macs:,}\n")
        f.write(f"Overall Computation Saved: {100*(1-total_effective_macs/total_theoretical_macs):.2f}%\n")
        f.write(f"Total Energy Consumption: {total_energy:.2f} pJ\n")

    print(f"\nAnalysis complete. Results written to {output_path}")

def load_vgg16(image_path: str = None, timesteps: int = 4, threshold: float = 0.5) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Load pretrained VGG16 and get activations for a sample image
    
    Args:
        image_path: Path to input image (if None, will use random input)
        timesteps: Number of timesteps for spike encoding
        threshold: Threshold for spike conversion
    """
    # 1. Load pretrained VGG16
    model = models.vgg16(pretrained=True)
    model = model.eval()

    # 2. Prepare input data
    if image_path:
        # Load and preprocess real image
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        img = Image.open(image_path)
        input_tensor = transform(img).unsqueeze(0)
    else:
        # Use random input for testing
        input_tensor = torch.randn(1, 3, 224, 224)

    # 3. Register hooks to capture activations
    activation_hook = ActivationHook()
    hooks = []
    conv_layers = []
    
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            hooks.append(module.register_forward_hook(activation_hook))
            conv_layers.append(module)
    
    # 4. Forward pass to get activations
    with torch.no_grad():
        output = model(input_tensor)
    
    # 5. Process weights and activations
    weights = []
    activations = []
    reference = []
    
    for layer in conv_layers:
        # Get weights
        w = layer.weight.data.cpu().numpy()
        w_reshaped = w.reshape(w.shape[0], -1)  # Reshape to 2D
        weights.append(w_reshaped)
        
        # Get activations
        act = activation_hook.activations[layer]
        act = act.cpu()
        # Reshape activation: [batch, channels, height, width] -> [batch * channels, height * width]
        act_reshaped = act.reshape(act.shape[0] * act.shape[1], -1)
        
        # Convert to spikes
        spikes = convert_to_spikes(act_reshaped, threshold, timesteps)
        activations.append(spikes)
        
        # Store reference output
        reference.append(act_reshaped.numpy())
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
        
    print("\nLayer Information:")
    print("=================")
    for i, (w, a, r) in enumerate(zip(weights, activations, reference)):
        print(f"\nLayer {i+1}:")
        print(f"  Weight Matrix:  {w.shape}")
        print(f"  Activation Matrix: {a.shape}")
        print(f"  Sparsity:")
        print(f"    Weight: {100 * (1 - np.count_nonzero(w)/w.size):.2f}%")
        print(f"    Activation: {100 * (1 - np.count_nonzero(r)/r.size):.2f}%")
    
    return weights, activations, reference

if __name__ == "__main__":
    # Run complete analysis
    analyze_vgg16()

# if __name__ == "__main__":
#     print("Running basic test...")
#     test_basic()
#     print("\nRunning corner case tests...")
#     test_corner_cases()