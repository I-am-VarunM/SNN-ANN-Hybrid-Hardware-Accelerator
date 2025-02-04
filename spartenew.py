import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from torch.hub import load_state_dict_from_url
import torch
import torch.nn as nn
import site
import sys

@dataclass
class PowerSpecs:
    """Power specifications from Table 4 in paper (45nm)"""
    CLOCK_FREQ = 800e6  # 800 MHz
    CLOCK_PERIOD = 1/CLOCK_FREQ  # seconds
    
    # Power values in Watts (converted from mW)
    BUFFER_POWER = 19.2e-3
    PREFIX_SUM_POWER = 48.0e-3
    PRIORITY_ENCODER_POWER = 6.4e-3
    MAC_POWER = 13.82e-3
    PERMUTE_POWER = 10.6e-3
    OTHER_POWER = 20.28e-3

@dataclass
class SparseData:
    sparsemap: np.ndarray
    values: np.ndarray
    original_shape: Tuple[int, ...]

    @classmethod
    def from_dense(cls, data: np.ndarray):
        sparsemap = (data != 0).astype(np.uint8)
        values = data[data != 0]
        return cls(sparsemap, values, data.shape)
    
    def get_values_at_positions(self, positions: np.ndarray) -> np.ndarray:
        value_indices = np.zeros_like(positions)
        for i, pos in enumerate(positions):
            value_indices[i] = np.sum(self.sparsemap[:pos])
        return self.values[value_indices]

@dataclass
class MACUnit:
    def __init__(self):
        self.accumulator = 0.0
        self.debug_multiplications = []
        
    def multiply_accumulate(self, a: float, b: float):
        self.debug_multiplications.append((a, b))
        self.accumulator += a * b
        
    def get_result(self):
        return self.accumulator
        
    def get_debug_info(self):
        return self.debug_multiplications
        
    def reset(self):
        self.accumulator = 0.0
        self.debug_multiplications = []

@dataclass
class EnergyStats:
    def __init__(self):
        self.cycles = {
            'and': 0,
            'priority_encode': 0,
            'prefix_sum': 0,
            'memory': 0,
            'mac': 0,
            'buffer': 0,
        }
        
    def compute_energy(self):
        """Compute energy in picoJoules"""
        energy = {
            'and': self.cycles['and'] * PowerSpecs.OTHER_POWER * PowerSpecs.CLOCK_PERIOD,
            'priority_encode': self.cycles['priority_encode'] * PowerSpecs.PRIORITY_ENCODER_POWER * PowerSpecs.CLOCK_PERIOD,
            'prefix_sum': self.cycles['prefix_sum'] * PowerSpecs.PREFIX_SUM_POWER * PowerSpecs.CLOCK_PERIOD,
            'memory': self.cycles['memory'] * PowerSpecs.OTHER_POWER * PowerSpecs.CLOCK_PERIOD,
            'mac': self.cycles['mac'] * PowerSpecs.MAC_POWER * PowerSpecs.CLOCK_PERIOD,
            'buffer': self.cycles['buffer'] * PowerSpecs.BUFFER_POWER * PowerSpecs.CLOCK_PERIOD,
        }
        return {k: v * 1e12 for k, v in energy.items()}  # Convert to picoJoules

@dataclass
class ComputeUnit:
    id: int
    assigned_filter_idx: int = None
    
    def __post_init__(self):
        self.mac_unit = MACUnit()
        self.filter_buffer = None
        self.energy_stats = EnergyStats()
    
    def assign_filter(self, filter_idx: int):
        """Assign a filter to this compute unit"""
        self.assigned_filter_idx = filter_idx
        print(f"CU{self.id} assigned Filter {filter_idx}")
    
    def process_sparse_multiply(self, input_chunk: SparseData) -> Tuple[int, int, Dict]:
        if self.filter_buffer is None:
            return 0, 0, self.energy_stats
            
        # Buffer active during entire operation
        self.energy_stats.cycles['buffer'] += 1
        
        # AND operation
        matches = self.filter_buffer.sparsemap & input_chunk.sparsemap
        match_positions = np.where(matches)[0]
        self.energy_stats.cycles['and'] += 1
        total_macs = len(match_positions)
        dense_macs = len(self.filter_buffer.sparsemap)  # Total possible MACs
        
        print(f"\nCU{self.id} (Filter {self.assigned_filter_idx}) processing:")
        print(f"  Found {total_macs} matches out of {dense_macs} possible")
        
        for pos_idx, pos in enumerate(match_positions):
            # Priority encode and prefix sum
            self.energy_stats.cycles['priority_encode'] += 1
            self.energy_stats.cycles['prefix_sum'] += 1
            
            # Memory access
            filter_offset = np.sum(self.filter_buffer.sparsemap[:pos])
            input_offset = np.sum(input_chunk.sparsemap[:pos])
            self.energy_stats.cycles['memory'] += 1
            
            # MAC operation
            filter_val = self.filter_buffer.values[filter_offset]
            input_val = input_chunk.values[input_offset]
            self.mac_unit.multiply_accumulate(filter_val, input_val)
            self.energy_stats.cycles['mac'] += 1
            
        energy = self.energy_stats.compute_energy()
        total_energy = sum(energy.values())
        
        print(f"  MAC efficiency: {(total_macs/dense_macs)*100:.2f}%")
        if((total_macs/dense_macs) == 1):
            print(self.filter_buffer.sparsemap)
            print(match_positions)
        print(f"  Energy consumed: {total_energy:.2f} pJ")
        
        return total_macs, dense_macs, self.energy_stats

class SparTenCluster:
    def __init__(self, num_compute_units: int = 32, chunk_size: int = 128):
        self.num_compute_units = num_compute_units
        self.chunk_size = chunk_size
        self.compute_units = [ComputeUnit(i) for i in range(num_compute_units)]
        self.current_cycle = 0
        
    def process_input(self, filters: np.ndarray, input_data: np.ndarray) -> Tuple[List[float], Dict]:
        num_filters = len(filters)
        
        print("\nAssigning filters to compute units:")
        for i in range(min(num_filters, self.num_compute_units)):
            self.compute_units[i].assign_filter(i)
        
        total_energy = 0
        total_macs = 0
        total_dense_macs = 0
        
        chunk_size = self.chunk_size
        num_chunks = (input_data.shape[0] + chunk_size - 1) // chunk_size
        
        print(f"\nProcessing {num_chunks} chunks")
        
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min((chunk_idx + 1) * chunk_size, input_data.shape[0])
            
            print(f"\nProcessing chunk {chunk_idx + 1}/{num_chunks}")
            input_chunk = SparseData.from_dense(input_data[chunk_start:chunk_end])
            
            for cu in self.compute_units:
                if cu.assigned_filter_idx is not None:
                    filter_chunk = filters[cu.assigned_filter_idx, chunk_start:chunk_end]
                    cu.filter_buffer = SparseData.from_dense(filter_chunk)
                    
                    macs, dense_macs, energy_stats = cu.process_sparse_multiply(input_chunk)
                    total_macs += macs
                    total_dense_macs += dense_macs
                    total_energy += sum(energy_stats.compute_energy().values())
        
        # Collect results
        results = []
        print("\nFinal Statistics:")
        print(f"Total MACs performed: {total_macs}")
        print(f"Total dense MACs possible: {total_dense_macs}")
        print(f"Overall MAC efficiency: {(total_macs/total_dense_macs)*100:.2f}%")
        print(f"Total energy consumed: {total_energy:.2f} pJ")
        
        for cu in self.compute_units:
            if cu.assigned_filter_idx is not None:
                results.append((cu.assigned_filter_idx, cu.mac_unit.get_result()))
        
        results.sort(key=lambda x: x[0])
        return [r[1] for r in results], {
            'total_energy': total_energy,
            'total_macs': total_macs,
            'dense_macs': total_dense_macs,
            'mac_efficiency': (total_macs/total_dense_macs)*100
        }
def verify_computation(filters: np.ndarray, inputs: np.ndarray, sparse_results: List[float]):
    dense_results = []
    for f, sparse_result in zip(filters, sparse_results):
        result = np.sum(f * inputs)
        dense_results.append(result)  # Include all results, not just non-zero
    
    print("Dense results:", dense_results)
    print("Sparse results:", sparse_results)
    
    matches = np.allclose(dense_results, sparse_results, rtol=1e-05, atol=1e-08)
    if matches:
        print("✅ Verification PASSED")
    else:
        print("❌ Verification FAILED")
        print("Differences:", np.array(dense_results) - np.array(sparse_results))
    
    return matches
def test_sparten():
    print("Running SparTen Tests...")
    
    # Test 1: Simple test with known sparsity
    print("\nTest 1: Simple sparse matrix")
    filters = np.array([
        [1, 0, 3, 0],  # 50% sparse
        [0, 2, 0, 4]   # 50% sparse
    ])
    inputs = np.array([1, 2, 0, 4])  # 25% sparse
    
    cluster = SparTenCluster(num_compute_units=2, chunk_size=4)
    results, stats = cluster.process_input(filters, inputs)
    verify_computation(filters, inputs, results)
    print("\nResults:", results)
    print("Statistics:", stats)
    
    # Test 2: High sparsity test
    print("\nTest 2: High sparsity test")
    filters = np.zeros((2, 8))
    filters[0, [0, 7]] = [1, 2]  # 75% sparse
    filters[1, [3, 4]] = [3, 4]  # 75% sparse
    
    inputs = np.zeros(8)
    inputs[[0, 4]] = [5, 6]  # 75% sparse
    
    cluster = SparTenCluster(num_compute_units=2, chunk_size=4)
    results, stats = cluster.process_input(filters, inputs)
    verify_computation(filters, inputs, results)
    print("\nResults:", results)
    print("Statistics:", stats)
    
    # Test 3: Multi-chunk processing
    print("\nTest 3: Multi-chunk processing")
    filters = np.zeros((4, 12))
    filters[0, [0, 4, 8]] = [1, 2, 3]
    filters[1, [1, 5, 9]] = [4, 5, 6]
    filters[2, [2, 6, 10]] = [7, 8, 9]
    filters[3, [3, 7, 11]] = [10, 11, 12]
    
    inputs = np.zeros(12)
    inputs[[0, 3, 6, 9]] = [13, 14, 15, 16]
    
    cluster = SparTenCluster(num_compute_units=4, chunk_size=4)
    results, stats = cluster.process_input(filters, inputs)
    verify_computation(filters, inputs, results)
    print("\nResults:", results)
    print("Statistics:", stats)

def verify_multi_round_processing(filters_2d: np.ndarray, input_patch: np.ndarray, 
                                sparse_results: np.ndarray, round_idx: int, 
                                filter_start: int, filter_end: int):
    """Verify results for each round of processing"""
    print(f"\nVerifying round {round_idx + 1}")
    print(f"Processing filters {filter_start} to {filter_end-1}")
    
    # Print shapes for debugging
    current_filters = filters_2d[filter_start:filter_end]
    print(f"Filter shape: {current_filters[0].shape}")
    print(f"Input patch shape: {input_patch.shape}")
    
    # Always reshape input patch regardless of shape match
    try:
        input_patch_reshaped = input_patch.reshape(current_filters[0].shape)
    except ValueError:
        print("Error: Cannot reshape input patch to match filter shape")
        print(f"Filter elements: {current_filters[0].size}")
        print(f"Input patch elements: {input_patch.size}")
        return False
        
    # Calculate expected results for this round
    dense_results = []
    for f in current_filters:
        try:
            result = np.sum(f * input_patch_reshaped)
            dense_results.append(result)
            
            # Print detailed multiplication info
            non_zero_pairs = np.count_nonzero(f * (input_patch_reshaped != 0))
            print(f"Non-zero multiplications: {non_zero_pairs} out of {len(f)}")
        except ValueError as e:
            print(f"Error in multiplication:")
            print(f"Filter shape: {f.shape}")
            print(f"Input shape: {input_patch_reshaped.shape}")
            return False
    
    # Compare with sparse results
    matches = np.allclose(dense_results, sparse_results, rtol=1e-05, atol=1e-08)
    
    print("\nResults comparison:")
    print("Dense results:", dense_results)
    print("Sparse results:", sparse_results)
    if matches:
        print("✅ Round verification PASSED")
    else:
        print("❌ Round verification FAILED")
        print("Differences:", np.array(dense_results) - np.array(sparse_results))
        exit()
    
    return matches

def save_layer_stats(stats: dict, filename: str = "vgg16_layer_stats.txt"):
    with open(filename, 'w') as f:
        f.write("VGG-16 Layer-wise Statistics\n")
        f.write("===========================\n\n")
        
        total_energy = sum(stat['total_energy'] for stat in stats.values())  # Changed key
        total_macs = sum(stat['total_macs'] for stat in stats.values())
        total_dense_macs = sum(stat['dense_macs'] for stat in stats.values())
        
        for layer_name, layer_stats in stats.items():
            f.write(f"Layer: {layer_name}\n")
            f.write("-" * 30 + "\n")
            f.write(f"MAC Efficiency: {layer_stats['mac_efficiency']:.2f}%\n")
            f.write(f"Energy Consumed: {layer_stats['total_energy']:.2f} pJ\n")  # Changed key
            f.write(f"MACs Performed: {layer_stats['total_macs']}\n")
            f.write(f"Dense MACs Possible: {layer_stats['dense_macs']}\n")
            f.write(f"Percentage of Total Energy: {(layer_stats['total_energy']/total_energy)*100:.2f}%\n")
            f.write(f"Percentage of Total MACs: {(layer_stats['total_macs']/total_macs)*100:.2f}%\n\n")
        
        f.write("\nOverall Statistics\n")
        f.write("=================\n")
        f.write(f"Total Energy: {total_energy:.2f} pJ\n")
        f.write(f"Total MAC Efficiency: {(total_macs/total_dense_macs)*100:.2f}%\n")
        f.write(f"Total MACs: {total_macs} / {total_dense_macs}\n")

def process_conv_layer(filters_4d: np.ndarray, input_feature_map: np.ndarray, 
                      cluster: SparTenCluster, stride: int = 1, padding: int = 1):
    """Process one convolutional layer with verification"""
    num_filters, in_channels, fh, fw = filters_4d.shape
    ih, iw, ic = input_feature_map.shape
    
    # Print sparsity check for input matrices
    print("\nSparsity Check:")
    print(f"Filter non-zeros: {np.count_nonzero(filters_4d)} out of {filters_4d.size}")
    print(f"Input non-zeros: {np.count_nonzero(input_feature_map)} out of {input_feature_map.size}")
    
    if padding > 0:
        padded_input = np.pad(input_feature_map, 
                            ((padding, padding), (padding, padding), (0, 0)),
                            mode='constant')
    else:
        padded_input = input_feature_map
    
    out_h = (ih + 2*padding - fh) // stride + 1
    out_w = (iw + 2*padding - fw) // stride + 1
    filters_2d = filters_4d.reshape(num_filters, -1)
    output = np.zeros((out_h, out_w, num_filters))
    
    total_energy = 0
    total_macs = 0
    total_dense_macs = 0
    all_rounds_verified = True
    
    # Process each spatial position
    for i in range(0, out_h):
        for j in range(0, out_w):
            print(f"\nProcessing spatial position ({i},{j})")
            
            h_start = i * stride
            h_end = h_start + fh
            w_start = j * stride
            w_end = w_start + fw
            patch = padded_input[h_start:h_end, w_start:w_end, :]
            patch_flat = patch.reshape(-1)
            
            # Reset all compute units before starting new position
            for cu in cluster.compute_units:
                cu.mac_unit.reset()
            
            # Process filters in rounds
            for round_idx, filter_start in enumerate(range(0, num_filters, cluster.num_compute_units)):
                filter_end = min(filter_start + cluster.num_compute_units, num_filters)
                current_filters = filters_2d[filter_start:filter_end]
                
                # Reset MACs before each round
                for cu in cluster.compute_units:
                    cu.mac_unit.reset()
                
                results, stats = cluster.process_input(current_filters, patch_flat)
                
                # Print detailed stats for this round
                print(f"\nRound {round_idx + 1} Stats:")
                print(f"Processing filters {filter_start} to {filter_end-1}")
                print(f"Current filters non-zeros: {np.count_nonzero(current_filters)} out of {current_filters.size}")
                print(f"Patch non-zeros: {np.count_nonzero(patch_flat)} out of {patch_flat.size}")
                
                round_verified = verify_multi_round_processing(
                    filters_2d, patch_flat, results, round_idx, 
                    filter_start, filter_end
                )
                all_rounds_verified &= round_verified
                
                output[i, j, filter_start:filter_end] = results
                total_energy += stats['total_energy']
                total_macs += stats['total_macs']
                total_dense_macs += stats['dense_macs']
    
    # Print final sparsity stats
    mac_efficiency = (total_macs/total_dense_macs)*100 if total_dense_macs > 0 else 0
    print("\nFinal Statistics:")
    print(f"Total MACs performed: {total_macs}")
    print(f"Total dense MACs possible: {total_dense_macs}")
    print(f"MAC efficiency: {mac_efficiency:.2f}%")
    
    return output, {
        'total_energy': total_energy,
        'total_macs': total_macs,
        'dense_macs': total_dense_macs,
        'mac_efficiency': mac_efficiency,
        'verified': all_rounds_verified
    }

def test_vgg16_sparten():
    print("Loading CIFAR10 pre-trained VGG16...")
    try:
        model = torch.hub.load('chenyaofo/pytorch-cifar-models', 'cifar10_vgg16_bn', pretrained=True)
    except:
        print("Failed to load from torch.hub, trying timm...")
        import timm
        model = timm.create_model('vgg16_bn_cifar10', pretrained=True)

    # Extract conv layer configurations
    conv_layers = []
    for name, module in model.features.named_children():
        if isinstance(module, torch.nn.Conv2d):
            weights = module.weight.data.cpu().numpy()
            conv_layers.append(weights)
            print(f"\nConv Layer {len(conv_layers)}:")
            print(f"Shape: {weights.shape}")
            print(f"Sparsity: {(1 - np.count_nonzero(weights)/weights.size)*100:.2f}%")

    # Create random input of correct size for CIFAR10 (32x32x3)
    input_data = np.random.randn(32, 32, 3)
    current_input = input_data

    # Create cluster with paper's configuration
    cluster = SparTenCluster(num_compute_units=32, chunk_size=128)
    
    # Process each layer
    layer_stats = {}
    
    for layer_idx, filters in enumerate(conv_layers):
        print(f"\nProcessing layer {layer_idx+1}")
        
        # Process through SparTen
        output, stats = process_conv_layer(filters, current_input, cluster)
        
        # Store statistics
        layer_stats[f'layer_{layer_idx+1}'] = stats
        
        # Apply ReLU
        output = np.maximum(0, output)
        
        # Apply MaxPool if needed (after certain layers according to VGG16 architecture)
        if layer_idx in [1, 3, 6, 9, 12]:  # MaxPool layers in VGG16
            output = output.reshape(1, *output.shape)
            output = torch.nn.functional.max_pool2d(
                torch.from_numpy(output), 
                kernel_size=2, 
                stride=2
            ).numpy()[0]
        
        current_input = output
        
        # Print layer statistics
        print(f"Layer {layer_idx+1} Statistics:")
        print(f"Shape: {current_input.shape}")
        print(f"MAC Efficiency: {stats['mac_efficiency']:.2f}%")
        print(f"Energy: {stats['total_energy']:.2f} pJ")
        print(f"MACs: {stats['total_macs']} / {stats['dense_macs']}")
    
    # Save detailed statistics to file
    save_layer_stats(layer_stats, "vgg16_cifar10_stats.txt")
    
    # Print overall statistics
    print("\nOverall VGG16-CIFAR10 Statistics:")
    total_energy = sum(stat['total_energy'] for stat in layer_stats.values())
    total_macs = sum(stat['total_macs'] for stat in layer_stats.values())
    total_dense_macs = sum(stat['dense_macs'] for stat in layer_stats.values())
    
    print(f"Total Energy: {total_energy:.2f} pJ")
    print(f"Total MAC Efficiency: {(total_macs/total_dense_macs)*100:.2f}%")
    print(f"Total MACs: {total_macs} / {total_dense_macs}")
    
    return layer_stats

if __name__ == "__main__":
    # Run test and collect statistics
    stats = test_vgg16_sparten()