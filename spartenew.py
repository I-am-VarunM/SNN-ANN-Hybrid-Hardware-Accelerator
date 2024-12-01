import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from collections import deque

@dataclass
class SparseData:
    sparsemap: np.ndarray  
    values: np.ndarray     
    original_shape: Tuple[int, ...]  

    @classmethod
    def from_dense(cls, data: np.ndarray):
        """Convert dense array to sparse format"""
        sparsemap = (data != 0).astype(np.uint8)
        values = data[data != 0]
        return cls(sparsemap, values, data.shape)
    
    def get_values_at_positions(self, positions: np.ndarray) -> np.ndarray:
        """Get values at specific positions in the sparsemap"""
        value_indices = np.zeros_like(positions)
        for i, pos in enumerate(positions):
            value_indices[i] = np.sum(self.sparsemap[:pos])
        return self.values[value_indices]

@dataclass
class MACUnit:
    """Represents a single multiply-accumulate unit"""
    def __init__(self):
        self.accumulator = 0.0
        self.debug_multiplications = []  # Store multiplications for debugging
        
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
class ComputeUnit:
    id: int
    pipeline_stages: dict = None
    current_stage: str = None
    stage_cycle_count: int = 0
    
    def __post_init__(self):
        self.mac_unit = MACUnit()
        self.pipeline_stages = {
            'and': None,
            'priority_encode': None,
            'prefix_sum': None,
            'memory': None,
            'mac': None
        }

    def process_sparse_multiply(self, filter_chunk: SparseData, input_chunk: SparseData) -> Tuple[int, int]:
        """Process sparse multiplication through pipeline stages"""
        # Initial AND operation (1 cycle)
        matches = filter_chunk.sparsemap & input_chunk.sparsemap
        match_positions = np.where(matches)[0]
        total_cycles = 0
        total_macs = len(match_positions)
        
        print(f"\nCU{self.id} processing:")
        print(f"  Filter sparsemap: {filter_chunk.sparsemap}")
        print(f"  Filter values: {filter_chunk.values}")
        print(f"  Input sparsemap: {input_chunk.sparsemap}")
        print(f"  Input values: {input_chunk.values}")
        print(f"  Found {len(match_positions)} matches at positions: {match_positions}")
        
        # For each matching position, go through pipeline
        for pos_idx, pos in enumerate(match_positions):
            # Pipeline stages timing
            if pos_idx > 0:
                total_cycles += 2  # AND + priority encode
            else:
                total_cycles += 1  # just priority encode since AND is done
                
            # Prefix Sum (1 cycle)
            filter_offset = np.sum(filter_chunk.sparsemap[:pos])
            input_offset = np.sum(input_chunk.sparsemap[:pos])
            total_cycles += 1
            
            # Memory Access (1 cycle)
            filter_val = filter_chunk.values[filter_offset]
            input_val = input_chunk.values[input_offset]
            total_cycles += 1
            
            # MAC Operation (1 cycle)
            self.mac_unit.multiply_accumulate(filter_val, input_val)
            total_cycles += 1
            
            print(f"  Position {pos}: filter_val={filter_val} * input_val={input_val}")
            
        if not match_positions.size:
            print("  No matches found")
            total_cycles = 4  # minimum pipeline stages
            
        print(f"  Total cycles: {total_cycles}")
        print(f"  Current accumulator value: {self.mac_unit.get_result()}")
        
        return total_cycles, total_macs

class SparTenCluster:
    def __init__(self, num_compute_units: int = 32, chunk_size: int = 128):
        self.num_compute_units = num_compute_units
        self.chunk_size = chunk_size
        self.compute_units = [ComputeUnit(i) for i in range(num_compute_units)]
        self.current_cycle = 0
        self.partial_sums = {}  # Store partial sums for each filter
        self.num_filters = 0  # Track total number of filters
            
    def process_chunk(self, filter_chunks: List[SparseData], input_chunk: SparseData, 
                     is_last_chunk: bool) -> List[float]:
        print(f"\nProcessing {'last' if is_last_chunk else 'new'} chunk at cycle {self.current_cycle}")
        print(f"Input chunk sparsemap: {input_chunk.sparsemap}")
        print(f"Input chunk values: {input_chunk.values}")
        
        print("\nFilter assignments to Compute Units:")
        for i, fc in enumerate(filter_chunks):
            print(f"CU{i} <- Filter sparsemap: {fc.sparsemap}, values: {fc.values}")
            
        # Calculate total number of filters if first chunk
        if self.current_cycle == 0:
            self.num_filters = len(filter_chunks) // 2  # Assuming 2 chunks per filter
        
        # First identify all matches that need to be processed
        all_matches = []
        print("\nChecking for non-zero matches:")
        for chunk_idx, filter_chunk in enumerate(filter_chunks):
            filter_idx = chunk_idx // 2  # Calculate actual filter index
            matches = filter_chunk.sparsemap & input_chunk.sparsemap
            match_positions = np.where(matches)[0]
            if len(match_positions) > 0:
                print(f"Filter {filter_idx} has {len(match_positions)} matches at positions: {match_positions}")
                filter_values = filter_chunk.get_values_at_positions(match_positions)
                input_values = input_chunk.get_values_at_positions(match_positions)
                print(f"     Values to multiply: {filter_values} * {input_values}")
                all_matches.append((filter_idx, match_positions, filter_chunk))
            
        if not all_matches:
            print("No matches found in any filters")
            return []
            
        # Process matches in pipelined rounds based on available compute units
        total_rounds = (len(all_matches) + self.num_compute_units - 1) // self.num_compute_units
        total_macs = 0
        
        print(f"\nProcessing {len(all_matches)} filter matches in {total_rounds} pipelined rounds using {self.num_compute_units} compute units")
        
        # Calculate total cycles needed
        total_cycles = 5 + (total_rounds - 1)
        
        # Pipeline execution visualization
        print("\nPipeline Execution:")
        base_cycle = self.current_cycle
        for cycle in range(total_cycles):
            current_cycle = base_cycle + cycle + 1
            print(f"\nCycle {current_cycle}:")
            for round_idx in range(total_rounds):
                if cycle >= round_idx and cycle < round_idx + 5:
                    stage_idx = cycle - round_idx
                    stage_name = ['AND', 'Priority Encode', 'Prefix Sum', 'Memory', 'MAC'][stage_idx]
                    print(f"  Round {round_idx + 1}: {stage_name} stage")
        
        self.current_cycle += total_cycles
        
        results = []
        # Process each round
        for round_idx in range(total_rounds):
            # Reset MAC units at start of each round
            for cu in self.compute_units:
                cu.mac_unit.reset()
                
            start_idx = round_idx * self.num_compute_units
            end_idx = min(start_idx + self.num_compute_units, len(all_matches))
            round_matches = all_matches[start_idx:end_idx]
            
            print(f"\nRound {round_idx + 1} processing:")
            
            # Process matches in this round
            for cu_idx, (filter_idx, match_positions, filter_chunk) in enumerate(round_matches):
                cu = self.compute_units[cu_idx]
                print(f"CU{cu_idx} processing Filter {filter_idx}")
                
                cycles, macs = cu.process_sparse_multiply(filter_chunk, input_chunk)
                total_macs += macs
                
                # Get result and add to previous partial sum if it exists
                result = cu.mac_unit.get_result()
                if filter_idx in self.partial_sums:
                    result += self.partial_sums[filter_idx]
                
                if is_last_chunk:
                    results.append((filter_idx, result))
                else:
                    self.partial_sums[filter_idx] = result
                
                print(f"\nCU{cu_idx} (Filter {filter_idx}):")
                if is_last_chunk:
                    print(f"  Final result: {result}")
                else:
                    print(f"  Partial sum: {result}")
                print(f"  Multiplications performed: {cu.mac_unit.get_debug_info()}")
        
        print(f"\nAll pipelined rounds complete at cycle {self.current_cycle}")
        print(f"Total cycles taken: {total_cycles}")
        print(f"Total MAC operations performed: {total_macs}")
        
        if is_last_chunk:
            # Ensure all filters have a result (even if zero)
            final_results = []
            for i in range(self.num_filters):
                if i not in [r[0] for r in results]:
                    final_results.append((i, 0.0))  # Add zero for filters with no matches
            final_results.extend(results)
            
            # Sort by filter index and extract just the values
            final_results.sort(key=lambda x: x[0])
            final_results = [r[1] for r in final_results]
            
            print("\nFinal results for all filters:", final_results)
            # Clear partial sums and filter count for next computation
            self.partial_sums.clear()
            self.num_filters = 0
            return final_results
        
        return []

def verify_computation(filters: np.ndarray, inputs: np.ndarray, sparse_results: List[float]):
    """Verify sparse computation against dense computation"""
    print("\nVerifying computation:")
    print(f"Original dense filters shape: {filters.shape}")
    print(f"Original dense inputs shape: {inputs.shape}")
    
    # Compute dense results but only keep non-zero results
    dense_results = []
    for f in filters:
        result = np.sum(f * inputs)
        # Only include non-zero results to match sparse computation behavior
        if result != 0:
            dense_results.append(result)
    
    print("\nResults comparison:")
    print("Dense results:", dense_results)
    print("Sparse results:", sparse_results)
    
    # Compare results
    if len(dense_results) != len(sparse_results):
        print("Number of results doesn't match!")
        print(f"Dense results length: {len(dense_results)}")
        print(f"Sparse results length: {len(sparse_results)}")
        return False
    
    matches = np.allclose(dense_results, sparse_results, rtol=1e-05, atol=1e-08)
    if matches:
        print("✅ Verification PASSED: Results match!")
    else:
        print("❌ Verification FAILED: Results don't match!")
        print("Differences:", np.array(dense_results) - np.array(sparse_results))
    
    return matches

def test_sparten_simulator():
    print("Running SparTen Simulator Tests...")
    
    print("\nTest Case 1: Simple sparse matrix multiplication")
    filters = np.array([
        [1, 0, 3, 0],
        [0, 2, 0, 4]
    ])
    inputs = np.array([1, 2, 0, 4])
    
    expected_results = [1*1 + 3*0, 2*2 + 4*4]  # [1, 20]
    print(f"Expected results: {expected_results}")
    
    # Convert to sparse format
    filter_chunks = [SparseData.from_dense(f) for f in filters]
    input_chunk = SparseData.from_dense(inputs)
    
    # Create cluster and process
    cluster = SparTenCluster(num_compute_units=2, chunk_size=4)
    results = cluster.process_chunk(filter_chunks, input_chunk, is_last_chunk=True)
    
    print("\nVerifying Test Case 1:")
    verify_computation(filters, inputs, results)

    print("\nTest Case 2: More matches than compute units")
    filters = np.array([
        [1, 0, 0, 2],  # 2 matches when paired with input
        [0, 3, 0, 4],  # 2 matches
        [5, 0, 6, 0],  # 1 match
        [0, 7, 0, 8]   # 2 matches
    ])
    inputs = np.array([1, 3, 0, 4])  # Will create multiple matches with filters
    
    print(f"\nInput array: {inputs}")
    print("Filters:")
    print(filters)
    print("\nExpected matches per filter:")
    for i, f in enumerate(filters):
        matches = np.where((f != 0) & (inputs != 0))[0]
        print(f"Filter {i}: {len(matches)} matches at positions {matches}")
    
    # Convert to sparse format
    filter_chunks = [SparseData.from_dense(f) for f in filters]
    input_chunk = SparseData.from_dense(inputs)
    
    # Process with only 2 compute units - should need multiple rounds
    cluster = SparTenCluster(num_compute_units=2, chunk_size=4)
    results = cluster.process_chunk(filter_chunks, input_chunk, is_last_chunk=True)
    
    print("\nVerifying Test Case 2:")
    verify_computation(filters, inputs, results)

    print("\nTest Case 3: Larger sparse matrix with multiple chunks")
    filters = np.zeros((6, 8))  # 6 filters
    # Set some values to create specific match patterns
    filters[0, [0, 4]] = [1, 2]
    filters[1, [1, 5]] = [3, 4]
    filters[2, [2, 6]] = [5, 6]
    filters[3, [0, 7]] = [7, 8]
    filters[4, [1, 4]] = [9, 10]
    filters[5, [3, 5]] = [11, 12]
    
    inputs = np.zeros(8)
    inputs[[0, 1, 4, 5]] = [7, 8, 9, 10]  # Will create multiple matches with filters
    
    print("\nInput array:", inputs)
    print("Filters:")
    print(filters)
    print("\nExpected matches per filter:")
    for i, f in enumerate(filters):
        matches = np.where((f != 0) & (inputs != 0))[0]
        print(f"Filter {i}: {len(matches)} matches at positions {matches}")
    
    # Split into chunks and convert to sparse
    chunk_size = 4
    filter_chunks = []
    for f in filters:
        chunks = [f[i:i+chunk_size] for i in range(0, len(f), chunk_size)]
        filter_chunks.extend([SparseData.from_dense(chunk) for chunk in chunks])
    
    input_chunks = [inputs[i:i+chunk_size] for i in range(0, len(inputs), chunk_size)]
    input_chunks = [SparseData.from_dense(chunk) for chunk in input_chunks]
    
    # Process chunks with only 2 compute units
    cluster = SparTenCluster(num_compute_units=2, chunk_size=chunk_size)
    final_results = []
    
    for chunk_idx, input_chunk in enumerate(input_chunks):
        is_last = chunk_idx == len(input_chunks) - 1
        chunk_results = cluster.process_chunk(
            filter_chunks[chunk_idx::len(input_chunks)], 
            input_chunk,
            is_last
        )
        if is_last:
            final_results = chunk_results
    
    print("\nVerifying Test Case 3:")
    verify_computation(filters, inputs, final_results)

if __name__ == "__main__":
    test_sparten_simulator()