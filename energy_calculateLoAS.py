import numpy as np

class ConvLoASCalculator:
    def __init__(self):
        self.energy_params = {
            'pseudo_accumulator': 0.16,
            'correction_accumulator': 0.16,
            'fast_prefix': 1.46,
            'laggy_prefix': 0.32,
            'lif_neuron': 0.075,
            'others': 0.88,
        }

    def extract_patches(self, input_image, kernel_size, timesteps):
        """Extract patches from input image for convolution."""
        H, W, T = input_image.shape
        Kh, Kw = kernel_size
        
        out_h = H - Kh + 1
        out_w = W - Kw + 1
        
        transformed = np.zeros((out_h * out_w, Kh * Kw, timesteps))
        
        idx = 0
        for i in range(out_h):
            for j in range(out_w):
                patch = input_image[i:i+Kh, j:j+Kw]
                transformed[idx] = patch.reshape(-1, timesteps)
                idx += 1
                
        return transformed

    def calculate_vector_energy(self, input_vector, kernel_vector, timesteps=4):
        """Calculate energy and statistics for one vector-vector multiplication."""
        # Calculate total possible accumulations in dense case
        dense_accumulations = len(kernel_vector) * timesteps
        # print('dense accumulations')
        # print(dense_accumulations)
        # Extract bitmasks
        bitmask_input = np.any(input_vector != 0, axis=1)
        bitmask_kernel = (kernel_vector != 0)
        
        # Count matches between bitmasks
        matches = np.count_nonzero(bitmask_input & bitmask_kernel)
        
        # Actual accumulations (assuming all 1s)
        actual_accumulations = matches * timesteps
        
        # Calculate corrections needed
        corrections_needed = 0
        match_positions = np.where(bitmask_input & bitmask_kernel)[0]
        matches_needing_correction = 0
        
        for pos in match_positions:
            spikes = input_vector[pos]
            if not np.all(spikes):
                corrections_needed += (timesteps - np.count_nonzero(spikes))
                matches_needing_correction += 1
        
        # Detailed statistics
        acc_stats = {
            'dense_accumulations': dense_accumulations,
            'actual_accumulations': actual_accumulations,
            'saved_accumulations': dense_accumulations - actual_accumulations,
            'sparsity_efficiency': 1 - (actual_accumulations / dense_accumulations) if dense_accumulations > 0 else 0,
            'input_sparsity': 1 - np.count_nonzero(bitmask_input) / len(bitmask_input),
            'kernel_sparsity': 1 - np.count_nonzero(bitmask_kernel) / len(bitmask_kernel),
            'total_matches': matches,
            'corrections_needed': corrections_needed,
            'matches_needing_correction': matches_needing_correction
        }
        
        # Energy breakdown
        energy_breakdown = {
            'fast_prefix': self.energy_params['fast_prefix'] if matches > 0 else 0,
            'laggy_prefix': self.energy_params['laggy_prefix'] if matches > 0 else 0,
            'pseudo_acc': matches * self.energy_params['pseudo_accumulator'],
            'correction_acc': corrections_needed * self.energy_params['correction_accumulator'],
            'lif': timesteps * self.energy_params['lif_neuron'],
            'others': self.energy_params['others']
        }
        
        return sum(energy_breakdown.values()), energy_breakdown, acc_stats

    # def convolve_and_calculate_energy(self, input_image, kernel, timesteps=4, threshold=1.0):
    #     """Perform convolution and calculate total energy consumption."""
    #     H, W, T = input_image.shape
    #     Kh, Kw = kernel.shape
        
    #     out_h = H - Kh + 1
    #     out_w = W - Kw + 1
        
    #     transformed_input = self.extract_patches(input_image, kernel.shape, timesteps)
    #     kernel_vector = kernel.flatten()
        
    #     # Initialize outputs
    #     output = np.zeros((out_h, out_w, timesteps))
    #     total_energy = 0
    #     energy_breakdowns = []
        
    #     # Initialize cumulative statistics
    #     total_acc_stats = {
    #         'dense_accumulations': 0,
    #         'actual_accumulations': 0,
    #         'saved_accumulations': 0,
    #         'corrections_needed': 0,
    #         'total_matches': 0,
    #         'matches_needing_correction': 0
    #     }
        
    #     # Process each patch
    #     for i in range(len(transformed_input)):
    #         patch_energy, breakdown, patch_acc_stats = self.calculate_vector_energy(
    #             transformed_input[i], kernel_vector, timesteps
    #         )
    #         print('patch acc stats')
    #         print(patch_acc_stats)
            
    #         total_energy += patch_energy
    #         energy_breakdowns.append(breakdown)
            
    #         # Accumulate statistics
    #         for key in total_acc_stats:
    #             if key in patch_acc_stats:
    #                 total_acc_stats[key] += patch_acc_stats[key]
            
    #         # Calculate output for this patch
    #         for t in range(timesteps):
    #             spikes = transformed_input[i, :, t]
    #             conv_result = np.sum(spikes * kernel_vector)
    #             output[i // out_w, i % out_w, t] = 1 if conv_result >= threshold else 0
        
    #     # Calculate overall efficiency metrics
    #     if total_acc_stats['dense_accumulations'] > 0:
    #         total_acc_stats['overall_sparsity_efficiency'] = (
    #             1 - (total_acc_stats['actual_accumulations'] / 
    #                  total_acc_stats['dense_accumulations'])
    #         )
    #     else:
    #         total_acc_stats['overall_sparsity_efficiency'] = 0
            
    #     return total_energy, energy_breakdowns, output, total_acc_stats


import numpy as np

# VGG16 Configuration with sparsity levels from paper
VGG16_CONFIG = [
    # (type, in_channels, out_channels, kernel_size, stride, weight_sparsity)
    ('conv', 3, 64, 3, 1, 0.982),      # Conv1-1
    ('conv', 64, 64, 3, 1, 0.982),     # Conv1-2
    ('maxpool', None, None, 2, 2, None),
    ('conv', 64, 128, 3, 1, 0.982),    # Conv2-1
    ('conv', 128, 128, 3, 1, 0.982),   # Conv2-2
    ('maxpool', None, None, 2, 2, None),
    ('conv', 128, 256, 3, 1, 0.982),   # Conv3-1
    ('conv', 256, 256, 3, 1, 0.982),   # Conv3-2
    ('conv', 256, 256, 3, 1, 0.982),   # Conv3-3
    ('maxpool', None, None, 2, 2, None),
    ('conv', 256, 512, 3, 1, 0.982),   # Conv4-1
    ('conv', 512, 512, 3, 1, 0.982),   # Conv4-2
    ('conv', 512, 512, 3, 1, 0.982),   # Conv4-3
    ('maxpool', None, None, 2, 2, None),
    ('conv', 512, 512, 3, 1, 0.982),   # Conv5-1
    ('conv', 512, 512, 3, 1, 0.982),   # Conv5-2
    ('conv', 512, 512, 3, 1, 0.982),   # Conv5-3
    ('maxpool', None, None, 2, 2, None),
]

class ExternalOperations:
    def __init__(self, timesteps=4):
        self.timesteps = timesteps
    
    def maxpool2d(self, input_data, kernel_size=2, stride=2):
        """
        Implement 2D max pooling for spike data.
        Input: (height, width, channels, timesteps)
        """
        H, W, C, T = input_data.shape
        out_h = (H - kernel_size) // stride + 1
        out_w = (W - kernel_size) // stride + 1
        
        output = np.zeros((out_h, out_w, C, T))
        
        for i in range(out_h):
            for j in range(out_w):
                for c in range(C):
                    for t in range(T):
                        patch = input_data[
                            i*stride:i*stride+kernel_size, 
                            j*stride:j*stride+kernel_size,
                            c,
                            t
                        ]
                        output[i, j, c, t] = 1 if np.any(patch == 1) else 0
        
        return output
    
    def prepare_layer_input(self, input_data, silent_neuron_ratio=0.713):
        """
        Prepare input with proper silent neuron ratio.
        silent_neuron_ratio: 71.3% from paper (proportion of neurons that never spike)
        """
        H, W, C = input_data.shape
        timesteps = self.timesteps
        
        # Initialize output with all neurons silent
        output = np.zeros((H, W, C, timesteps))
        
        # Create mask for active neurons (28.7% of neurons)
        active_mask = np.random.choice(
            [0, 1],
            size=(H, W, C),
            p=[silent_neuron_ratio, 1-silent_neuron_ratio]
        )
        
        # For active neurons, generate spike patterns
        active_positions = np.where(active_mask == 1)
        for i, j, k in zip(*active_positions):
            # For active neurons, generate some spikes across timesteps
            # You might want to adjust this probability based on desired activity level
            output[i, j, k] = np.random.choice([0, 1], size=timesteps, p=[0.5, 0.5])
            
            # Ensure at least one spike for active neurons
            if np.sum(output[i, j, k]) == 0:
                # If no spikes generated, force at least one
                random_timestep = np.random.randint(0, timesteps)
                output[i, j, k, random_timestep] = 1
        
        return output

class VGG16LoASCalculator:
    def __init__(self, timesteps=4):
        self.conv_calculator = ConvLoASCalculator()
        self.external_ops = ExternalOperations(timesteps)
        self.timesteps = timesteps
        self.spike_sparsity = 0.796  # From paper

    def analyze_layer_results(self, layer_results):
        """Analyze and print detailed per-layer statistics."""
        print("\n=== Per-Layer Analysis ===")
        
        total_energy = 0
        total_accumulations = 0
        total_saved = 0
        total_corrections = 0
        
        for i, result in enumerate(layer_results):
            layer_type = result['layer_type']
            print(f"\nLayer {i+1} ({layer_type}):")
            print(f"Output Shape: {result['output_shape']}")
            
            if layer_type == 'conv':  # Only conv layers have energy and stats
                energy = result['energy']
                stats = result['stats']     # Energy breakdown
                acc_stats = result['acc_stats']  # Accumulation statistics
                total_energy += energy
                
                # Print layer statistics using acc_stats
                if acc_stats:
                    print(f"\nComputation Statistics:")
                    print(f"Total possible accumulations: {acc_stats['dense_accumulations']:,}")
                    print(f"Actual accumulations: {acc_stats['actual_accumulations']:,}")
                    print(f"Accumulations saved: {acc_stats['saved_accumulations']:,}")
                    if acc_stats['dense_accumulations'] > 0:
                        sparsity_efficiency = (acc_stats['saved_accumulations'] / acc_stats['dense_accumulations']) * 100
                        print(f"Sparsity efficiency: {sparsity_efficiency:.2f}%")
                    else:
                        print("Sparsity efficiency: N/A (no accumulations)")
                    print(f"Corrections needed: {acc_stats['corrections_needed']:,}")
                    
                    # Update totals
                    total_accumulations += acc_stats['dense_accumulations']
                    total_saved += acc_stats['saved_accumulations']
                    total_corrections += acc_stats['corrections_needed']
                
                print(f"\nEnergy Breakdown:")
                # Use stats (energy_breakdown) for energy components
                for component, value in stats.items():
                    print(f"{component}: {value:.2f} pJ")
                print(f"Total Layer Energy: {energy:.2f} pJ")
            
            else:  # maxpool layers
                print("No energy consumption (external operation)")
        
        # Print overall statistics
        print("\n=== Overall Network Statistics ===")
        print(f"Total Energy Consumption: {total_energy:.2f} pJ")
        print(f"Total Possible Accumulations: {total_accumulations:,}")
        print(f"Total Accumulations Saved: {total_saved:,}")
        if total_accumulations > 0:
            overall_efficiency = (total_saved / total_accumulations) * 100
            print(f"Overall Sparsity Efficiency: {overall_efficiency:.2f}%")
            correction_rate = (total_corrections / total_accumulations) * 100
            print(f"Correction Rate: {correction_rate:.2f}%")
        else:
            print("Overall Sparsity Efficiency: N/A (no accumulations)")
            print("Correction Rate: N/A (no accumulations)")
        
        return {
            'total_energy': total_energy,
            'total_accumulations': total_accumulations,
            'total_saved': total_saved,
            'total_corrections': total_corrections
        }
        
    def generate_sparse_kernel(self, shape, sparsity):
        """Generate sparse kernel with integer weights between -10 to 10."""
        # Generate random integers between -10 and 10
        kernel = np.random.randint(-10, 11, size=shape)
        # Generate sparsity mask
        mask = np.random.choice(
            [0, 1], 
            size=shape, 
            p=[sparsity, 1-sparsity]
        )
        return kernel * mask
    
    def process_layer(self, input_data, layer_config):
        """Process a single layer and return energy stats."""
        layer_type, in_channels, out_channels, kernel_size, stride, weight_sparsity = layer_config
        
        if layer_type == 'conv':
            # Generate sparse kernel
            kernel_shape = (kernel_size, kernel_size, in_channels, out_channels)
            kernel = self.generate_sparse_kernel(kernel_shape, weight_sparsity)
            
            # Process convolution through LoAS
            energy, stats, output, acc_stats = self.process_conv_layer(input_data, kernel)
            
            return energy, stats, output, acc_stats
            
        elif layer_type == 'maxpool':
            # Handle maxpool externally
            output = self.external_ops.maxpool2d(input_data, kernel_size, stride)
            return 0, None, output, None  # No energy consumption or stats for external ops
    
    def process_conv_layer(self, input_data, kernel):
        """Process convolution layer through LoAS."""
        total_energy = 0
        all_acc_stats = []  # Only need accumulation stats
        H, W, C, T = input_data.shape
        Kh, Kw, Cin, Cout = kernel.shape
        
        # Output dimensions
        out_h = H - Kh + 1
        out_w = W - Kw + 1
        output = np.zeros((out_h, out_w, Cout, T))
        
        # For each output channel
        for cout in range(Cout):
            # For each spatial location
            for i in range(out_h):
                for j in range(out_w):
                    # Extract patch and reshape for LoAS
                    patch = input_data[i:i+Kh, j:j+Kw, :, :]
                    patch_flat = patch.reshape(-1, T)  # Flatten spatial and channel dims
                    kernel_flat = kernel[:, :, :, cout].reshape(-1)
                    
                    # Process through LoAS
                    energy, energy_breakdown, acc_stats = self.conv_calculator.calculate_vector_energy(
                        patch_flat, kernel_flat, self.timesteps
                    )
                    
                    # Compute output for this patch using spike patterns and kernel
                    patch_result = np.zeros(T)
                    for t in range(T):
                        conv_result = np.sum(patch_flat[:, t] * kernel_flat)
                        patch_result[t] = 1 if conv_result >= 1.0 else 0  # LIF threshold
                    
                    total_energy += energy
                    all_acc_stats.append(acc_stats)  # Store accumulation statistics
                    output[i, j, cout, :] = patch_result
        
        # Combine accumulation statistics
        combined_acc_stats = {
            'dense_accumulations': sum(stat['dense_accumulations'] for stat in all_acc_stats),
            'actual_accumulations': sum(stat['actual_accumulations'] for stat in all_acc_stats),
            'saved_accumulations': sum(stat['saved_accumulations'] for stat in all_acc_stats),
            'corrections_needed': sum(stat['corrections_needed'] for stat in all_acc_stats)
        }
        
        return total_energy, energy_breakdown, output, combined_acc_stats
    
    def run_inference(self, input_image):
        """Run full VGG16 inference and collect statistics."""
        current_input = self.external_ops.prepare_layer_input(input_image, self.spike_sparsity)
        
        layer_results = []
        total_energy = 0
        
        print("\nProcessing VGG16 layers...")
        
        for i, layer_config in enumerate(VGG16_CONFIG):
            # Process layer
            energy, stats, output, acc_stats = self.process_layer(current_input, layer_config)
            
            # Collect results
            layer_results.append({
                'layer_type': layer_config[0],
                'energy': energy,
                'stats': stats,
                'acc_stats': acc_stats,  # Add accumulation statistics
                'output_shape': output.shape,
                'config': layer_config
            })
            
            total_energy += energy
            current_input = output
        
        # Analyze and print detailed layer results
        self.analyze_layer_results(layer_results)
        
        return total_energy, layer_results

def main():
    # Create sample input image (224x224x3 for VGG16)
    input_image = np.random.rand(224, 224, 3)
    
    # Initialize calculator
    calculator = VGG16LoASCalculator(timesteps=4)
    
    # Run inference
    total_energy, layer_results = calculator.run_inference(input_image)

if __name__ == "__main__":
    main()
