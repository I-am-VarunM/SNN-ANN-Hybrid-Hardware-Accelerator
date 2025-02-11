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

    def convolve_and_calculate_energy(self, input_image, kernel, timesteps=4, threshold=1.0):
        """Perform convolution and calculate total energy consumption."""
        H, W, T = input_image.shape
        Kh, Kw = kernel.shape
        
        out_h = H - Kh + 1
        out_w = W - Kw + 1
        
        transformed_input = self.extract_patches(input_image, kernel.shape, timesteps)
        kernel_vector = kernel.flatten()
        
        # Initialize outputs
        output = np.zeros((out_h, out_w, timesteps))
        total_energy = 0
        energy_breakdowns = []
        
        # Initialize cumulative statistics
        total_acc_stats = {
            'dense_accumulations': 0,
            'actual_accumulations': 0,
            'saved_accumulations': 0,
            'corrections_needed': 0,
            'total_matches': 0,
            'matches_needing_correction': 0
        }
        
        # Process each patch
        for i in range(len(transformed_input)):
            patch_energy, breakdown, patch_acc_stats = self.calculate_vector_energy(
                transformed_input[i], kernel_vector, timesteps
            )
            
            total_energy += patch_energy
            energy_breakdowns.append(breakdown)
            
            # Accumulate statistics
            for key in total_acc_stats:
                if key in patch_acc_stats:
                    total_acc_stats[key] += patch_acc_stats[key]
            
            # Calculate output for this patch
            for t in range(timesteps):
                spikes = transformed_input[i, :, t]
                conv_result = np.sum(spikes * kernel_vector)
                output[i // out_w, i % out_w, t] = 1 if conv_result >= threshold else 0
        
        # Calculate overall efficiency metrics
        if total_acc_stats['dense_accumulations'] > 0:
            total_acc_stats['overall_sparsity_efficiency'] = (
                1 - (total_acc_stats['actual_accumulations'] / 
                     total_acc_stats['dense_accumulations'])
            )
        else:
            total_acc_stats['overall_sparsity_efficiency'] = 0
            
        return total_energy, energy_breakdowns, output, total_acc_stats

def main():
    # Example setup
    timesteps = 4
    height = 4
    width = 4
    
    # Create input image with sparse spikes
    input_image = np.zeros((height, width, timesteps))
    for i in range(height):
        for j in range(width):
            input_image[i, j, :] = np.random.choice([0, 1], size=timesteps, p=[0.8, 0.2])
    
    # Create sparse kernel
    kernel = np.array([
        [1, 0],
        [0, 1]
    ])
    
    calculator = ConvLoASCalculator()
    
    # Calculate energy and get output
    total_energy, energy_breakdowns, output, acc_stats = calculator.convolve_and_calculate_energy(
        input_image, kernel, timesteps
    )
    
    # Print results
    print("\n=== LoAS Convolution Analysis ===")
    print("\nInput/Output Information:")
    print(f"Input shape: {input_image.shape}")
    print(f"Kernel shape: {kernel.shape}")
    print(f"Output shape: {output.shape}")
    
    print("\nAccumulation Statistics:")
    print(f"Total possible accumulations: {acc_stats['dense_accumulations']}")
    print(f"Actual accumulations needed: {acc_stats['actual_accumulations']}")
    print(f"Accumulations saved: {acc_stats['saved_accumulations']}")
    print(f"Sparsity efficiency: {acc_stats['overall_sparsity_efficiency']*100:.2f}%")
    
    print("\nCorrection Statistics:")
    print(f"Total matches: {acc_stats['total_matches']}")
    print(f"Corrections needed: {acc_stats['corrections_needed']}")
    print(f"Matches needing correction: {acc_stats['matches_needing_correction']}")
    
    print("\nEnergy Breakdown:")
    for component, energy in energy_breakdowns[0].items():
        print(f"{component}: {energy:.2f} pJ")
    print(f"Total Energy: {total_energy:.2f} pJ")

if __name__ == "__main__":
    main()