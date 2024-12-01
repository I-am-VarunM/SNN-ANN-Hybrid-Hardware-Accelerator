from collections import defaultdict
import numpy as np

class FiberLine:
    def __init__(self, bitmask, data, pointer=None):
        self.bitmask = bitmask  # Bitmask representation
        self.data = data        # Non-zero values
        self.pointer = pointer  # Pointer to next line if needed
        self.priority = 0       # Priority counter for replacement
        self.srrip = 2         # 2-bit SRRIP counter (2 = long, 1 = intermediate, 0 = immediate)
        self.dirty = False     # Dirty bit for writes
        self.valid = True      # Valid bit

class FiberCache:
    def __init__(self, num_banks=16, ways=16, lines_per_bank=1024, num_pes=32):
        """Initialize FiberCache
        
        Args:
            num_banks: Number of banks (default 16)
            ways: Associativity per set (default 16) 
            lines_per_bank: Number of lines per bank (default 1024)
            num_pes: Number of processing elements (for priority counter)
        """
        self.num_banks = num_banks
        self.ways = ways
        self.lines_per_bank = lines_per_bank
        
        # Priority counter bits based on number of PEs
        self.priority_bits = (num_pes.bit_length() + 1)  # +1 for margin
        self.max_priority = (1 << self.priority_bits) - 1
        
        # Initialize cache structure
        self.banks = []
        for _ in range(num_banks):
            sets = []
            for _ in range(lines_per_bank // ways):
                sets.append([None] * ways)  # Each set has 'ways' number of lines
            self.banks.append(sets)
            
        # Statistics
        self.hits = 0
        self.misses = 0
        
    def _get_bank_set_way(self, addr):
        """Calculate bank and set from address"""
        bank_id = addr % self.num_banks
        set_id = (addr // self.num_banks) % (self.lines_per_bank // self.ways)
        return bank_id, set_id
        
    def _get_victim_way(self, bank_id, set_id):
        """Get victim way using priority and SRRIP"""
        ways = self.banks[bank_id][set_id]
        min_priority = float('inf')
        min_srrip = float('inf')
        victim_way = 0
        found_empty = False
        
        # First try to find empty way
        for way in range(self.ways):
            if ways[way] is None:
                return way
                
        # Find way with minimum priority
        for way in range(self.ways):
            line = ways[way]
            if line.priority < min_priority:
                min_priority = line.priority
                min_srrip = line.srrip
                victim_way = way
            elif line.priority == min_priority:
                # Break ties using SRRIP
                if line.srrip < min_srrip:
                    min_srrip = line.srrip
                    victim_way = way
                    
        return victim_way

    def fetch(self, addr, bitmask, data, pointer=None):
        """Fetch data into cache (explicit data orchestration)
        Increments priority as this indicates future use"""
        bank_id, set_id = self._get_bank_set_way(addr)
        way = self._get_victim_way(bank_id, set_id)
        
        # Create new line
        line = FiberLine(bitmask, data, pointer)
        line.priority = min(self.max_priority, 1)  # Set initial priority
        line.srrip = 2  # Set SRRIP to long re-reference
        
        # Handle eviction if needed
        old_line = self.banks[bank_id][set_id][way]
        if old_line and old_line.dirty:
            # In real implementation, write back to memory
            pass
            
        # Insert new line    
        self.banks[bank_id][set_id][way] = line
        
    def read(self, addr):
        """Read data from cache
        Decrements priority as data was used"""
        bank_id, set_id = self._get_bank_set_way(addr)
        
        # Search in ways
        for way in range(self.ways):
            line = self.banks[bank_id][set_id][way]
            if line and line.valid:
                # Cache hit
                self.hits += 1
                
                # Update priority and SRRIP
                line.priority = max(0, line.priority - 1)  # Decrease priority
                line.srrip = 2  # Reset SRRIP on hit
                
                return line.bitmask, line.data
                
        # Cache miss
        self.misses += 1
        return None
        
    def write(self, addr, bitmask, data, pointer=None):
        """Write data to cache"""
        bank_id, set_id = self._get_bank_set_way(addr)
        way = self._get_victim_way(bank_id, set_id)
        
        # Create new line
        line = FiberLine(bitmask, data, pointer) 
        line.dirty = True
        line.priority = 1  # Set initial priority
        line.srrip = 2    # Set SRRIP to long re-reference
        
        # Handle eviction
        old_line = self.banks[bank_id][set_id][way]
        if old_line and old_line.dirty:
            # Write back in real implementation
            pass
            
        self.banks[bank_id][set_id][way] = line
        
    def consume(self, addr):
        """Read and invalidate line (for partial outputs)"""
        bank_id, set_id = self._get_bank_set_way(addr)
        
        # Search in ways
        for way in range(self.ways):
            line = self.banks[bank_id][set_id][way]
            if line and line.valid:
                self.hits += 1
                line.valid = False  # Invalidate without writeback
                return line.bitmask, line.data
                
        self.misses += 1
        return None

    def get_stats(self):
        """Return cache statistics"""
        total_accesses = self.hits + self.misses
        hit_rate = self.hits / total_accesses if total_accesses > 0 else 0
        return {
            'hits': self.hits,
            'misses': self.misses, 
            'hit_rate': hit_rate
        }