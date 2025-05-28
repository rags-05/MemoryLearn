class MemoryManager:
    """
    Manages the virtual memory system, including address translation and page replacement.
    """
    def __init__(self, virtual_memory_size=2**16, page_size=2**8, physical_memory_size=2**10):
        """
        Initialize the memory manager.
        
        Args:
            virtual_memory_size: Size of virtual memory in bytes
            page_size: Size of a page in bytes
            physical_memory_size: Size of physical memory in bytes
        """
        self.virtual_memory_size = virtual_memory_size
        self.page_size = page_size
        self.physical_memory_size = physical_memory_size
        
        # Calculate the number of pages and frames
        self.num_pages = virtual_memory_size // page_size
        self.num_frames = physical_memory_size // page_size
        
        # Initialize memory structures
        self.virtual_memory = [0] * virtual_memory_size
        self.physical_memory = [0] * physical_memory_size
        
        # Initialize page table
        from vmm.page_table import PageTable
        self.page_table = PageTable(self.num_pages, self.num_frames)
        
        # Initially, no replacement algorithm is set
        self.replacement_algorithm = None
        
        # Performance metrics
        self.page_faults = 0
        self.memory_accesses = 0
        self.hit_ratio = 0.0
    
    def set_replacement_algorithm(self, algorithm):
        """Set the page replacement algorithm."""
        self.replacement_algorithm = algorithm
        
    def access_memory(self, virtual_address):
        """
        Access memory at the given virtual address.
        
        Args:
            virtual_address: The virtual address to access
            
        Returns:
            The value at the given virtual address
        """
        if virtual_address < 0 or virtual_address >= self.virtual_memory_size:
            raise ValueError(f"Virtual address {virtual_address} out of bounds")
        
        self.memory_accesses += 1
        
        # Calculate page number and offset
        page_number = virtual_address // self.page_size
        offset = virtual_address % self.page_size
        
        # Check if page is in physical memory
        frame_number = self.page_table.get_frame_number(page_number)
        
        if frame_number is None:
            # Page fault - page not in physical memory
            self.page_faults += 1
            
            # Use replacement algorithm to find a frame to replace
            if self.replacement_algorithm is None:
                raise RuntimeError("No replacement algorithm set")
            
            frame_number = self.replacement_algorithm.select_victim_frame()
            
            # If a page is being replaced, update page table
            replaced_page = self.page_table.get_page_number(frame_number)
            if replaced_page is not None:
                self.page_table.remove_mapping(replaced_page)
            
            # Add new mapping to page table
            self.page_table.add_mapping(page_number, frame_number)
            
            # Notify algorithm of page access
            self.replacement_algorithm.page_accessed(page_number, frame_number)
        else:
            # Page hit - page already in physical memory
            # Notify algorithm of page access
            self.replacement_algorithm.page_accessed(page_number, frame_number)
        
        # Calculate physical address
        physical_address = frame_number * self.page_size + offset
        
        # Update hit ratio
        self.hit_ratio = (self.memory_accesses - self.page_faults) / self.memory_accesses
        
        # Return value at physical address
        return self.physical_memory[physical_address]
    
    def write_memory(self, virtual_address, value):
        """
        Write a value to the given virtual address.
        
        Args:
            virtual_address: The virtual address to write to
            value: The value to write
        """
        if virtual_address < 0 or virtual_address >= self.virtual_memory_size:
            raise ValueError(f"Virtual address {virtual_address} out of bounds")
        
        # Access memory first to ensure page is in physical memory
        self.access_memory(virtual_address)
        
        # Calculate page number and offset
        page_number = virtual_address // self.page_size
        offset = virtual_address % self.page_size
        
        # Get frame number from page table
        frame_number = self.page_table.get_frame_number(page_number)
        
        # Calculate physical address
        physical_address = frame_number * self.page_size + offset
        
        # Write value to physical memory
        self.physical_memory[physical_address] = value
        
        # Also write to virtual memory for simulation purposes
        self.virtual_memory[virtual_address] = value
    
    def get_metrics(self):
        """Return current performance metrics."""
        return {
            'page_faults': self.page_faults,
            'memory_accesses': self.memory_accesses,
            'hit_ratio': self.hit_ratio
        }
    
    def reset_metrics(self):
        """Reset performance metrics."""
        self.page_faults = 0
        self.memory_accesses = 0
        self.hit_ratio = 0.0