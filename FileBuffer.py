class FileBuffer():

    def __init__(self, data_reference, buffer = 15000, padding = 5000):

        self.data_reference = data_reference
        self.current_index = 0
        self.length = len(self.data_reference)
        self.buffer = buffer
        self.padding = padding

        print(f'Creating buffer for {data_reference}')

        self.load_buffer(self.current_index)

    def load_buffer(self, starting_index):

        num_items = self.buffer
        if starting_index + num_items > len(self.data_reference):
            num_items = len(self.data_reference) - starting_index

        self.buffered_data = self.data_reference[starting_index:starting_index+num_items]

        self.current_index = starting_index
    
    def __getitem__(self, index):

        if (index < self.current_index).any() or (index > self.current_index + self.buffer).any():
            raise IndexError(f'Index out of bounds of the current buffer ({self.current_index} < {index} < {self.current_index + self.buffer})')
        
        if (index > self.current_index + self.padding).all():
            self.load_buffer(self.current_index + self.padding)
            # print(f'Loading buffer to {self.current_index}')

        index -= self.current_index # Reduce 

        return self.buffered_data[index]