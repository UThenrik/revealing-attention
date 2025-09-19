import h5py
import numpy as np
from collections import namedtuple

# Named tuple for storing transition data
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 
                         'frame_id', 'gaze_pos', 'done'))

# HDF5 replay buffer with RAM caching for efficient sampling
class HDF5ReplayBufferRAM:
    def __init__(self, file_path, initial_capacity, state_shape, action_shape, gaze_shape, train_val_split=0.8, RAM_ratio=1/8):
        """Initialize the HDF5 replay buffer."""
        self.file_path = file_path
        self.capacity = int(initial_capacity)
        self.index = 0
        self.size = 0
        self.train_val_split=train_val_split

        self.train_flags=np.random.rand(self.size)<self.train_val_split
        self.size_RAM=int(self.size*RAM_ratio)
        self.file_RAM={}

        self.shuffle_RAM()

        compression_type = 'gzip'
        chunk_size = 1

        # Open the HDF5 file and create resizable datasets
        self.file = h5py.File(file_path, 'w')
        
        # Create datasets with compression
        self.file.create_dataset(
            'state', shape=(initial_capacity, *state_shape),
            maxshape=(None, *state_shape), dtype=np.uint8,
            compression=compression_type, chunks=(chunk_size, *state_shape)
        )
        self.file.create_dataset(
            'action', shape=(initial_capacity, *action_shape),
            maxshape=(None, *action_shape), dtype=np.int32,
            compression=compression_type, chunks=(chunk_size, *action_shape)
        )
        self.file.create_dataset(
            'next_state', shape=(initial_capacity, *state_shape),
            maxshape=(None, *state_shape), dtype=np.uint8,
            compression=compression_type, chunks=(chunk_size, *state_shape)
        )
        self.file.create_dataset(
            'reward', shape=(initial_capacity,), maxshape=(None,),
            dtype=np.float32, compression=compression_type
        )
        self.file.create_dataset(
            'frame_id', shape=(initial_capacity,), maxshape=(None,),
            dtype=h5py.string_dtype(encoding='utf-8'), compression=compression_type
        )
        self.file.create_dataset(
            'gaze_pos', shape=(initial_capacity, *gaze_shape),
            maxshape=(None, *gaze_shape), dtype=np.float32, compression=compression_type
        )
        self.file.create_dataset(
            'done', shape=(initial_capacity,), maxshape=(None,),
            dtype=np.bool_, compression=compression_type
        )

    def shuffle_RAM(self):
        if self.size_RAM<self.size or not self.file_RAM:
            n_RAM_shelves=int(np.floor(self.size/self.size_RAM))
    
            i_RAM_start=np.random.randint(n_RAM_shelves)*self.size_RAM
    
            self.file_RAM={}
    
            for key in self.file.keys():
                if key != 'next_state':
                    self.file_RAM[key]=self.file[key][i_RAM_start:i_RAM_start+self.size_RAM]
            self.file_RAM['train_flags']=self.train_flags[i_RAM_start:i_RAM_start+self.size_RAM]

    def clear_RAM(self):
        self.file_RAM={}
    
    def shuffle_train_val(self):
        rng = np.random.default_rng(seed=42)
        self.train_flags = rng.random(self.size) < self.train_val_split

    def push(self, state, action, next_state, reward, frame_id, gaze_pos, done):
        """Add a new transition to the buffer."""
        self.file['state'][self.index] = state
        self.file['action'][self.index] = action
        self.file['next_state'][self.index] = next_state
        self.file['reward'][self.index] = reward
        self.file['frame_id'][self.index] = frame_id
        self.file['gaze_pos'][self.index] = gaze_pos
        self.file['done'][self.index] = done

        self.index = (self.index + 1) % self.capacity  # Circular buffer
        self.size = min(self.size + 1, self.capacity)

    def sample_stacked(self, num_transitions, stack_size=4,consecutive=False,return_idxs=False):
        """Read a sequence of consecutive transitions from the buffer with stacked frames along the last dimension."""
        
        file_RAM=self.file_RAM
        size_RAM=self.size_RAM

        if consecutive:
            start_i= np.random.randint(size_RAM-stack_size-num_transitions)+stack_size
            idxs = np.arange(start_i,start_i+num_transitions)
        else:
            idxs = np.random.randint(size_RAM-stack_size,size=num_transitions)+stack_size

        all_states = file_RAM['state']
        state_shape = all_states.shape[1:-1]  # Shape of a single state frame *without* channels
        channels=all_states.shape[-1]
        stacked_states = np.empty((num_transitions, *state_shape, channels, stack_size), dtype=all_states.dtype)
        next_stacked_states = np.empty_like(stacked_states)

        for i in range(num_transitions):
            for j in range(stack_size):
                stacked_states[i, ..., :, j] = all_states[idxs[i] - stack_size + 1 + j]
                next_stacked_states[i, ..., :, j] = all_states[np.clip(idxs[i] + 1, a_min=0, a_max=size_RAM - 1) - stack_size + 1 + j]


        actions = file_RAM['action'][idxs]
        rewards = file_RAM['reward'][idxs]
        frame_ids = file_RAM['frame_id'][idxs].astype(str)
        gaze_positions = file_RAM['gaze_pos'][idxs]
        dones = file_RAM['done'][idxs]
        train_flags=self.train_flags[idxs]

        if return_idxs:
            return stacked_states, actions, next_stacked_states, rewards, frame_ids, gaze_positions, dones, train_flags, idxs
        else:
            return stacked_states, actions, next_stacked_states, rewards, frame_ids, gaze_positions, dones, train_flags

    def sample_stacked_fwd(self, num_transitions, stack_size=4,consecutive=False,return_idxs=False):
        """Read a sequence of consecutive transitions from the buffer with stacked frames along the last dimension."""

        file_RAM=self.file_RAM
        size_RAM=self.size_RAM
        
        if consecutive:
            start_i= np.random.randint(size_RAM-stack_size-num_transitions)+stack_size
            idxs = np.arange(start_i,start_i+num_transitions)
        else:
            idxs = np.random.randint(size_RAM-stack_size,size=num_transitions)+stack_size

        all_states = file_RAM['state']
        state_shape = all_states.shape[1:-1]  # Shape of a single state frame *without* channels
        channels=all_states.shape[-1]
        stacked_states = np.empty((num_transitions, *state_shape, channels, stack_size), dtype=all_states.dtype)

        for i in range(num_transitions):
            for j in range(stack_size):
                stacked_states[i, ..., :, j] = all_states[idxs[i] - stack_size + 1 + j]

        actions = file_RAM['action'][idxs]
        rewards = file_RAM['reward'][idxs]
        frame_ids = file_RAM['frame_id'][idxs].astype(str)
        gaze_positions = file_RAM['gaze_pos'][idxs]
        dones = file_RAM['done'][idxs]
        train_flags=self.train_flags[idxs]

        if return_idxs:
            return stacked_states, actions, rewards, frame_ids, gaze_positions, dones, train_flags, idxs
        else:
            return stacked_states, actions, rewards, frame_ids, gaze_positions, dones, train_flags
    def sample_stacked_fwd_val(self, num_transitions, stack_size=4, consecutive=False, return_idxs=False):
        """Read a sequence of consecutive transitions from the buffer with stacked frames along the last dimension, only for validation set."""
        file_RAM = self.file_RAM
        size_RAM = self.size_RAM

        # Get indices where train_flags is False (validation set)
        val_indices = np.where(self.train_flags[:size_RAM] == False)[0]
        # Exclude indices that are too close to the start for stacking
        val_indices = val_indices[val_indices >= stack_size]

        if len(val_indices) < num_transitions:
            raise ValueError("Not enough validation transitions available.")

        if consecutive:
            # Find possible start indices for consecutive blocks
            possible_starts = []
            for i in range(len(val_indices) - num_transitions + 1):
                block = val_indices[i:i+num_transitions]
                if np.all(np.diff(block) == 1):
                    if np.all(block >= stack_size):
                        possible_starts.append(i)
            if not possible_starts:
                raise ValueError("No consecutive validation blocks available.")
            start_idx = np.random.choice(possible_starts)
            idxs = val_indices[start_idx:start_idx+num_transitions]
        else:
            idxs = np.random.choice(val_indices, size=num_transitions, replace=False)

        all_states = file_RAM['state']
        state_shape = all_states.shape[1:-1]
        channels = all_states.shape[-1]
        stacked_states = np.empty((num_transitions, *state_shape, channels, stack_size), dtype=all_states.dtype)

        for i in range(num_transitions):
            for j in range(stack_size):
                stacked_states[i, ..., :, j] = all_states[idxs[i] - stack_size + 1 + j]

        actions = file_RAM['action'][idxs]
        rewards = file_RAM['reward'][idxs]
        frame_ids = file_RAM['frame_id'][idxs].astype(str)
        gaze_positions = file_RAM['gaze_pos'][idxs]
        dones = file_RAM['done'][idxs]
        train_flags = self.train_flags[idxs]

        if return_idxs:
            return stacked_states, actions, rewards, frame_ids, gaze_positions, dones, train_flags, idxs
        else:
            return stacked_states, actions, rewards, frame_ids, gaze_positions, dones, train_flags



    def __len__(self):
        return self.size

    def close(self):
        """Close the HDF5 file."""
        self.file.close()

    @staticmethod
    def load(file_path, file_rw_option='r', train_val_split=0.8, RAM_ratio=1/8):
        """
        Load an existing buffer for sampling.

        Args:
            file_path (str): Path to the HDF5 file.

        Returns:
            HDF5ReplayBuffer: A buffer instance with the file opened in read mode.
        """
        cache_size = 100 * 1024 * 1024  # 10 MB cache size, you can adjust as needed
        cache_params = {
            'rdcc_nbytes': cache_size,     # Cache size in bytes
            'rdcc_w0': 0.75,               # Caching algorithm: balance between writing and caching
            'rdcc_nslots': 5000            # Number of slots in the cache
        }

        buffer = HDF5ReplayBufferRAM.__new__(HDF5ReplayBufferRAM)
        buffer.file_path = file_path
        buffer.file = h5py.File(file_path, file_rw_option, **cache_params)  # Open in read-only mode
        buffer.capacity = len(buffer.file['state'])
        buffer.index = 0
        buffer.train_val_split=train_val_split
        buffer.size = buffer.capacity
        buffer.size_RAM=int(buffer.size*RAM_ratio)
        buffer.file_RAM={}
        buffer.shuffle_train_val()
        buffer.shuffle_RAM()

        return buffer
