import numpy as np
from torch.utils import data

np.random.seed(3)

# MAML dataloader for SHD
class dataset_numpy(data.Dataset):
    ''' Numpy based generator
    '''
    def __init__(self, spikes, labels, extra, 
                 n_way, k_shot, excluded_labels):

        self.nb_steps = 100
        self.nb_units = 700
        self.max_time = 1.4 
        self.n_way = n_way               # Number of support classes per task
        self.k_shot = k_shot             # Number of support examples per class
        self.k_query = k_shot            # Number of query examples per classs

        self.firing_times = np.array(spikes['times'])
        self.units_fired  = np.array(spikes['units'])
        self.speakers = np.array(extra['speaker'], dtype=np.uint8)
        self.num_samples = self.firing_times.shape[0]
        self.time_bins = np.linspace(0, self.max_time, num=self.nb_steps)

        self.input  = np.zeros((self.num_samples, self.nb_steps, self.nb_units), dtype=np.uint8)
        self.output = np.array(labels, dtype=np.uint8)

        self.num_all_classes = np.unique(self.output).shape[0]
        self.num_all_speakers = np.unique(self.speakers).shape[0]
        self.excluded_labels = excluded_labels
        self.included_labels = np.setdiff1d(np.arange(self.num_all_classes), 
                                             self.excluded_labels)
        
        self.generate_input_spike_tensor()
        self.reduce_input_dim(target_dim=256, axis=2)

        # Calculate category locations in SHD
        self.catlocs = tuple()
        for cat in np.unique(self.output):
            self.catlocs += (np.argwhere(self.output == cat).reshape(-1),)

        # Calculate number of samples for meta-training
        num_excluded_inputs = np.sum([np.sum(self.output == c) for c in self.excluded_labels])
        self.num_samples = self.input.shape[0] - num_excluded_inputs

    def __len__(self):
        return int(self.num_samples)

    def generate_input_spike_tensor(self):
        for idx in range(self.num_samples):
            times = np.digitize(self.firing_times[idx], self.time_bins)
            units = self.units_fired[idx] 
            self.input[idx, times, units] = 1

    def reduce_input_dim(self, target_dim, axis):
        sample_ind = int(np.ceil(self.nb_units / target_dim))
        index = [np.arange(i, 700, sample_ind) for i in range(sample_ind)]
        reshaped = [np.take(self.input, index[i], axis) for i in range(sample_ind)]
        reshaped = [np.pad(reshaped[i], [(0,0), (0,0), (0,int(target_dim-reshaped[i].shape[2]))], mode='constant') for i in range(sample_ind)]
        reshaped = np.concatenate(reshaped, axis=0)
        self.input = reshaped
        self.output = np.tile(self.output, sample_ind)

    def __getitem__(self, idx):
        x1, y1, x2, y2 = self.__data_generation(idx)
        return x1, y1, x2, y2

    def __data_generation(self, index):
        ''' Return single task 
                sX, sY(N_class, K examples)
                qX, qY(N_class, Q examples)
        '''
        cats = np.random.choice(self.included_labels, self.n_way, replace=False)
        
        shot, query = [], []
        for c in cats:
            c_shot, c_query = [], []
            idx_list = np.random.choice(self.catlocs[c], 
                                         self.k_shot + self.k_query, 
                                         replace=False)
            support_idx, query_idx = idx_list[:self.k_shot], idx_list[-self.k_query:]
            for idx in support_idx:
                c_shot.append(self.input[idx])
            for idx in query_idx:
                c_query.append(self.input[idx])
            shot.append(np.stack(c_shot))
            query.append(np.stack(c_query))

        shot_x  = np.concatenate(shot, axis=0)     # [n_way * n_shot, t, i]
        query_x = np.concatenate(query, axis=0)    # [n_way * n_shot, t, i]
        shot_y  = np.repeat(cats, self.k_shot)     # [n_way * n_shot]
        query_y = np.repeat(cats, self.k_query)    # [n_way * n_shot]

        return shot_x, shot_y, query_x, query_y

def numpy_collate(batch):
  if isinstance(batch[0], np.ndarray):
    return np.stack(batch)
  elif isinstance(batch[0], (tuple,list)):
    transposed = zip(*batch)
    return [numpy_collate(samples) for samples in transposed]
  else:
    return np.array(batch)

class NumpyLoader(data.DataLoader):
  def __init__(self, dataset, batch_size=1,
                shuffle=False, sampler=None,
                batch_sampler=None, num_workers=0,
                pin_memory=False, drop_last=False,
                timeout=0, worker_init_fn=None):
    super(self.__class__, self).__init__(dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=numpy_collate,
        pin_memory=pin_memory,
        drop_last=drop_last,
        timeout=timeout,
        worker_init_fn=worker_init_fn)
    
