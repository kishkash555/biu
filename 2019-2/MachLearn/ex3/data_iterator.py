import numpy as np

class data_iterator:
    def __init__(self, array_x, array_y, batch_size, shuffle = True):
        self.array_x = array_x
        self.array_y = array_y.squeeze()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.length = array_x.shape[0]

    def __iter__(self):
        self.schedule = np.arange(self.length, dtype=int)
        if self.shuffle:
            np.random.shuffle(self.schedule)
        self.n = 0
        return self
    
    def __next__(self):
        if self.n == self.length:
            raise StopIteration()
        n, next_n = self.n, min(self.n + self.batch_size, self.length)
        self.n = next_n
        return self.array_x[self.schedule[n:next_n],:], self.array_y[self.schedule[n:next_n]]
