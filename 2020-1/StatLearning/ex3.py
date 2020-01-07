import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1234)


class grid_mrf:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.values = np.random.rand(rows,cols) > 0.5 
        self.values_y = None

    def get_point_energies(self, row, col, return_res=False):
        neighbors = []
        if row > 0:
            neighbors.append(self.values[row-1,col])
        if row < self.rows-1:
            neighbors.append(self.values[row+1, col])
        if col > 0:
            neighbors.append(self.values[row,col-1])
        if col < self.cols-1:
            neighbors.append(self.values[row,col+1])
        
        res = sum(neighbors), len(neighbors)-sum(neighbors)
        # the probability of a 1: exp(sum of neighbors==1) / (exp(neighbors==1)+ exp(neighbors==0))
        p_pos = np.exp(res[0])/(np.exp(res[0]) + np.exp(res[1]))
        if return_res:
            return res
        return p_pos
    
    def get_point_energies_with_y(self,row,col):
        res = self.get_point_energies(row,col,True)
        e1 = np.exp(res[0] - 0.5*(1-self.values_y[row,col])**2)
        e0 = np.exp(res[1] - 0.5*self.values_y[row,col]**2)
        return e1/(e1+e0)

    def gibbs_traverse(self, ncycles = 20000, buffer_size=20000, with_y=False):
        value_history_buffer = np.zeros((self.rows, self.cols,buffer_size), dtype=bool)
        for i in range(ncycles):
            points = np.arange(self.rows*self.cols,dtype=int)
            np.random.shuffle(points)
            for point in points:
                row = int(np.floor(point/self.rows))
                col = np.mod(point,self.cols)
                if with_y:
                    p_pos = self.get_point_energies_with_y(row, col)
                else:
                    p_pos = self.get_point_energies(row, col)
                    
                new_val = np.random.rand() < p_pos
                self.values[row,col] = new_val
            value_history_buffer[:,:,i % buffer_size] = self.values
        return value_history_buffer

    def genernate_y(self):
        y = np.random.randn(self.rows,self.cols)
        y = y + self.values.astype(float)
        self.values_y = y
        

    def grid_energy(self,x_values,y_values):
        hor_neighbors = np.logical_not(np.logical_xor(x_values[:,:self.cols-1],x_values[:,1:])).sum()
        ver_neighbors = np.logical_not(np.logical_xor(x_values[:self.rows-1,:],x_values[1:,:])).sum()
        y_energy = (-0.5*(x_values-y_values)**2).sum()
        return np.exp(hor_neighbors + ver_neighbors + y_energy)


    def marginal_state_probability(self):
        res = np.zeros((self.rows*self.cols,2)) # a grid to keep the energies
        values_y = self.values_y
        ncells = (self.rows*self.cols)
        for i in range(2**ncells):
            l = list(map(lambda b: bool(int(b)), "{0:025b}".format(i)))
            values_x = np.array(l).reshape((self.rows, self.cols))
            e = self.grid_energy(values_x,values_y)
            for j in range(ncells):
                res[j,int(l[j])] += e
            if i % 10000 ==0:
                print(i)
        return res



if __name__ == "__main__":
    gr = grid_mrf(5,5)
    gr.values = np.ones((5,5),dtype=bool)
    gr.gibbs_traverse()


