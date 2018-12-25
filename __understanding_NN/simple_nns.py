import _dynet as dy
import matplotlib.pyplot as plt
import numpy as np
import seaborn
from random import shuffle
from collections import namedtuple

def derlu(x):
    return -dy.rectify(-dy.rectify(x + 1) + 2) + 1

# creating test cases
def circle_iid(radius, n_points):
    """
    draw points unifrom within a circle in the 2D plane
    """
    radii = np.sqrt(np.random.uniform(high=radius ** 2, size=n_points))
    phi = np.random.uniform(high=2 * np.pi, size=n_points)
    return np.array([radii*np.cos(phi),radii * np.sin(phi)])

def one_hot(n, k):
    ret = np.zeros(n, np.float)
    ret[k] = 1
    return ret

class abstract_network:
    def __init__(self):
        self.pc = dy.ParameterCollection()
        self.params = {}
        self.last_output = None
        self.with_bias = None

    def eval_loss(self, case_x, case_class):
        output = self.evaluate_network(case_x)
        y_param = dy.vecInput(2)
        y_param.set(one_hot(2,case_class))
        loss = dy.squared_distance(output, y_param)
        return loss

    def last_case_class(self):
        return np.argmax(self.last_output_value())

    def last_output_value(self):
        return self.last_output.value()

    def evaluate_network(self, t):
        return NotImplementedError
    
    def train_network(self, train_data, epochs = 3):
        trainer = dy.AdamTrainer(self.pc)
        i = 0
        mloss = 0.
        goods = 0.
            
        for e in range(epochs):
            shuffle(train_data)
            for x, y in train_data:
                i = i + 1
                loss = self.eval_loss(x, y)
                good = y == self.last_case_class()
                #print y, self.last_output_value(), np.argmax(self.last_output_value()), self.last_case_class()
                mloss += loss.value()
                goods += int(good)
                loss.backward()
                trainer.update()
        print("average loss: {} acc: {}".format(mloss/i, goods/i))
        
    def plot_2d_layer(self, layer_ind=0):
        W = self.params["W"][layer_ind].npvalue()
        b = self.params["b"][layer_ind].npvalue()
        #if inverse and np.diff(W.shape)[0] ==0:
        W = np.linalg.inv(W)
        if self.with_bias[layer_ind]:
            origin = b
        else:
            origin = np.zeros(2)
        vec_1 = W[:,0]
        vec_2 = W[:,1]
        x1 = np.array([0, vec_1[0]]) - origin[0]
        x2 = np.array([0, vec_2[0]]) - origin[0]

        y1 = np.array([0, vec_1[1]]) - origin[1]
        y2 = np.array([0, vec_2[1]]) - origin[1]
        plt.plot(x1, y1, 'k-', linewidth = 1.2)
        plt.plot(x2, y2, 'k-', linewidth = 1)


class single_layer(abstract_network):
    """
    derlu(W*(x+b))
    """
    def __init__(self, with_bias = True):
        abstract_network.__init__(self)
        self.params.update({
            "W": [self.pc.add_parameters((2,2), name='W0', init='uniform', scale = 0.7)],
            "b": [self.pc.add_parameters(2, name='b0', init='uniform', scale = 0.7)]
        })
        self.with_bias = [with_bias]
    
    def evaluate_network(self, t):
        dy.renew_cg()
        W = self.params["W"][0]
        b = self.params["b"][0]
        x = dy.vecInput(2)
        x.set(t)
        
        if self.with_bias[0]:
            output = derlu(W * (x + b))
        else:
            output = derlu(W * x)
        self.last_output = output
        return output

    def eval_list(self, pts):
        return [self.evaluate_network(t).npvalue() for t in pts]


class train_data():
    def __init__(self, data_x=None, data_y=None):
        self.data_x = data_x
        self.data_y = data_y

    def data(self):
        return zip(self.data_x, self.data_y)
    
    def points_x(self):
        return [p[0] for p in self.data_x]

    def points_y(self):
        return [p[1] for p in self.data_x]

    def plot(self):
        classes = sorted(list(set(self.data_y)))
        plt.plot([x for x, cc in zip(self.points_x(),self.data_y) if cc == classes[0]],
            [x for x, cc in zip(self.points_y(),self.data_y) if cc == classes[0]],
            'r.')
        plt.plot([x for x, cc in zip(self.points_x(),self.data_y) if cc == classes[1]],
            [x for x, cc in zip(self.points_y(),self.data_y) if cc == classes[1]],
            'b.')
        plt.legend(['class '+str(c) for c in classes[:2]])
    

class patch_train_data(train_data):
    patch_spec = namedtuple('spotty_spec',['centroid','noise','count','case_class'])
    def __init__(self, specs):
        train_data.__init__(self)
        data_x = []
        data_y = []
        for spec in specs:
            points = circle_iid(spec.noise, spec.count) + spec.centroid[:, np.newaxis]
            for i in range(points.shape[1]):
                data_x.append(points[:,i])
                data_y.append(spec.case_class) 
        self.data_x = data_x
        self.data_y = data_y

            

"""
def create_spot(centroids, noise, count):
    coord = []
    for i in range(centroids.shape[1]):
        coord.append(circle_iid(noise, count) + centroids[:,i, np.newaxis])
    return np.concatenate(coord, axis=1)

def create_train_data(z_centroids, o_centroids, noise, count):
    train_data = []
    z = create_train_x(z_centroids, noise, count)
    o = create_train_x(o_centroids, noise, count)
    print o.shape
    for i in range(z.shape[1]):
        train_data.append((z[:,i], 0))
    for i in range(o.shape[1]):
        train_data.append((o[:,i], 1))
    return train_data
"""