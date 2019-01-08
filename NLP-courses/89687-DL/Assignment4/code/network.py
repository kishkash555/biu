import _dynet as dy
import numpy as np
from random import shuffle, randint

def now_string():
    import datetime as dt
    tm = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return tm

EVALUATE_LOSS_EVERY = 50000
UPDATE_EVERY = 4
MIN_SAVE_ACC = 0.5
START_SAVE_AFTER = 250000
SAVE_TO = '../save/network'
SAVE_REPORT_TO = '../save/report'
DROPOUT_RATE = 0.25
class network:
    @classmethod
    def __init__(self):
        self.pc = None
        

    def load(self, fname):
        raise NotImplementedError()

    def eval_loss(self, *args):
        raise NotImplementedError()

    def params_iterable(self, *args):
        raise NotImplementedError()
    
    def save(self, basefile):
        dy.save(basefile, self.params_iterable())

    def train_network(self, train_data, epochs = 3, dev_data = None):
        trainer = dy.AdagradTrainer(self.pc,0.05)
        i = 0
        mloss = 0.
        goods = 0.
        loss = []
        dy.renew_cg()
 
        max_dev_acc = MIN_SAVE_ACC
        run_id = randint(0,9999)
        save_path = "{}{:04d}".format(SAVE_TO,run_id)
        report_path = "{}{:04d}.txt".format(SAVE_REPORT_TO,run_id)
        rprt = open(report_path,'wt')
        print report_path
        for e in range(epochs):
            shuffle(train_data)
            for x, y in train_data:
                i = i + 1
                loss = loss + [self.eval_loss(x, y, dropout=True)]
                good = y == self.last_case_class
                goods += int(good)
                if i % UPDATE_EVERY == 0:
                    losses = dy.esum(loss)
                    mloss += losses.value()
                    losses.backward()
                    trainer.update()
                    loss = []
                    dy.renew_cg()
    
                if i % EVALUATE_LOSS_EVERY == 1000:
                    goods_dev = 0.
                    j = 0
                    for d in dev_data or []:
                        dy.renew_cg()
                        j+=1
                        x, y = d
                        self.eval_loss(x, y)
                        goods_dev += 1 if y==self.last_case_class else 0
                    dev_acc = goods_dev / len(dev_data or 'a') 

                    message = "{} average loss after {} iterations: {} acc: {}".format(
                        now_string(), i, mloss/EVALUATE_LOSS_EVERY, goods/EVALUATE_LOSS_EVERY)
                    dev_acc_str = " dev acc: {}".format(dev_acc) if dev_data else ""
                    print(message + dev_acc_str)
                    rprt.write(message + dev_acc_str+'\n')
                    mloss = 0.
                    goods = 0.

                    if dev_acc > max_dev_acc and i > START_SAVE_AFTER:
                        max_dev_acc = dev_acc
                        print("saving.")
                        rprt.write("saving.\n")
                        self.save(save_path)
                rprt.flush()

class mlp_subnetwork():
    def __init__(self, pc, layer_sizes, hidden_activation, output_activation):
        self.pc = pc
        self.hidden_activation = hidden_activation
        self. output_activation = output_activation
        if len(layer_sizes):
            self.layer_sizes = layer_sizes
            self.n_layers = len(layer_sizes)
            self.input_size = layer_sizes[0]
            params = [(
                    pc.add_parameters((a,b), name = "W{:02d}".format(i)), 
                    pc.add_parameters(a, name = "b{:02d}".format(i))
                    ) for (i, a, b) in zip(range(len(layer_sizes)-1), layer_sizes[1:], layer_sizes[:-1])
                        ]
            self.params = { 
                "W": [p[0] for p in params],
                "b": [p[1] for p in params]
            }
    
    @classmethod
    def load(cls, params, pc, hidden_activation, output_activation):
        i = 0
        w = []
        b = []
        for param in params:
            if i %2 ==0:
                w.append(param)
            else:
                b.append(param)
            i+=1
        self = cls(None, [], hidden_activation, output_activation)
        self.params = {"W": w, "b": b}
        self.pc = pc
        self.layer_sizes = [x.dim()[0][0] for x in w] + [b[-1].dim()[0][0]]
        self.n_layers = len(self.layer_sizes)
        print "layer sizes", self.layer_sizes
        return self

    def check_input_size(self, x_list):
        for x_np in x_list:
            x_size = x_np.squeeze().shape[0]
            if  x_size != self.input_size:
                raise ValueError("Expected input of size {} got {}".format(self.input_size, x_size))
        
    def params_iterable(self):
        for w,b in zip(self.params["W"], self.params["b"]):
            yield w
            yield b

    def evaluate_network(self, x_np, apply_final_activation=True, dropout=False):
        """
        return an expression that is the result of feeding the input through the entire 
        network, except the last activation
        """
        #self.check_input_size(x_np)
        n_stages = self.n_layers-1

        # will be skipped for x_np that are already 
        # _dynet.__tensorInputExpression or _dynet._vecInputExpression
        if type(x_np) == np.ndarray:
            print "ndarray"
            x = dy.vecInput()
            x.set(x_np)
        elif type(x_np) == list:
            print "list"
            x = dy.inputTensor(x_np, batched = True)
        else:
            x = x_np
        final_activation = self.output_activation if apply_final_activation else lambda x: x
        activation = self.hidden_activation
        for i, W, b in zip(range(n_stages), self.params["W"], self.params["b"]):
            #print "i", i
            if i == n_stages-1:
                    # print "final layer"
                    activation = final_activation
            x = activation(W*x + b)
            if dropout:
                x = dy.dropout(x, DROPOUT_RATE)
        return x
    

    
    def eval_loss(self, x, y):
       # y_hat = self.evaluate_network(x)
        raise NotImplementedError()



