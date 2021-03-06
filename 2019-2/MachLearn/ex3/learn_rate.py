
class learn_rate_schedule:
    def __init__(self, lr_type, **params):
        if lr_type == 'exponential':
            lr_generator = self.exponential_decay()
        elif lr_type == 'inverse_time':
            lr_generator = self.inverse_time_decay()
        elif lr_type == 'constant':
            lr_generator = self.constant_lr()
        else:
            raise ValueError("unrecognized lr_type")
        if 'momentum' in params and bool(params['momentum'])==True:
            self.lr_generator = self.apply_momentum(lr_generator)
        else:
            self.lr_generator = lr_generator
        self.params = params
        self.step = 0
        self.step_width = 1
        self.trigger_now = False
        self.epoch_as_trigger = False

    def apply_momentum(self, lr_generator):
        v = 0.
        gamma = self.params['gamma']
        for st in lr_generator:
            v = gamma*v + st
            yield v

    def inverse_time_decay(self):
        step =0
        alpha = self.params['alpha']
        eta = self.params['eta']
        while True:
            yield eta/(1 + alpha* step)
            if self.do_step():
                step += 1
            

    def exponential_decay(self):
        alpha = self.params['alpha']
        eta = self.params['eta']
        if alpha >= 1.:
            raise ValueError("alpha ({}) must be <1".format(alpha))
        while True:
            yield eta
            if self.do_step():
                eta *= alpha 
    
    def constant_lr(self):
        eta = self.params['eta']
        while True:
            yield eta

    def set_step_width(self, st):
        if st == 'epoch':
            st = 0
            self.epoch_as_trigger = True
        else:
            self.epoch_as_trigger = False
        self.step_width = max(0, st)
        return self

    def do_step(self):
        self.step += 1
        is_trigger = self.trigger_now
        self.trigger_now = False
        return is_trigger or (self.step_width > 0 and self.step % self.step_width == 0)

    def _trigger_now(self):
        self.trigger_now = True

    # def set_epoch_trigger(self, trigger=True):
    #     self.epoch_as_trigger = trigger
    #     return self

    def new_epoch(self):
        if self.epoch_as_trigger:
            self._trigger_now()

