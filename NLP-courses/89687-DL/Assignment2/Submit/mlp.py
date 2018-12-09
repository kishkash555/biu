
import _dynet as dy
import time
import numpy as np

EMB = 50
INPUT = EMB*5
INITIAL_LEARN_RATE = 0.15 # can be changed by argv[1]
HIDDEN = 400  # can be changed by argv[2]
SET_RANDOM_SEED = True # can be changed by argv[3]
REG_LAMBDA = 0.00 # can be changed by argv[5]

global MIN_ACC 
MIN_ACC = 0.77
UNK = '**UNK**'
START = '**START**'
STOP = '**STOP**'
NUMBER = '**NUM**'


PREFIX_MARK = '^^'
SUFFIX_MARK = '$$'
MIN_LENGTH_FOR_PRE_SUF = 5

def create_network_params(nwords, ntags, external_E = None):
    # create a parameter collection and add the parameters.
    print("adding parameters")
    m = dy.ParameterCollection()
    
    print("nwords: {}".format(nwords))
    E = m.add_lookup_parameters((nwords,EMB), name='E')
    if external_E and sum(external_E.shape) > 0:
        assert external_E.shape[1] == EMB
        external_rows = external_E.shape[0]
        for r in range(external_rows):
            E.init_row(r, external_E[r,:])
 
    b = m.add_parameters(HIDDEN, name='b')
    U = m.add_parameters((ntags, HIDDEN), name='U')
    W = m.add_parameters((HIDDEN, INPUT), name='W')
    bp = m.add_parameters(ntags, name='bp')
    dy.renew_cg()
    return m, E, b, U, W, bp

def build_network(params, x_data):
    _, E, b, U, W, bp = params
    if type(x_data) == dict:
        # print("DICT")
        prefix_ordinals = x_data['prefix']
        suffix_ordinals = x_data['suffix']
        x_ordinals = x_data['fullwords']
    else:
        prefix_ordinals = None
        suffix_ordinals = None
        x_ordinals = x_data
    x = dy.concatenate([E[ord] for ord in x_ordinals])
    if prefix_ordinals:
        x_pre = dy.concatenate([E[ord] for ord in prefix_ordinals])
        x = x + x_pre
    if suffix_ordinals:
        x_suf = dy.concatenate([E[ord] for ord in suffix_ordinals])
        x = x + x_suf
    output = dy.softmax(U * (dy.tanh(W*x + b)) + bp)
    return output

def train_network(params, ntags, train_data, dev_set, telemetry_file, randstring, very_common_tag = -1):
    global  MIN_ACC
    prev_acc = 0
    m = params[0]
    t0 = time.clock()
    # train the network
    trainer = dy.SimpleSGDTrainer(m)
    total_loss = 0
    seen_instances = 0
    train_good = 0
    very_common_tag_count = 0
    for x_data, train_y in train_data:
        dy.renew_cg()
        output = build_network(params, x_data)
        # l2 regularization did not look promising at all, so it's commented out
        loss = -dy.log(output[train_y])  #+ REG_LAMBDA * sum([dy.l2_norm(p) for p in params[2:]])
        if train_y == np.argmax(output.npvalue()):
            train_good +=1
        seen_instances += 1
        total_loss += loss.value()
        loss.backward()
        trainer.update()
        if seen_instances % 20000 == 0:
            # measure elapsed seconds
            secs = time.clock() - t0
            t0 = time.clock()
            good = case = 0
            max_dev_instances = 70*1000
            dev_instances = 0
            for x_tuple, dev_y in dev_set:
                output =  build_network(params, x_tuple)
                y_hat = np.argmax(output.npvalue())
                case +=1
                if  y_hat == dev_y and y_hat == very_common_tag:
                    case -= 1 # don't count this case
                    very_common_tag_count += 1
                elif y_hat == dev_y:
                    good +=1
                
                dev_instances += 1
                if dev_instances >= max_dev_instances:
                    break
            acc = float(good)/case
            print("iterations: {}. train_accuracy: {} accuracy: {} avg loss: {} secs per 1000:{}".format(seen_instances, float(train_good)/20000, acc, total_loss / (seen_instances+1), secs/20))
            train_good = 0
            if acc > MIN_ACC and acc > prev_acc:
                print("saving.")
                dy.save("params_"+randstring,list(params)[1:])
                prev_acc = acc
            
            telemetry_file.write("{}\t{}\t{}\t{}\n".format(seen_instances, acc, total_loss / (seen_instances+1), secs/20))
            print("very common tag count: {}".format(very_common_tag_count))
    MIN_ACC = max(prev_acc, MIN_ACC) 
