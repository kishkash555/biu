from ex4 import calc_cer
from collections import OrderedDict
import numpy as np

d = OrderedDict((c,i) for i,c in enumerate(' abcdefghijklmnopqrstuvwxyz'))

a = np.array([[5,5,5,16,8,9,0]]) 
b = np.array([[5,26,8,9,0,0,0]])

calc_cer(a,b,[4],d)
