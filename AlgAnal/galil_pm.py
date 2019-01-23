from math import ceil
from random import randint, random, seed
K=4

def newp1(x, y, p, q, s, p1, q1, p2, q2):
  # find the maximum overlap with shift p1 starting at position s
  while True:
    while s + p1 + q1 < len(x) and x[s + p1 + q1] == x[s + q1]:
      q1 += 1
    if p1 + q1 >= K * p1: # overlap occured 3 or more times
      p2, q2 = q1, 0
      params = x, y, p, q, s, p1, q1, p2, q2
      return newp2(*params)
    if s + p1 + q1 == len(x):
      params = x, y, p, q, s, p1, q1, p2, q2
      return search(*params)
    p1, q1 = p1 + max(1, int(ceil(q1/K))), 0

def newp2(x, y, p, q, s, p1, q1, p2, q2):
  # find the maximum overlap with shift p2 starting at position s
  while True:
    while s + p2 + q2 < len(x) and x[s + p2 + q2] == x[s + q2] and p2 + q2 < K*p2:
      q2 += 1

    params = x, y, p, q, s, p1, q1, p2, q2
    if p2 + q2 == K * p2: # overlap occured exactly 3 times
      return parse(*params)
    if s + p2 + q2 == len(x): # overlap continued till last character of x
      return search(*params)
    if q2 == p1 + q1: 
      p2, q2 = p2 + p1, q2 - p1
    else:
      p2, q2 = p2 + max(1, int(ceil(q2/K))), 0

def parse(x, y, p, q, s, p1, q1, p2, q2):
    while x[s+p1+q1] == x[s+q1]:
      q1 += 1
    while p1+q1 >= K*p1:
      s, q1 = s+p1, q1 - p1
    p1, q1 = p1 + max(1, int(ceil(q1/K))), 0
    params = x, y, p, q, s, p1, q1, p2, q2
    if p1 < p2:
      return parse(*params)
    else:
      return newp1(*params)
    
def search(x, y, p, q, s, p1, q1, p2, q2):
  ret = []
  while p + s <= len(y):
    while s+q < len(x) and p+s+q < len(y) and y[p+s+q] == x[s+q]:
      q+=1
    if q == len(x) - s and all([y[t+p]==x[t] for t in range(0,s)]):
      ret.append(p)
    if q == p1 + q1:
      p,q = p+p1, q-p1
    else:
      p, q = p + max(1, int(ceil(q/K))), 0
  return ret



def generate_text_containing_pattern(vocab, pat, p, text_len):
  text = ''
  while len(text) < text_len:
    if random() < p:
      text += pat
    else:
      text += vocab[randint(0,len(vocab)-1)]
  return text



if __name__ == "__main__":
  seed(12345)
  p, q = 0, 0
  s, p1, q1 = 0, 1, 0
  p2, q2 = 0, 0
  vocab = ''.join([chr(ord('a')+t) for t in range(8)])
  x = 'abc'*5 + 'de'
  y = generate_text_containing_pattern(vocab, x, 0.05, 100)
  print(f'x: {x}\ny: {y}')
  params = x, y, p, q, s, p1, q1, p2, q2
  positions = newp1(*params)
  positions_naive = [t for t in range(len(y)-len(x)+1) if y[t:t+len(x)]==x]
  print(f'      positions: {positions}')
  
  print(f'naive positions: {positions_naive}')
  # for p in positions:
  #   print(y[p:p+len(x)])
  