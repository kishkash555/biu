class rk_hash:
    def __init__(self, d,q):
        self.d = d # the polynom basis 
        self.q = q # the prime number defining the field
        #self.H = pow(d,m-1) % q
        self.current_hash = 0
        self.current_string = []

    def add_char(self,char_code):
        h, d = self.current_hash, self.d
        self.current_hash = (d * h + char_code) % self.q
        self.current_string.append(char_code)
        return self.current_hash

    def remove_first(self):
        char_to_remove = self.current_string.pop(0)
        self.current_hash -= pow(self.d, len(self.current_string))*char_to_remove
        self.current_hash = self.current_hash % self.q
        return self.current_hash

    def remove_last(self):
         char_to_remove = self.current_string.pop()
         self.current_hash -= char_to_remove
         self.current_hash = self.current_hash
    
    def roll_char(self, char_code):
        self.add_char(char_code)
        return self.remove_first()
    
    

def rk_patter_match(text, pattern, d, q):
    n = len(text)
    m = len(pattern)
    h = pow(d,m-1)%q
    p = 0
    t = 0
    result = []
    for i in range(m): # preprocessing
        p = (d*p+ord(pattern[i]))%q
        t = (d*t+ord(text[i]))%q
    for s in range(n-m+1): # note the +1
        if p == t: # check character by character
            match = True
            for i in range(m):
                if pattern[i] != text[s+i]:
                    match = False
                    break
            if match:
                result = result + [s]
        if s < n-m:
            t = (t-h*ord(text[s]))%q # remove letter s
            t = (t*d+ord(text[s+m]))%q # add letter s+m
            t = (t+q)%q # make sure that t >= 0
    return result



def Rabin_Karp_Matcher(text, pattern, d, q):
    n = len(text)
    m = len(pattern)
    h = pow(d,m-1)%q # the shift of the mth character
    p = 0
    t = 0
    result = []
    for i in range(m): # preprocessing
        p = (d*p+ord(pattern[i]))%q
        t = (d*t+ord(text[i]))%q
    for s in range(n-m+1): # note the +1: ensures last position is tested
        if p == t: # check character by character
            match = True
            for i in range(m): # "cheat": if probablitstic algorithm finds a match, we verify it deterministically
                if pattern[i] != text[s+i]:
                    match = False
                    break
            if match:
                result = result + [s]
        if s < n-m:
            t = (t-h*ord(text[s]))%q # remove s'th letter of text
            t = (t*d+ord(text[s+m]))%q # add letter s+m
            t = (t+q)%q # make sure that t >= 0
    return result
