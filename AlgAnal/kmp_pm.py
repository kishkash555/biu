class KMP:
    def __init__(self, pattern):
        """ Calculate partial match table: String -> [Int]"""
        ret = [0]
        
        for i in range(1, len(pattern)):
            j = ret[i - 1]
            pi = pattern[i]
            pj = pattern[j]
            while j > 0 and pi != pj:
                j = ret[j - 1]
                pj = pattern[j]
            ret.append(j + 1 if pattern[j] == pattern[i] else j)
        self.partial = ret
        self.P = pattern
        
    def search(self, T):
        """ 
        KMP search main algorithm: String -> String -> [Int] 
        Return all the matching position of pattern string P in S
        """
        P, partial, ret, j = self.P, self.partial, [], 0
        
        for i in range(len(T)):
            ti = T[i]
            pj = P[j]
            matched = T[i-j : i+1]
            while j > 0 and T[i] != P[j]:
                j = partial[j - 1]
                pj = P[j]
            matched = T[i-j : i+1]
            if ti == pj: j += 1
            # matched = T[max(0,i-j) : i+1]
            if j == len(P): 
                ret.append(i - (j - 1))
                j = partial[j - 1]
            
        return ret


if __name__ == "__main__":
    kmp = KMP('ababababcccddd')
    kmp.search('abababababababcccdddaaaabbbccdddaaaaa')