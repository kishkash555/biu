import numpy as np

class triad:
    def __init__(self, df, group, signal_type):
        cols = [c for c in df.columns if c[0]==group and c[1].startswith(signal_type)]
        #print(cols)
        cosine_simi = np.corrcoef(df[cols].values.T)
        angular_dist = np.arccos(cosine_simi)/np.pi
        self.corrs = sorted([angular_dist[0,1], angular_dist[1,2], angular_dist[2,0]])
        
        self._triangle_vertices()
    
    def _triangle_vertices(self):
        corrs = self.corrs
        a,b,c = corrs[1], corrs[2], corrs[0]
        cos_alpha = (b**2 + c**2 - a**2)/(2*b*c)
        if cos_alpha > 1:
            print(corrs)
            raise ValueError("Triangle vertices encountered miscalculation")
        sin_alpha = np.sqrt(1-cos_alpha**2)
        A = np.zeros(2)
        C = np.array((b*cos_alpha, b*sin_alpha))
        B = np.array((c,0))
        self.vertices = np.vstack([A,B,C,A])
    
    @property
    def point(self):
        return np.array([ self.corrs[0], (self.corrs[1]+self.corrs[2])/2 ])
    
    @property
    def triangle(self):
        return self.vertices
    
    def to_triangle(self, plt=False):
        if plt:
            return self.vertices[:,0], self.vertices[:,1]
        return self.vertices
    
    @property
    def triangle_ratio(self):
        p = self.point
        return p[1]/p[0]