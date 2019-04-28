class Clustering(object):
    def fit(self, X, cluster_num=2):
        self.membership = None
        
    def get_membership(self):
        return self.membership