class Meter:
    def __init__(self):
        self.count = 0
        self.data = {}

    def update(self, x):
        for k, v in x.items():
            self.data[k] = self.data.get(k, 0.0) + v.detach().cpu()
        self.count += 1
    
    def avg(self):
        ret = {}
        for k, v in self.data.items():
            ret[k] = v / self.count
        return ret
    