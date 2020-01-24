# cording: utf-8

class cycle:
    def __init__(self, list):
        self.i = 0
        self.list = list
 
    def next(self):
        self.i = (self.i + 1) % len(self.list)
        return self.list[self.i]
 
    def previous(self):
        self.i = (self.i - 1 + len(self.list)) % len(self.list)
        return self.list[self.i]

    def forward(self, j, k):
        self.i = j % len(self.list)
        out = [self.next() for _ in range(k)]
        return out
    
    def backward(self, j, k):
        self.i = j % len(self.list)
        out = [self.previous() for _ in range(k)]
        return out
 
    def present(self):
        return self.list[self.i]
 
    def set_present(self, j):
        self.i = j % len(self.list)
        return

if __name__ == "__main__":
    a = cycle([i for i in range(1,21)])
    print(a.present())
    print(a.forward(17,10))
