from collections import deque


class CallStack:
    def __init__(self):
        self.stack = deque()
    
    def register(self, fun):
        def wrapped_fun(*args, **kwargs):
            try:
                self.stack.append(fun)
                return fun(*args, **kwargs)
            finally:
                self.stack.pop()
        return wrapped_fun

    def __len__(self):
        return len(self.stack)
