from contextlib import contextmanager


class StackCounter:
    def __init__(self):
        self.value = 0
    
    @contextmanager
    def increment(self):
        prev_value = self.value
        try:
            self.value += 1
            yield 
        finally:
            self.value = prev_value
