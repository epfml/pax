class Registry:
    def __init__(self):
        self.registry = {}
    
    def register(self, identifier, implementation):
        self.registry[identifier] = implementation
    
    def __call__(self, identifier):
        if identifier in self.registry:
            return self.registry[identifier]
        else:
            raise ValueError(f"Could not find {identifier}")
    
    def list(self):
        return list(self.registry.keys())

task = Registry()
model = Registry()
dataset = Registry()
