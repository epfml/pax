import os
import numpy as np

class Registry:
    def __init__(self):
        self.registry = {}
    
    def register(self, identifier, implementation):
        self.registry[identifier] = implementation
    
    def __call__(self, identifier):
        if identifier in self.registry:
            return self.registry[identifier]
        else:
            keys = list(self.registry.keys())
            rel_distances = [levenshtein_edit_distance(identifier, key) / len(key) for key in keys]
            indices = np.argsort(rel_distances)[:8]
            options = "\n".join(["- " + str(keys[i]) for i in indices])
            raise ValueError(f"Could not find '{identifier}'. Did you mean any of:\n" + options)
    
    def __setitem__(self, identifier, implementation):
        self.register(identifier, implementation)
    
    def __getitem__(self, identifier):
        if identifier in self.registry:
            return self(identifier)
        else:
            super()(identifier)
    
    def list(self):
        return list(self.registry.keys())

def levenshtein_edit_distance(seq1, seq2):
    """
    Copied from Frank Hofmann, https://stackabuse.com/levenshtein-distance-and-text-similarity-in-python/
    """
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros ((size_x, size_y))
    for x in range(size_x):
        matrix [x, 0] = x
    for y in range(size_y):
        matrix [0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix [x,y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1],
                    matrix[x, y-1] + 1
                )
            else:
                matrix [x,y] = min(
                    matrix[x-1,y] + 1,
                    matrix[x-1,y-1] + 1,
                    matrix[x,y-1] + 1
                )
    return matrix[size_x - 1, size_y - 1]

task = Registry()
model = Registry()
dataset = Registry()
config = Registry()

config["data_root"] = os.path.join(os.getenv("HOME"), "pax_data")
