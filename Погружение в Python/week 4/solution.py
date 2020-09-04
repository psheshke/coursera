import os.path
import tempfile

class File():
    
    def __init__(self, path_to_file):
        
        self.path_to_file = path_to_file
        
        if os.path.exists(self.path_to_file):
            self.f = open(self.path_to_file, 'r')
        else:
            self.f = open(self.path_to_file, 'w+')
        self.f.close()
            
    def read(self):
        with open(self.path_to_file, 'r') as f:
            return f.read()
    
    def write(self, str):
        with open(self.path_to_file, 'w') as f:
            f.write(str)
            
    def __add__(self, obj):
        filename = str(abs(hash(self.path_to_file)+hash(obj.path_to_file)))
        path_to_file = os.path.join(tempfile.gettempdir(), filename)
        
        new_file = File(path_to_file)
        text1 = self.read()
        text2 = obj.read()
        f= open(new_file.path_to_file, 'w')
        f.write(text1+text2)
        f.close
        return new_file
    
    def __str__(self):
        return self.path_to_file
    
    def __getitem__(self, index):
        with open(self.path_to_file, 'r') as f:
            lines = f.readlines()
        return lines[index]
    
    def __exit__(self, *args):
        self.f.close