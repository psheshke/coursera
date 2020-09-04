class FileReader:
    def __init__(self, path = None):
        self.path = path
       
    def read(self):
        try:
            with open(self.path, 'r') as file:
                text = file.read()
                
        except FileNotFoundError:
            text = ''
        
        return text