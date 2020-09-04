import time
import socket

class ClientError (Exception):
    pass

class Client():
    
    def __init__(self,  adress, port, timeout = None):
        self.adress = adress
        self.port = port
        self.timeout = timeout
        
        
    def get(self, metric):
        
        msg = f'get {metric}\n'
        
        with socket.create_connection((self.adress, self.port), self.timeout) as sock:
            try:
                sock.sendall(msg.encode("utf8"))
                resp = sock.recv(1024).decode()
                ans = {}
                if resp == "ok\n\n":
                    return ans
                else:
                    if resp.split('\n')[0] == 'ok':
                        try:
                            
                            for r in resp.split('\n')[1:]:
                                if r != '':
                                    metric, value, timestamp = r.split(' ')[0], r.split(' ')[1], r.split(' ')[2]
                                    ans.setdefault(metric,  []).append((int(timestamp), float(value)))
                                    
                            if len(ans.keys()) != 0:
                                for k in ans.keys():
                                    ans[k].sort()
                                    
                            return ans
                        except:
                            raise ClientError()
                    else:
                        raise ClientError()
            except socket.error:
                raise ClientError() 
    
    def put(self, metric, value, timestamp = None):
        
        if not timestamp:
            timestamp = int(time.time())
        
        msg = f'put {metric} {value} {timestamp}\n'
        
        with socket.create_connection((self.adress, self.port), self.timeout) as sock:
            try:
                sock.sendall(msg.encode("utf8"))
                resp = sock.recv(1024).decode()
                if resp != "ok\n\n":
                    raise ClientError() 
            except socket.error:
                raise ClientError() 