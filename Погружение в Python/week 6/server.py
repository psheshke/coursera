import asyncio

data_storage = {}

def run_server(host, port):
    
    loop = asyncio.get_event_loop()
    coro = loop.create_server(
        ClientServerProtocol,
        host, port
    )

    server = loop.run_until_complete(coro)

    loop.run_forever()

    server.close()
    loop.run_until_complete(server.wait_closed())
    loop.close()
    

class ClientServerProtocol(asyncio.Protocol):
    
    def connection_made(self, transport):
        self.transport = transport

    def data_received(self, data):
        resp = self.process_data(data.decode())
        self.transport.write(resp.encode())

    def process_data(self, req):
        err_msg = 'error\nwrong command\n'
        ok_msg = 'ok\n'
        metric, value, timestamp = None, None, None
        cmd = req.split()
        if cmd == []:
            res = err_msg
        else:
            if cmd[0] == 'put':
#                 print(f'cmd: {cmd}')
                if len(cmd) == 4:
#                     print('put 1')
                    metric, value, timestamp = cmd[1], cmd[2], cmd[3]
                    try:
                        value = float(value)
                        timestamp = int(timestamp)
                        if data_storage.setdefault(metric,  []) == []:
#                             print('put 2')
                            data_storage.setdefault(metric,  []).append((int(timestamp), float(value)))
                        else:
#                             print('put 3')
                            F = True
                            for n, c in enumerate(data_storage.setdefault(metric,  [])):
                                if c[0] == timestamp:
#                                     print('put 4')
                                    F = False
                                    data_storage.setdefault(metric,  [])[n] = (int(timestamp), float(value))
                            if F:
                                data_storage.setdefault(metric,  []).append((int(timestamp), float(value)))
                        res = ok_msg
                    except:
#                         print('put 5')
                        res = err_msg
                else:
#                     print('put 6')
                    res = err_msg
            elif cmd[0] == 'get':
                if len(cmd) == 2:
                    if cmd[1] in ['*']:
                        res = 'ok\n'
                        for m in data_storage.keys():
                            for c in data_storage[m]:
                                res += f'{m} {c[1]} {c[0]}\n'
                    else:
                        if data_storage.get(cmd[1], []) == []:
                            res = ok_msg
                        else:
                            res = 'ok\n'
                            for c in data_storage[cmd[1]]:
                                res += f'{cmd[1]} {c[1]} {c[0]}\n'
                            
                else:
                    res = err_msg
            else:
                res = err_msg
                
#         print(data_storage)
        return res+'\n'
    
# if __name__ == "__main__":
#     run_server("127.0.0.1", 8885)