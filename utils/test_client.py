import json
import pickle as pkl

import numpy as np
import zmq


class CarlaServer(object):
    def __init__(self, num_clients=10):
        self.context = zmq.Context()
        url = "tcp://*:35000"
        self.server = self.context.socket(zmq.ROUTER)
        self.server.bind(url)

        self.client_info = []
        for i in range(num_clients):
            self.client_info.append(None)

    def recv_data(self):
        clients_check = np.zeros(10)
        data = []
        for i in range(len(self.client_info)):
            data.append(None)
        while True:
            client_identity, client_data = self.server.recv_multipart()
            client_data = json.loads(client_data)
            data[client_data["client"]] = client_data["data"]
            clients_check[client_data["client"]] = 1
            if self.client_info[client_data["client"]] is None:
                self.client_info[client_data["client"]] = client_identity
            if np.sum(clients_check) == len(self.client_info):
                break
        return data

    def send_data(self, data):
        for idx, client_data in enumerate(data):
            self.server.send_multipart([self.client_info[idx], client_data[idx]])
        return


class CarlaClient(object):
    def __init__(self, name: int, ip, port=35000):
        self.name = name
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.DEALER)
        self.socket.connect("tcp://{}:{}".format(ip, port))

    def send_data(self, client_data):
        self.socket.send_json({"client": self.name, "data": client_data})
        return

    def recv_data(self):
        data = self.socket.recv()
        return data


if __name__ == "__main__":
    client = CarlaClient(name=0, ip="127.0.0.1")
    client.send_data("kkkkk")
    print(json.loads(pkl.loads(client.recv_data())))
