import ast
import json
import pickle as pkl

import numpy as np
import zmq


class CarlaServer(object):
    def __init__(self, num_clients=10):
        self.context = zmq.Context()
        url = "tcp://*:45000"
        self.server = self.context.socket(zmq.ROUTER)
        self.server.bind(url)

        self.client_info = []
        for i in range(num_clients):
            self.client_info.append(bytes("", "utf-8"))

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
            if self.client_info[client_data["client"]] == bytes("", "utf-8"):
                self.client_info[client_data["client"]] = client_identity
            if np.sum(clients_check) == len(self.client_info):
                break

        if isinstance(data, list):
            data = np.array(data)
        return data

    def send_data(self, client_idx, client_data):
        # pkl.dump 사용시 byte type이라  recv_json으로 받기 어렵다.
        # json.dump 사용시 ndarray라서 객체가 json으로 변환되지 않는다.
        client_data = pkl.dumps(client_data)
        self.server.send_multipart([self.client_info[client_idx], client_data])
        return


class CarlaClient(object):
    def __init__(self, name: int, ip, port=45000):
        self.name = name
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.DEALER)
        self.socket.connect("tcp://{}:{}".format(ip, port))

    def send_data(self, client_data):
        self.socket.send_json({"client": self.name, "data": client_data})
        return

    def recv_data(self):
        byte_data = self.socket.recv()
        data = pkl.loads(byte_data)
        return data


if __name__ == "__main__":
    print("Carla Server open")
    server = CarlaServer(num_clients=10)
    data = server.recv_data()
    server.send_data(0, [{"a": "b"}])
