import os

import logging

logger = logging.getLogger("logger")

from datetime import datetime

def convert_to_timestamp(date_string):
    # 주어진 형식에 따라 문자열을 datetime 객체로 변환
    dt = datetime.strptime(date_string, "%Y-%m-%dT%H:%M:%S.%f")
    
    # datetime 객체를 타임스탬프로 변환
    timestamp = dt.timestamp()
    
    return timestamp

class FlowKey:
    def __init__(self):
        self.sid = None
        self.did = None
        self.protocol = None
        self.additional = None
    
    def __str__(self):
        return str(self.sid) + '_' + str(self.did) + '_' + str(self.protocol) + '_' + str(self.additional)

    def set_key(self, pkt):
        try:
            self.sid = pkt["ipaddr SRC_IP"]
            self.did = pkt["ipaddr DST_IP"]
            self.protocol = 'TCP'
            self.additional = (pkt["uint16 SRC_PORT"], pkt["uint16 DST_PORT"])
            return True
        except:
            return False


class FlowValue:
    def __init__(self):
        self.raw_time = None
        self.direction = None
        self.length = None
        self.delta_time = None
        self.protocol = None
    
    def __str__(self):
        return str(self.raw_time) + '_' + str(self.direction) + '_' + str(self.length) + '_' + str(self.delta_time) + '_' + str(self.protocol)
    
    def __repr__(self):
        return str(self.raw_time) + '_' + str(self.direction) + '_' + str(self.length) + '_' + str(self.delta_time) + '_' + str(self.protocol)

    def set_raw_value(self, pkt, flow_key):
        self.protocol = flow_key.protocol
        
        self.raw_time = pkt["time TIME_LAST"]
        self.length = pkt.length

class Flows:
    def __init__(self):
        self.value = {}
    
    def __str__(self):
        return str(self.value)
    
    def find(self, key):
        try:
            for k in self.value:
                if k.protocol == "ZBEE_NWK":
                    if k.protocol == key.protocol and k.additional == key.additional:
                        if k.sid == key.sid and k.did == key.did:
                            return k, True
                        elif k.sid == key.did and k.did == key.sid:
                            return k, False
                elif k.protocol == "TCP" or k.protocol == "UDP":
                    if k.protocol == key.protocol:
                        if k.sid == key.sid and k.did == key.did and k.additional == key.additional:
                            return k, True
                        elif k.sid == key.did and k.did == key.sid and k.additional[::-1] == key.additional:
                            return k, False
        except:
            return None
                
        return None
    
    def create(self, key, value, direction):
        value.direction = direction
        self.value[key] = [value]
    
    def append(self, key, value, direction):
        value.direction = direction
        self.value[key].append(value)
    
    def sort(self):
        for k in self.value:
            self.value[k].sort(key=lambda x: x.raw_time)

    def tune(self):
        for k in self.value:
            self.value[k][0].delta_time = 0

            for i in range(1, len(self.value[k])):
                self.value[k][i].delta_time = self.value[k][i].raw_time - self.value[k][i-1].raw_time
    
    def print(self, path):
        for k in self.value:
            with open(path + str(k) + ".txt", 'w') as f:
                for i in range(len(self.value[k])):
                    f.write(str(self.value[k][i].delta_time) + ' ' + str(self.value[k][i].direction) + ' ' + str(self.value[k][i].length) + '\n')