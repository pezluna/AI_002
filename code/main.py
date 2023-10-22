import os
import sys
import argparse

import logging

from log_conf import *
from load_files import *
from code.device import *
from evaluate import *
from flow import *

init_logger()
logger = logging.getLogger("logger")

parser = argparse.ArgumentParser()

parser.add_argument(
    "-m", "--mode", 
    type=str, 
    required=True, 
    help="mode: (d)evice, (a)ttack"
)
parser.add_argument(
    "-a", "--algorithm", 
    type=str, 
    action="store",
    required=False,
    default="dfrl",
    help="algorithm: (d)ecision tree, random (f)orest, (r)nn, (l)stm"
)
parser.add_argument(
    "-t", "--type", 
    type=str, 
    action="store",
    required=False,
    default="",
    help="type: (n)ame, (t)ype, (v)endor (only for device mode)"
)
parser.add_argument(
    "-r", "--reset",
    action="store_true",
    required=False,
    help="reset flows"
)

if __name__ == "__main__":
    logger.info(f"Starting...")
    args = parser.parse_args()
    
    # flow 및 test flow 파일 존재 여부 확인
    # 없으면 생성
    if not os.path.exists("../data/flows.pkl") or args.reset:
        logger.info(f"Flows not found. Creating flows...")
        # 학습용 pcap 로드
        pcaps_by_folder = []

        for folder in os.listdir("../train/"):
            if os.path.isdir("../train/" + folder + "/"):
                pcaps_by_folder.append(load_files("../train/" + folder + "/"))

        train_pcaps = []
        for pcaps_in_folder in pcaps_by_folder:
            for pcap in pcaps_in_folder:
                train_pcaps.append(pcap)

        logger.info(f"Loaded {len(train_pcaps)} pcaps for training.")

        # 검증용 pcap 로드
        pcaps_by_folder = []

        for folder in os.listdir("../valid/"):
            if os.path.isdir("../valid/" + folder + "/"):
                pcaps_by_folder.append(load_files("../valid/" + folder + "/"))
        
        valid_pcaps = []
        for pcaps_in_folder in pcaps_by_folder:
            for pcap in pcaps_in_folder:
                valid_pcaps.append(pcap)

        logger.info(f"Loaded {len(valid_pcaps)} pcaps for validation.")

        # 테스트용 pcap 로드
        pcaps_by_folder = []

        for folder in os.listdir("../test/"):
            if os.path.isdir("../test/" + folder + "/"):
                pcaps_by_folder.append(load_files("../test/" + folder + "/"))

        test_pcaps = []
        for pcaps_in_folder in pcaps_by_folder:
            for pcap in pcaps_in_folder:
                test_pcaps.append(pcap)

        logger.info(f"Loaded {len(test_pcaps)} pcaps for testing.")

        # flow 생성
        logger.info(f"Creating flows...")
        flows = Flows()
        for pcap in train_pcaps:
            for pkt in pcap:
                flow_key = FlowKey()
                if not flow_key.set_key(pkt):
                    continue

                flow_value = FlowValue()
                flow_value.set_raw_value(pkt, flow_key)

                key = flows.find(flow_key)

                if key is None:
                    flows.create(flow_key, flow_value, True)
                else:
                    flows.append(key[0], flow_value, key[1])

        logger.info(f"Created {len(flows.value)} flows.")

        # valid flow 생성
        logger.info(f"Creating valid flows...")
        valid_flows = Flows()
        for pcap in valid_pcaps:
            for pkt in pcap:
                flow_key = FlowKey()
                if not flow_key.set_key(pkt):
                    continue

                flow_value = FlowValue()
                flow_value.set_raw_value(pkt, flow_key)

                key = valid_flows.find(flow_key)

                if key is None:
                    valid_flows.create(flow_key, flow_value, True)
                else:
                    valid_flows.append(key[0], flow_value, key[1])
                    
        logger.info(f"Created {len(valid_flows.value)} valid flows.")

        # test flow 생성
        logger.info(f"Creating test flows...")
        test_flows = Flows()
        for pcap in test_pcaps:
            for pkt in pcap:
                flow_key = FlowKey()
                if not flow_key.set_key(pkt):
                    continue

                flow_value = FlowValue()
                flow_value.set_raw_value(pkt, flow_key)

                key = test_flows.find(flow_key)

                if key is None:
                    test_flows.create(flow_key, flow_value, True)
                else:
                    test_flows.append(key[0], flow_value, key[1])

        logger.info(f"Created {len(test_flows.value)} test flows.")
        
        # flow 정렬 및 튜닝
        logger.info(f"Sorting and tuning flows...")
        flows.sort()
        flows.tune()
        test_flows.sort()
        test_flows.tune()
        valid_flows.sort()
        valid_flows.tune()
        logger.info(f"Sorted and tuned flows.")

        # flow 저장
        logger.info(f"Saving flows...")
        save_flows(flows, "../data/flows.pkl")
        save_flows(test_flows, "../data/test_flows.pkl")
        save_flows(valid_flows, "../data/valid_flows.pkl")
        logger.info(f"Saved flows.")
    else:
        logger.info(f"Loading flows...")
        flows = load_flows("../data/flows.pkl")
        valid_flows = load_flows("../data/valid_flows.pkl")
        test_flows = load_flows("../data/test_flows.pkl")
        logger.info(f"Loaded flows.")

    # label 데이터 불러오기
    logger.info(f"Loading labels...")
    labels = load_device_labels("../labels/testbed.csv")
    logger.info(f"Loaded {len(labels)} labels.")

    # 모델 생성
    model_list = ["rnn", "lstm"]
    mode_list = ["name", "dtype", "vendor"]

    for model_type in model_list:
        for mode in mode_list:
            model = device_learn(flows, valid_flows, labels, mode, model_type)

            device_evaluate(test_flows, labels, mode, model_type, model)

    logger.info(f"Done.")