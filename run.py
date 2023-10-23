import os
import sys
import argparse

import logging

from code.log_conf import *
from code.load_files import *
from code.device import *
from code.attack import *
from code.flow import *

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
    required=False,
    default="ntv",
    help="type: (n)ame, (t)ype, (v)endor (only for device mode)"
)
parser.add_argument(
    "-r", "--reset",
    action="store_true",
    required=False,
    help="reset flows"
)

parser.add_argument(
    "-d", "--debug",
    action="store_true",
    required=False,
    help="debug mode"
)

if __name__ == "__main__":
    logger.info(f"Starting...")
    args = parser.parse_args()
    
    # flow 데이터 생성
    if not os.path.exists("./data/") or args.reset:
        logger.info(f"Creating Flow files...")
        
        # 학습용 csv 불러오기
        train_folders = []
        train_pcaps = []

        for train_folder in os.listdir("./train"):
            if os.path.isdir("./train/" + train_folder + "/"):
                train_folders.append(load_files("./train/" + train_folder + "/"))
        for train_folder in train_folders:
            for pcap in train_folder:
                train_pcaps.append(pcap)
        logger.info(f"Loaded train pcaps - {len(train_pcaps)}")

        # 검증용 csv 불러오기
        valid_folders = []
        valid_pcaps = []

        for valid_folder in os.listdir("./valid"):
            if os.path.isdir("./valid/" + valid_folder + "/"):
                valid_folders.append(load_files("./valid/" + valid_folder + "/"))
        for valid_folder in valid_folders:
            for pcap in valid_folder:
                valid_pcaps.append(pcap)
        logger.info(f"Loaded valid pcaps - {len(valid_pcaps)}")

        # 테스트용 csv 불러오기
        test_folders = []
        test_pcaps = []

        for test_folder in os.listdir("./test"):
            if os.path.isdir("./test/" + test_folder + "/"):
                test_folders.append(load_files("./test/" + test_folder + "/"))
        for test_folder in test_folders:
            for pcap in test_folder:
                test_pcaps.append(pcap)
        logger.info(f"Loaded test pcaps - {len(test_pcaps)}")

        # flow 생성
        logger.info(f"Creating flows...")
        flows = Flows()

        for pcap in train_pcaps:
            logger.debug(f"Creating flows from {pcap}...")
            for idx, pkt in pcap.iterrows():
                if idx > 5000:
                    break
                flow_key = FlowKey()

                if not flow_key.set_key(pkt):
                    continue
                
                key = flows.find(flow_key)

                flow_value = FlowValue()
                flow_value.set_raw_value(pkt, flow_key)

                if key is None:
                    flows.create(flow_key, flow_value, True)
                else:
                    flows.append(key[0], flow_value, key[1])
        logger.info(f"Created flows - {len(flows.value)}")

        # valid flow 생성
        logger.info(f"Creating valid flows...")
        valid_flows = Flows()

        for pcap in valid_pcaps:
            for pkt in pcap.iterrows():
                flow_key = FlowKey()

                if not flow_key.set_key(pkt):
                    continue

                key = valid_flows.find(flow_key)

                flow_value = FlowValue()
                flow_value.set_raw_value(pkt, flow_key)

                if key is None:
                    valid_flows.create(flow_key, flow_value, True)
                else:
                    valid_flows.append(key[0], flow_value, key[1])
        logger.info(f"Created valid flows - {len(valid_flows.value)}")

        # test flow 생성
        logger.info(f"Creating test flows...")
        test_flows = Flows()

        for pcap in test_pcaps:
            for pkt in pcap.iterrows():
                flow_key = FlowKey()

                if not flow_key.set_key(pkt):
                    continue

                key = test_flows.find(flow_key)

                flow_value = FlowValue()
                flow_value.set_raw_value(pkt, flow_key)

                if key is None:
                    test_flows.create(flow_key, flow_value, True)
                else:
                    test_flows.append(key[0], flow_value, key[1])
        logger.info(f"Created test flows - {len(test_flows.value)}")

        # flow 정렬 및 튜닝
        logger.info(f"Sorting and tuning all flows...")
        flows.sort()
        valid_flows.sort()
        test_flows.sort()
        flows.tune()
        valid_flows.tune()
        test_flows.tune()
        logger.info(f"Sorted and tuned all flows.")

        # flow 저장
        logger.info(f"Saving flow files...")
        save_flows(flows, "./data/flows.pkl")
        save_flows(valid_flows, "./data/valid_flows.pkl")
        save_flows(test_flows, "./data/test_flows.pkl")
        logger.info(f"Saved flow files.")
    else:
        logger.info(f"Loading flow files...")
        flows = load_flows("./data/flows.pkl")
        valid_flows = load_flows("./data/valid_flows.pkl")
        test_flows = load_flows("./data/test_flows.pkl")
        logger.info(f"Loaded flow files.")

        logger.info(f"train flows: {len(flows.value)}")
        logger.info(f"valid flows: {len(valid_flows.value)}")
        logger.info(f"test flows: {len(test_flows.value)}")

    # 라벨링 데이터 불러오기
    logger.info(f"Loading labels...")
    if args.mode == "d":
        labels = load_device_labels("./labels/device_label.csv")
    elif args.mode == "a":
        labels = load_attack_labels("./labels/attack_label.csv")
    else:
        logger.error(f"Invalid mode: {args.mode}")
        exit(1)
    logger.info(f"Loaded labels.")
    logger.info(f"labels: {len(labels)}")

    # 디버그 모드
    if args.debug:
        logger.info(f"Debug mode on.")

        flows = under_sampling(flows, 34)

    # 학습
    logger.info(f"Learning...")
    algorithms = []
    for arg in args.algorithm:
        if arg in algorithms:
            continue
        if arg == "d":
           algorithms.append("dt")
           continue
        if arg == "f":
            algorithms.append("rf")
            continue
        if arg == "r":
            algorithms.append("rnn")
            continue
        if arg == "l":
            algorithms.append("lstm")
            continue
        logger.error(f"Invalid algorithm: {arg}")
        exit(1)

    mode = []
    if args.mode == "d":
        for arg in args.type:
            if arg in mode:
                continue
            if arg == "n":
                mode.append("name")
                continue
            if arg == "t":
                mode.append("type")
                continue
            if arg == "v":
                mode.append("vendor")
                continue
            logger.error(f"Invalid type: {arg}")
            exit(1)
        
        for algorithm in algorithms:
            for m in mode:
                model = device_learn(flows, valid_flows, labels, m, algorithm)
                device_evaluate(test_flows, labels, m, algorithm, model)
        logger.info(f"Done.")
    elif args.mode == "a":
        for algorithm in algorithms:
            model = attack_learn(flows, valid_flows, labels, algorithm)
            attack_evaluate(test_flows, labels, algorithm, model)
    else:
        logger.error(f"Invalid mode: {args.mode}")
        exit(1)