import sys
import signal
from scapy.all import sniff
from collections import defaultdict
import os
from threading import Thread
import pandas as pd


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--serv_addr')
args = parser.parse_args()

serv_addr = args.serv_addr.split(':')
port = int( serv_addr[1] )
transfer = 0


def handler(signum, frame):
    print("Transferred data: ", transfer)
    sys.exit(0)

signal.signal(signal.SIGINT, handler)


def process_packet(packet):
    global transfer
    try:
        # get the packet source & destination IP addresses and ports
        packet_connection = (packet.sport, packet.dport)
    except (AttributeError, IndexError):
        # sometimes the packet does not have TCP/UDP layers, we just ignore these packets
        pass
    else:
        #print(packet_connection)
        if((packet_connection[0]== port) or (packet_connection[1]==port)):
            print("packet found", packet_connection[0], packet_connection[1])
            transfer += len(packet)
            print(transfer)

# start sniffing
print("Started sniffing")
sniff(prn=process_packet, iface="lo", store=False)