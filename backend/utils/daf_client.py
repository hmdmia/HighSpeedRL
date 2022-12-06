# daf_client.py - Communicate with DAF code running in MATLAB

# DAF comm protocol:
#
#        Python                                MATLAB
#        ------                                ------
#                           - Ready Mode -
#
#  "RUNSIM,<runner_name>,<index>,<params_json>" -->
#                                <-- "PARAMS,<params_json>"
#
#                     - Run Mode, (multiple exchanges) -
#
#                                <-- "STATE,<state_data>"
#          "ACTION,<action_data>" -->
#                             - Sim Terminates -
#                                <-- "OK"
#
#                             - End Mode -
#
#                         "EXIT" -->


import os
import json
import logging
import time
import socket
import numpy as np

log = logging.getLogger('a4h')


class DafClient:
    # Create single definition file common between MATLAB and Python

    # Messages:
    #    Format: "[{<header>}, {<payload>}]
    #    Header: "'type':<msg_type>"
    #    Payload: "<name1>:<value2>, <name2>:<value2>, ...'

    # Message types with payload descriptions
    RUNSIM = "RUNSIM"  # <runner_name>,<sim_time><incomming_parameters>
    PARAMS = "PARAMS"  # <outgoing_parameters>
    STATE = "STATE"    # <var_name1>:<var_value1>...
    ACTION = "ACTION"  # <action_number>
    EXIT = "EXIT"      # (none)
    OK = "OK"          # (none)
    ERROR = "ERROR"    # <error_info>

    # Reserved header field names
    MTYPE = 'type'

    # Reserved payload field names
    RUNNER = 'runner'
    SIM_TIME = 'sim_time'
    ACTION_NUM = 'action_num'

    DAF_HOST = 'localhost'  # Currently only supporting local connections
    BUFFER_SIZE = 1024
    COMM_RETRY = 25
    COMM_WAIT = 1  # secs

    def __init__(self, port=None):
        if not port:
            with open('port') as f:
                port = int(next(f).split()[0])

        log.info('Using port: ' + str(port))
        self.port = port
        self.sock = None

    def debug(_, txt):
        log.debug('DEBUG: '+txt)

    def send(self, mtype, payload={}):
        """Encode message and transmit on socket connection"""
        msg_str = json.dumps([{self.MTYPE: mtype}, payload], cls=NpEncoder)
        self.debug('COMM|PY->|' + msg_str)
        self.sock.sendall(msg_str.encode())

    def receive(self):
        """Receive data via socket connection and decode message"""
        msg_str = self.sock.recv(self.BUFFER_SIZE)  # TODO Read all data?
        self.debug('COMM|PY<-|' + str(msg_str))
        return json.loads(msg_str.decode())

    def run_sim(self, runner, sim_secs, send_params={}):
        """Ask DAF to run simulation, send our params, receive DAF params"""
        payload = {self.RUNNER: runner, self.SIM_TIME: sim_secs}
        payload.update(send_params)
        rec_params = ''

        for i in range(self.COMM_RETRY):

            try:

                if not self.sock:
                    self.sock = socket.socket(socket.AF_INET,
                                              socket.SOCK_STREAM)
                    self.sock.connect((self.DAF_HOST, self.port))

                self.send(self.RUNSIM, payload)
                rec_params = self.receive()
                break

            except Exception:

                # Might be waiting for MATLAB startup or busy DAF, wait, retry
                if i < self.COMM_RETRY-1:
                    log.debug('DAF comm retry #%d...' % (i+1))
                    self.sock.close()
                    self.sock = []
                    time.sleep(self.COMM_WAIT)
                else:
                    log.error('ERROR: Failed to communicate with DAF')
                    exit(1)

        return rec_params[1]

    def get_state(self):
        [_, state] = self.receive()
        return state

    def send_action(self, action):
        self.send(self.ACTION, {self.ACTION_NUM: action})
        return

    def exit_daf(self):
        """Tell DAF we're done"""
        self.send(self.EXIT)
        log.debug('Requested DAF exit')
        return


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
