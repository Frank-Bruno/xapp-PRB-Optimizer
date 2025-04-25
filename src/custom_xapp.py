# Imports form local libraries
from .asn1_defs.e2sm_kpm_rc import E2SM_KPM_RC
from .asn1_defs.e2ap_2_3 import E2AP_PDU_Descriptions

# Imports from OSC libraries
from ricxappframe.xapp_frame import RMRXapp, rmr
from mdclogpy import Logger, Level
from ricxappframe import xapp_rest, xapp_subscribe
from ricxappframe.entities.rnib.nb_identity_pb2 import NbIdentity

# Imports from other libraries
from time import sleep
from threading import Thread
import signal
import json
import requests
import numpy as np
from typing import Dict
from .env import MobNet, server_rl
import csv
from pathlib import Path
from ray import tune
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.env.policy_client import PolicyClient


class XappNori:
    """
    Custom xApp class.
    """

    def __init__(self):
        """
        Initializes the custom xApp instance and instatiates the xApp framework object.
        """

        # Initializing a logger for the custom xApp instance in Debug level (logs everything)
        self.logger = Logger(
            name="XappNori", level=Level.DEBUG
        )  # The name is included in each log entry, Levels: DEBUG < INFO < WARNING < ERROR
        # self.logger.get_env_params_values() # Getting the MDC key-value pairs from the environment
        self.logger.info("Initializing the xApp.")

        # Initializing custom control variables
        self._shutdown = False  # Stops the xApp loop if True
        self._ready = False  # True when the xApp is ready to start
        self.subscription_responses: Dict[int, Dict] = (
            {}
        )  # Stores the subscription responses for each E2 node inventory name
        self.sub_id_to_node: Dict[str, str] = (
            {}
        )  # Maps subscription IDs to E2 node inventory names

        # Instatiating the xApp framework object
        self._rmrxapp = RMRXapp(
            default_handler=self.default_rmr_handler,  # Called when no specific handler is found for an RMR message
            config_handler=self.config_change_handler,  # Called when a config change event is detected by inotify
            post_init=self.post_init,  # Called during the RMRXapp initialization, right after _BaseXapp is initialized
            rmr_port=4560,  # Port for RMR data
            rmr_wait_for_ready=True,  # Block xApp initiation until RMR is ready
            use_fake_sdl=False,  # Use a fake in-memory SDL
        )

        # RL Server
        self.env = MobNet()
        self.env_thread = Thread(target=server_rl, daemon=True)
        self.env_thread.start()


        # Registering RMR message handlers
        self._rmrxapp.register_callback(
            handler=self.ric_indication_handler, message_type=12050
        )

        # Registering a handler for terminating the xApp after TERMINATE, QUIT, or INTERRUPT signals
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGQUIT, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

        # Starting a threaded HTTP server listening to any host at port 8080
        self.http_server = xapp_rest.ThreadedHTTPServer("0.0.0.0", 8080)
        self.http_server.handler.add_handler(
            self.http_server.handler,
            method="GET",
            name="config",
            uri="/ric/v1/config",
            callback=self.config_handler,
        )
        self.http_server.handler.add_handler(
            self.http_server.handler,
            method="GET",
            name="liveness",
            uri="/ric/v1/health/alive",
            callback=self.liveness_handler,
        )
        self.http_server.handler.add_handler(
            self.http_server.handler,
            method="GET",
            name="readiness",
            uri="/ric/v1/health/ready",
            callback=self.readiness_handler,
        )
        self.http_server.handler.add_handler(
            self.http_server.handler,
            method="POST",
            name="sub_resp",
            uri="/ric/v1/subscriptions/response",
            callback=self.subscription_response_handler,
        )
        self.http_server.handler.add_handler(
            self.http_server.handler,
            method="GET",
            name="resubscribe",
            uri="/ric/v1/resubscribe",
            callback=self.resubscribe_handler,
        )
        self.logger.info("Starting HTTP server.")
        self.http_server.start()

        # xApp is ready to start
        self._ready = True
        self.logger.info("xApp is ready.")

        # RL Client
        sleep(10)
        SERVER_ADDRESS = "localhost"
        RAY_STORAGE = "./ray_results/"
        AGENT_NAME = "ppo"
        SERVER_BASE_PORT = 9900
        self.test_mode = False
        self.env.reset()
        if self.test_mode:  # Testing
            ray_storage = str(Path(RAY_STORAGE).resolve())
            analysis = tune.ExperimentAnalysis(f"{ray_storage}/{AGENT_NAME}/")
            assert analysis.trials is not None, "Analysis trial is None"
            last_checkpoint = analysis.get_last_checkpoint(analysis.trials[0])
            assert last_checkpoint is not None, "Last checkpoint is None"
            self.algo = Algorithm.from_checkpoint(last_checkpoint)
        else:  # Training
            self.client = PolicyClient(
                f"http://{SERVER_ADDRESS}:{SERVER_BASE_PORT}",
                inference_mode="local",
            )
            self.eid = self.client.start_episode(training_enabled=True)

    # ------------------ START AND STOP

    def start(self):
        """
        Starts the xApp loop.
        """

        self.log_gnbs()
        Thread(target=self.subscribe_to_e2_nodes).start()
        self._rmrxapp.run()

    def stop(self):
        """
        Terminates the xApp. Can only be called if the xApp is running in threaded mode.
        """
        self._shutdown = True
        self.unsubscribe_from_e2_nodes()
        self.logger.info(
            "Calling framework termination to unregister the xApp from AppMgr."
        )
        self._rmrxapp.stop()
        self.http_server.stop()

    # ------------------ RMRXAPP INTERNAL FUNCTIONS

    def config_change_handler(self, rmrxapp: RMRXapp, json: dict):
        """
        Handler for the config change event.
        """
        self.logger.info("Detected a config change event.")

        rmrxapp._config_data = json
        self.logger.debug("New config data: {}.".format(json))

    def post_init(self, rmrxapp: RMRXapp):
        """
        Post initialization function.
        """
        self.logger.info("Post initialization called.")

    # ------------------ E2 NODES

    def log_gnbs(self):
        """
        Logs RAN information for each registered gNB.
        """

        nbid_list = self._rmrxapp.GetListNodebIds()
        if len(nbid_list) == 0:
            self.logger.info("No gNBs registered.")
            return
        self.logger.info("Logging gNBs.")
        for nbid in nbid_list:
            self.logger.info(f"Logging NodeB info for gNB: {nbid.inventory_name}")
            global_nb_id = nbid.global_nb_id
            self.logger.info(f"PLMN ID: {global_nb_id.plmn_id}")
            self.logger.info(f"NNBID: {global_nb_id.nb_id}")
            self.logger.info(f"Connection status: {nbid.connection_status}")  # 1 ????
            self.logger.info(
                f"Health Check Timestamp Received: {nbid.health_check_timestamp_received}"
            )  # 0 ???
            self.logger.info(
                f"Health Check Timestamp Sent: {nbid.health_check_timestamp_sent}"
            )  # 0 ???

            nbinfo = self._rmrxapp.GetNodeb(nbid.inventory_name)
            self.logger.info(
                f"Associated E2T Instance Address: {nbinfo.associated_e2t_instance_address}"
            )  # E2Term IP:PORT
            self.logger.info(
                f"E2 Application Protocol: {nbinfo.e2_application_protocol}"
            )  # 0 ???
            self.logger.info(f"Failure Type: {nbinfo.failure_type}")  # 0 ???
            self.logger.info(f"Node Type: {nbinfo.node_type}")  # 2 ????
            self.logger.info(
                f"RAN Name: {nbinfo.ran_name}"
            )  # Same as the inventory name
            self.logger.info(
                f"Setup From Network: {nbinfo.setup_from_network}"
            )  # True ??

            gnb = nbinfo.gnb
            self.logger.info(f"GNB Type: {gnb.gnb_type}")  # 1 ????
            self.logger.info(
                f"Served NR Cells: {gnb.served_nr_cells}"
            )  # No cells informed

            self.logger.info("NodeB Configurations:")
            i = 1
            for config in gnb.node_configs:
                self.logger.info(f"Configuration {i}")
                self.logger.info(
                    f"e2nodeComponentInterfaceType: {config.e2nodeComponentInterfaceType}"
                )
                self.logger.info(
                    f"e2nodeComponentInterfaceTypeE1: {config.e2nodeComponentInterfaceTypeE1}"
                )
                self.logger.info(
                    f"e2nodeComponentInterfaceTypeF1: {config.e2nodeComponentInterfaceTypeF1}"
                )
                self.logger.info(
                    f"e2nodeComponentInterfaceTypeNG: {config.e2nodeComponentInterfaceTypeNG}"
                )
                self.logger.info(
                    f"e2nodeComponentInterfaceTypeS1: {config.e2nodeComponentInterfaceTypeS1}"
                )
                self.logger.info(
                    f"e2nodeComponentInterfaceTypeW1: {config.e2nodeComponentInterfaceTypeW1}"
                )
                self.logger.info(
                    f"e2nodeComponentInterfaceTypeX2: {config.e2nodeComponentInterfaceTypeX2}"
                )
                self.logger.info(
                    f"e2nodeComponentInterfaceTypeXn: {config.e2nodeComponentInterfaceTypeXn}"
                )
                self.logger.info(
                    f"e2nodeComponentRequestPart: {config.e2nodeComponentRequestPart}"
                )
                self.logger.info(
                    f"e2nodeComponentResponsePart: {config.e2nodeComponentResponsePart}"
                )
                i += 1

            self.logger.info("RAN Functions:")
            for ran_func in gnb.ran_functions:
                self.logger.info(f"RAN Function ID: {ran_func.ran_function_id}")
                self.logger.info(
                    f"RAN Function Revision: {ran_func.ran_function_revision}"
                )
                self.logger.info(f"RAN Function OID: {ran_func.ran_function_oid}")
                self.logger.info(
                    f"RAN Function Definition: {ran_func.ran_function_definition}"
                )

    def subscribe_to_e2_nodes(self):
        """
        Subscribes to all available E2 nodes.
        """
        e2_nodes = self._rmrxapp.GetListNodebIds()
        sub_trs_id = self._rmrxapp.sdl_get(
            namespace="xappnori", key="subscription_transaction_id"
        )
        if sub_trs_id is None:
            sub_trs_id = 54321
        for node in e2_nodes:
            self.logger.info(
                f"Subscribing to node {node.inventory_name}"
            )  # We use the inventory name as the node ID

            # Sending the subscription request
            subscription_req = self.generate_subscription_request(
                node.inventory_name, sub_trs_id
            )
            self.logger.debug(f"Subscription request: {subscription_req}")
            resp = requests.post(
                "http://service-ricplt-submgr-http.ricplt.svc.cluster.local:8088/ric/v1/subscriptions",
                json=subscription_req,
            )
            status = resp.status_code
            reason = resp.reason

            # Handling the subscription response
            if int(resp.status_code / 100) != 2:
                self.logger.error(
                    f"Failed to subscribe to node {node.inventory_name}. Status code: {resp.status_code}, reason: {resp.reason}"
                )
                continue
            data = (
                resp.json()
            )  # {"SubscriptionId": "my_string_id", "SubscriptionInstances": null}
            self.sub_id_to_node[data["SubscriptionId"]] = node.inventory_name
            self.subscription_responses[node.inventory_name] = data
            self.logger.debug(
                f"Subscription response from {node.inventory_name}: status = {status}, reason = {reason}, data = {data}"
            )
            self._rmrxapp.sdl_set(
                namespace="xappnori",
                key="subscription_transaction_id",
                value=sub_trs_id + 1,
            )  # Update sub_trs_id on SDL

    def unsubscribe_from_e2_nodes(self):
        """
        Unsubscribes from all subscribed E2 nodes (stored in the self.subscription_responses dict).
        """

        for sub_id in self.sub_id_to_node.keys():
            resp = requests.delete(
                f"http://service-ricplt-submgr-http.ricplt.svc.cluster.local:8088/ric/v1/subscriptions/{sub_id}"
            )
            status = resp.status_code
            reason = resp.reason
            self.logger.info(
                f"Unsubscribe from sub id {sub_id}: status = {status}, reason = {reason}"
            )

    def resubscribe_to_e2_nodes(self):
        """
        Resubscribes to all E2 nodes.
        """
        self.unsubscribe_from_e2_nodes()
        self.subscribe_to_e2_nodes()

    # ------------------ SIGNAL HANDLERS

    def _handle_signal(self, signum: int, frame):
        """
        Function called when a Kubernetes signal is received to stop the xApp execution.
        """
        self.logger.info(
            "Received signal {} to stop the xApp.".format(signal.Signals(signum).name)
        )
        self.stop()  # Custom xApp termination routine

    # ------------------ RMR MESSAGE HANDLERS

    def default_rmr_handler(self, rmrxapp: RMRXapp, summary: dict, sbuf):
        """
        Default RMR message handler.
        """
        self.logger.info("Received RMR message with summary: {}.".format(summary))
        rmrxapp.rmr_free(sbuf)  # Freeing the RMR message buffer

    def ric_indication_handler(self, rmrxapp: RMRXapp, summary: dict, sbuf):
        """
        Handler for RIC indication messages.
        """
        # self.logger.info(f"Received RIC indication message with summary: {summary}.")

        msg = summary["payload"]
        # self.logger.debug(f"Received payload from RIC indication message: {msg}")

        # Decoding the E2AP PDU data
        pdu = E2AP_PDU_Descriptions.E2AP_PDU
        pdu.from_aper(msg)
        decoded_pdu = pdu.get_val()
        e2pdu_data = {
            "ricRequestorID": decoded_pdu[1]["value"][1]["protocolIEs"][0]["value"][1][
                "ricRequestorID"
            ],
            "ricInstanceID": decoded_pdu[1]["value"][1]["protocolIEs"][0]["value"][1][
                "ricInstanceID"
            ],
            "RANfunctionID": decoded_pdu[1]["value"][1]["protocolIEs"][1]["value"][1],
            "RICactionID": decoded_pdu[1]["value"][1]["protocolIEs"][2]["value"][1],
            "RICindicationSN": decoded_pdu[1]["value"][1]["protocolIEs"][3]["value"][1],
            "RICindicationType": decoded_pdu[1]["value"][1]["protocolIEs"][4]["value"][
                1
            ],
            "RICcallProcessID": decoded_pdu[1]["value"][1]["protocolIEs"][7]["value"][
                1
            ],
        }

        # Decoding the E2SM RIC Indication data
        ric_indication_header_msg = decoded_pdu[1]["value"][1]["protocolIEs"][5][
            "value"
        ][1]
        ric_indication_message_msg = decoded_pdu[1]["value"][1]["protocolIEs"][6][
            "value"
        ][1]
        ric_indication_header = E2SM_KPM_RC.E2SM_KPM_IndicationHeader
        ric_indication_message = E2SM_KPM_RC.E2SM_KPM_IndicationMessage
        ric_indication_header.from_aper(ric_indication_header_msg)
        ric_indication_message.from_aper(ric_indication_message_msg)
        decoded_ric_indication_header = ric_indication_header.get_val()
        decoded_ric_indication_message = ric_indication_message.get_val()
        ric_indication_data = {
            "indicationHeader": decoded_ric_indication_header,
            "indicationMessage": decoded_ric_indication_message,
        }

        # self.logger.info(f"Decoded E2AP PDU data: {e2pdu_data}")
        self.logger.info(f"Decoded RIC indication header: {ric_indication_data}")
        self.logger.info(f"Decoded RIC indication data: {ric_indication_data}")

        ########### Interaction with RL Environment
        # Observation
        slice_1_avg_thr = 4 # TODO obtain the throughput information from the RIC indication message
        slice_2_avg_thr = 20
        obs = np.array([slice_1_avg_thr, slice_2_avg_thr])

        if self.test_mode:  # Testing
            action = self.algo.compute_single_action(obs, explore=False)
        else:  # Training
            action = self.client.get_action(self.eid, obs)
        assert isinstance(action, np.ndarray), "Action must be a numpy array."
        perc_action = np.floor((action / np.sum(action)) * 100)
        self.env.set_obs(obs)
        self.obs, reward, terminated, truncated, info = self.env.step(action)
        if not self.test_mode:  # Training
            self.client.log_returns(self.eid, reward, info=info)
        if terminated or truncated:
            obs, info = self.env.reset()
            if not self.test_mode:  # Training
                self.client.end_episode(self.eid, obs)
                self.eid = self.client.start_episode(training_enabled=True)
        ###################################

        # Sending the RAN slicing control message to the RIC
        for slice_id in range(2):
            sst = b"\x01" if slice_id == 1 else b"\x00"
            sd = b"\x00\x00\x00"
        slices_id = [
            {"sST": b"\x00", "sD": b"\x00\x00\x00"},
            {"sST": b"\x01", "sD": b"\x00\x00\x00"},]
        
        self.send_ran_slicing_control(
            slices_id, perc_action, rmrxapp, summary, sbuf
        )

        rmrxapp.rmr_free(sbuf)

    # ------------------ HTTP HANDLERS

    def send_ran_slicing_control(
        self, slices_id: dict, action: np.ndarray, rmrxapp: RMRXapp, summary: dict, sbuf
    ):
        """
        Sends a RAN slicing control message to the RIC.
        """

        ####### Creating RIC Control Request to control PRB slice quota
        # Encoding RIC Control Header
        asn1_control_header = E2SM_KPM_RC.E2SM_RC_ControlHeader
        rrm_policy_list = []
        for slice_id, rb_alloc in zip(slices_id, action):
            rrm_policy = {
                "rrmPolicy": {
                    "rrmPolicyMemberList": [
                        {
                            "plmnIdentity": b"\x00\x01\x02",
                            "sNSSAI": {
                                "sST": slice_id["sST"],
                                "sD": slice_id["sD"],
                            },
                        }
                    ]
                },
                "dedicatedPRBPolicyRatio": int(rb_alloc),
                "minPRBPolicyRatio": int(rb_alloc),
                "maxPRBPolicyRatio": 100,
            }
            rrm_policy_list.append(rrm_policy)
        ric_control_header = ('controlHeader-Format1', {
            "ueId":b'\x00\x01', 
            "ric-ControlStyle-Type": 1, 
            "ric-ControlAction-ID": 1, 
            "rrmPolicyList": rrm_policy_list,
        }
        )

        asn1_control_header.set_val(ric_control_header)
        coded_control_header = asn1_control_header.to_aper()

        # Encoding RIC Control Request
        asn1_pdu = E2AP_PDU_Descriptions.E2AP_PDU
        ric_request_msg = (
            "initiatingMessage",
            {
                "procedureCode": 4,
                "criticality": "ignore",
                "value": (
                    "RICcontrolRequest",
                    {
                        "protocolIEs": [
                            {
                                "id": 29,
                                "criticality": "reject",
                                "value": (
                                    "RICrequestID",
                                    {"ricRequestorID": 1003, "ricInstanceID": 1},
                                ),
                            },
                            {
                                "id": 5,
                                "criticality": "reject",
                                "value": ("RANfunctionID", 200),
                            },
                            {
                                "id": 20,
                                "criticality": "reject",
                                "value": ("RICcallProcessID", b"\x00\x01"),
                            },
                            {
                                "id": 22,
                                "criticality": "reject",
                                "value": ("RICcontrolHeader", coded_control_header),
                            },
                            {
                                "id": 23,
                                "criticality": "reject",
                                "value": ("RICcontrolMessage", b"\x00\x01"),
                            },
                        ]
                    },
                ),
            },
        )
        asn1_pdu.set_val(ric_request_msg)
        coded_pdu = asn1_pdu.to_aper()

        # Sending the RIC Control Request
        rmrxapp.rmr_rts(
            sbuf, new_payload=coded_pdu, new_mtype=12040
        )  # 12040 = RIC Control Request

        # # Decoding RIC Control Request
        # asn1_pdu.from_aper(coded_pdu)
        # decoded_pdu = asn1_pdu.get_val()
        # self.logger.info(f"\n\n\n################\nDecoded PDU: {decoded_pdu}")

        # # Decoding RIC Control Header
        # assert decoded_pdu is not None, "Decoded PDU is None"
        # coded_control = decoded_pdu[1]["value"][1]["protocolIEs"][3]["value"][1]
        # asn1_control_header.from_aper(coded_control)
        # decoded_control = asn1_control_header.get_val()
        # self.logger.info(f"\n\n\n################\nDecoded Control Header: {decoded_control}")

    def resubscribe_handler(self, name: str, path: str, data: bytes, ctype: str):
        """
        Handler for the HTTP GET /ric/v1/resubscribe request.
        """
        self.logger.info(
            "Received GET /ric/v1/resubscribe request, resubscribing to E2 Nodes"
        )
        self.resubscribe_to_e2_nodes()
        response = xapp_rest.initResponse(
            status=200, response="xApp ACKs resubscription request"  # Status = 200 OK
        )  # Initiating HTTP response
        return response

    def subscription_response_handler(
        self, name: str, path: str, data: bytes, ctype: str
    ):
        """
        Handler for the HTTP POST /ric/v1/subscriptions/response request.
        """
        sub_resp = json.loads(data.decode())
        self.logger.info(
            "Received POST /ric/v1/subscriptions/response request with data {}.".format(
                sub_resp
            )
        )
        nodeb = self.sub_id_to_node[sub_resp["SubscriptionId"]]
        self.subscription_responses[nodeb]["SubscriptionInstances"] = sub_resp[
            "SubscriptionInstances"
        ]
        response = xapp_rest.initResponse(
            status=200,  # Status = 200 OK
            response="xApp ACKs the subscription response",
        )  # Initiating HTTP response

        # Resubscribing if RtMgr did not create xApp routes
        # if sub_resp["SubscriptionInstances"] is not None:
        #    for sub_inst in sub_resp["SubscriptionInstances"]:
        #        if "CREATE routeinfo" in sub_inst["ErrorCause"]:
        #            self.logger.error("RtMgr did not creat a route to the xApp yet, resubscribing to E2 Nodes...")
        #            Thread(target=self.resubscribe_to_e2_nodes).start()
        return response

    def config_handler(self, name: str, path: str, data: bytes, ctype: str):
        """
        Handler for the HTTP GET /ric/v1/config request.
        """
        # self.logger.info("Received GET /ric/v1/config request with content type {}.".format(ctype))
        response = xapp_rest.initResponse(
            status=200, response="Config data"  # Status = 200 OK
        )  # Initiating HTTP response
        response["payload"] = json.dumps(
            self._rmrxapp._config_data
        )  # Payload = the xApp config-file
        # self.logger.debug("Config handler response: {}.".format(response))
        return response

    def liveness_handler(self, name: str, path: str, data: bytes, ctype: str):
        """
        Handler for the HTTP GET /ric/v1/health/alive request.
        """
        # self.logger.info("Received GET /ric/v1/health/alive request with content type {}.".format(ctype))
        response = xapp_rest.initResponse(
            status=200, response="Liveness"  # Status = 200 OK
        )  # Initiating HTTP response
        response["payload"] = json.dumps(
            {"status": "Healthy"}
        )  # Payload = status: Healthy
        # self.logger.debug("Liveness handler response: {}.".format(response))
        return response

    def readiness_handler(self, name: str, path: str, data: bytes, ctype: str):
        """
        Handler for the HTTP GET /ric/v1/health/ready request.
        """
        # self.logger.info("Received GET /ric/v1/health/ready request with content type {}.".format(ctype))
        if self._ready:
            response = xapp_rest.initResponse(
                status=200, response="Readiness"  # Status = 200 OK
            )  # Initiating HTTP response
            response["payload"] = json.dumps(
                {"status": "Ready"}
            )  # Payload = status: Healthy
        else:
            response = xapp_rest.initResponse(
                status=503, response="Readiness"  # Status = 503 Service Unavailable
            )
            response["payload"] = json.dumps({"status": "Not ready"})
        # self.logger.debug("Readiness handler response: {}.".format(response))
        return response

    # ------------------ SUBSCRIPTION REQUEST JSON

    # Hard coded as workaround for the wrong keys in the SubscriptionParams object
    def generate_subscription_request(
        self, inventory_name, subscription_transaction_id
    ):
        return {
            "SubscriptionId": "",
            "ClientEndpoint": {
                "Host": "service-ricxapp-xappnori-http.ricxapp",
                "HTTPPort": 8080,
                "RMRPort": 4560,
            },
            "Meid": inventory_name,  # nobe B inventory_name
            "RANFunctionID": 200,
            "E2SubscriptionDirectives": {  # Optional
                "E2TimeoutTimerValue": 2,  # Default = 2
                "E2RetryCount": 2,  # Default = 2
                "RMRRoutingNeeded": True,  # Default = True
            },
            "SubscriptionDetails": [  # Can make multiple subscriptions
                {
                    "XappEventInstanceId": subscription_transaction_id,  # "Transaction id"
                    "EventTriggers": [],
                    "ActionToBeSetupList": [
                        {
                            "ActionID": 0,
                            "ActionType": "report",
                            "ActionDefinition": [],
                            "SubsequentAction": {
                                "SubsequentActionType": "continue",
                                "TimeToWait": "w10ms",  # Default = "zero"
                            },
                        }
                    ],
                }
            ],
        }
