from gabriel_protocol import gabriel_pb2
from gabriel_server import cognitive_engine
from google.protobuf.message import Message


class LEGOCognitiveEngine(cognitive_engine.Engine):

    def __init__(self):
        pass

    def handle(self, from_client):
        if from_client.payload_type != gabriel_pb2.PayloadType.IMAGE:
            return cognitive_engine.wrong_input_format_message(
                from_client.frame_id)

        lego_state = cognitive_engine.unpack_engine_fields()
