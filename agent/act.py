from ..main import server
import threading


# From another class/function:
def random_action_test():

    # Or send to specific client
    for client in server.clients:
        server.send_to_client(client, "Your turn!")
