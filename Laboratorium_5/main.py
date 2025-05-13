from graph_lib import FlowNetwork

flowNetwork = FlowNetwork.random_flow_network()

flowNetwork.draw_flow_network()

print(f"|f_max|={flowNetwork.ford_fulkerson()}")

flowNetwork.draw_flow_network()

