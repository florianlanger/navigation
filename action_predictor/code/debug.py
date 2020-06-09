import networkx as nx

G = nx.read_gpickle('../graphs/no_fly_living_room.gpickle')

print(G.number_of_nodes())
print(G.nodes[0]['pose'])