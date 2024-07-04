import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

#Generate a dataframe of the imported data
df = pd.read_csv(r'data.csv')

#Define columns of data with specific column headers
red = df["R_fighter"]
blue = df["B_fighter"]
results = df["Winner"]

#Append pairs of fighters to a list with the first fighter being the winner
fights_data = []
for x,y,z in zip(red, blue, results):
    if z == "Red":
        fights_data.append((str(x), str(y)))
    if z == "Blue":
        fights_data.append((str(y), str(x)))

#Use nx graphs to build a directed graph in which there is an 'edge' pointing from the winner to the loser
def build_graph(fights):
    G = nx.DiGraph()
    for fight in fights:
        winner, loser = fight
        G.add_edge(winner, loser)
    return G

#Creates a subgraph from a list of the path parameter
def extract_path_subgraph(graph, path):
    subgraph = graph.subgraph(path)
    return subgraph

#Function that builds the plot based on the subgraph
def plot_path_subgraph(subgraph):
    col = ['skyblue'] * len(subgraph)
    nx.draw(subgraph, pos=None, with_labels=True, arrowsize = 20, font_weight='bold', node_size=700, node_color=col, font_size=8)
    plt.show()
    
#Tries to find the shortest path, if no path, it returns an empty list
def find_path(graph, fighter_a, fighter_b):
    try:
        return nx.shortest_path(graph, fighter_a, fighter_b)
    except nx.NetworkXNoPath:
        return []

#Build the graph
mma_graph = build_graph(fights_data)

#Check if one fighter can beat another
#Must manually input fighter names with correct spelling and formatting 
fighter_a = "Tom Aspinall"
fighter_b = "Conor McGregor"
path = find_path(mma_graph, fighter_a, fighter_b)

if path:
    print(f"{fighter_a} can beat {fighter_b} based on MMA math.")
    subgraph = extract_path_subgraph(mma_graph, path)
    plot_path_subgraph(subgraph)
else:
    print(f"{fighter_a} cannot beat {fighter_b} based on MMA math.")

