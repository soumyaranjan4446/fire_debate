import json
import networkx as nx
import matplotlib.pyplot as plt
import glob

def visualize_latest():
    # Find latest file
    files = glob.glob("data/processed/training_set/*.json")
    if not files: return
    latest = max(files, key=lambda x: x)
    
    with open(latest, 'r') as f:
        data = json.load(f)

    G = nx.DiGraph()
    
    # Add Nodes
    for i, turn in enumerate(data['turns']):
        node_id = f"Turn {i}\n({turn['agent_id']})"
        G.add_node(node_id, color='green' if turn['stance']=='PRO' else 'red')
        
        # Add edges (sequential)
        if i > 0:
            prev_id = f"Turn {i-1}\n({data['turns'][i-1]['agent_id']})"
            G.add_edge(prev_id, node_id)

    # Draw
    pos = nx.spring_layout(G)
    colors = [nx.get_node_attributes(G, 'color')[n] for n in G.nodes()]
    
    plt.figure(figsize=(10, 6))
    nx.draw(G, pos, with_labels=True, node_color=colors, node_size=2000, font_size=8)
    plt.title(f"Debate Graph: {data['claim_text'][:50]}...")
    plt.savefig("debate_structure.png")
    print("🖼️ Graph saved to debate_structure.png")

if __name__ == "__main__":
    visualize_latest()