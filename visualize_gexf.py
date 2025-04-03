import re
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from collections import Counter
from community import community_louvain
from pylab import rcParams

def extract_nodes_from_xml(file_path):
    """Extract node information directly from the GEXF file"""
    nodes = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                if '<node id=' in line:
                    # Extract the label using regex
                    match = re.search(r'label="([^"]*)"', line)
                    if match:
                        label = match.group(1)
                        # Filter out problematic labels with control characters
                        if all(ord(c) >= 32 or c in ['\n', '\t', '\r'] for c in label):
                            nodes.append(label)
        print(f"Extracted {len(nodes)} nodes from the file")
        return nodes
    except Exception as e:
        print(f"Error extracting nodes: {e}")
        return []

def filter_nodes_by_importance(nodes, top_n=300):
    """Filter nodes to keep only the most important ones based on word frequency"""
    # Count word frequencies across all nodes
    word_counts = Counter()
    for node in nodes:
        words = re.findall(r'\b[A-Za-z]{3,}\b', node)
        word_counts.update(words)
    
    # Calculate node importance by summing word frequencies
    node_importance = []
    for node in nodes:
        words = re.findall(r'\b[A-Za-z]{3,}\b', node)
        # Importance is the sum of word frequencies, with a penalty for common words
        importance = sum(word_counts[word] for word in words)
        # Bonus for specific important terms related to crypto/finance
        important_terms = ["token", "crypto", "blockchain", "contract", "agreement", 
                         "regulation", "compliance", "sec", "legal", "financial",
                         "trading", "exchange", "asset", "digital", "network"]
        for term in important_terms:
            if term.lower() in node.lower():
                importance += 50
        node_importance.append((node, importance))
    
    # Sort by importance and take top N
    sorted_nodes = sorted(node_importance, key=lambda x: x[1], reverse=True)
    return [node for node, _ in sorted_nodes[:top_n]]

def create_clustered_graph(nodes, output_path, max_nodes=300):
    """Create a more readable graph by clustering and filtering nodes"""
    # If we have too many nodes, filter them
    if len(nodes) > max_nodes:
        print(f"Filtering down from {len(nodes)} to {max_nodes} most important nodes")
        nodes = filter_nodes_by_importance(nodes, top_n=max_nodes)
    
    # Create a graph
    G = nx.Graph()
    
    # Add nodes
    for node in nodes:
        G.add_node(node)
    
    # Create edges between related nodes based on text similarity
    print("Creating edges between related nodes...")
    for i, node1 in enumerate(nodes):
        # Get meaningful words from this node (length > 3)
        words1 = set(w.lower() for w in re.findall(r'\b[A-Za-z]{4,}\b', node1))
        
        for j, node2 in enumerate(nodes[i+1:], i+1):
            if i != j:
                # Get meaningful words from other node
                words2 = set(w.lower() for w in re.findall(r'\b[A-Za-z]{4,}\b', node2))
                
                # Calculate similarity based on shared words
                intersection = words1.intersection(words2)
                if intersection:
                    # Weight by number of shared words
                    weight = len(intersection) / max(len(words1), len(words2))
                    if weight > 0.1:  # Only connect if there's meaningful similarity
                        G.add_edge(node1, node2, weight=weight)
    
    # Remove isolated nodes
    isolated_nodes = list(nx.isolates(G))
    G.remove_nodes_from(isolated_nodes)
    print(f"Removed {len(isolated_nodes)} isolated nodes")
    
    # Find communities for coloring
    print("Detecting communities...")
    partition = community_louvain.best_partition(G)
    
    # Get unique community IDs
    communities = set(partition.values())
    print(f"Found {len(communities)} communities")
    
    # Create a colormap
    cmap = cm.get_cmap('tab20', len(communities))
    
    # Draw the graph
    plt.figure(figsize=(30, 20))
    
    # Use a more advanced layout for better node distribution
    print("Computing layout...")
    layout = nx.spring_layout(G, k=0.3, iterations=100, seed=42, weight='weight')
    
    # Draw nodes with community colors
    for comm in communities:
        node_list = [node for node in G.nodes() if partition[node] == comm]
        nx.draw_networkx_nodes(G, layout, nodelist=node_list, 
                            node_size=[300 * G.degree(node) for node in node_list],
                            node_color=[cmap(comm)], alpha=0.8, 
                            label=f"Community {comm}")
    
    # Draw edges with alpha based on weight
    edge_weights = [G[u][v]['weight'] * 2 for u, v in G.edges()]
    nx.draw_networkx_edges(G, layout, width=edge_weights, alpha=0.3, edge_color='gray')
    
    # Draw labels with different font sizes based on degree
    node_degrees = dict(G.degree())
    font_sizes = {node: min(12, 8 + node_degrees[node]) for node in G.nodes()}
    nx.draw_networkx_labels(G, layout, font_size=8, font_family='sans-serif')
    
    plt.axis('off')
    plt.title('Crypto Entity Relationship Clusters', fontsize=24)
    
    # Add a small legend for communities
    plt.legend(fontsize=10, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Graph visualization saved to {output_path}")
    
    # Save a second version with only community centers labeled for clarity
    plt.figure(figsize=(30, 20))
    
    # Draw nodes with community colors (same as before)
    for comm in communities:
        node_list = [node for node in G.nodes() if partition[node] == comm]
        nx.draw_networkx_nodes(G, layout, nodelist=node_list, 
                            node_size=[300 * G.degree(node) for node in node_list],
                            node_color=[cmap(comm)], alpha=0.8)
    
    # Draw edges (same as before)
    nx.draw_networkx_edges(G, layout, width=edge_weights, alpha=0.3, edge_color='gray')
    
    # Only label the highest-degree node in each community
    community_centers = {}
    for node, comm in partition.items():
        if comm not in community_centers or node_degrees[node] > node_degrees[community_centers[comm]]:
            community_centers[comm] = node
    
    # Draw labels only for community centers
    nx.draw_networkx_labels(G, layout, 
                          font_size=12, 
                          font_family='sans-serif',
                          font_weight='bold',
                          labels={node: node for node in community_centers.values()})
    
    plt.axis('off')
    plt.title('Crypto Entity Relationship Clusters (Key Entities Only)', fontsize=24)
    plt.tight_layout()
    plt.savefig(output_path.replace('.png', '_simplified.png'), dpi=300, bbox_inches='tight')
    print(f"Simplified graph visualization saved to {output_path.replace('.png', '_simplified.png')}")
    
    # Print key nodes from each community
    print("\nKey entities in each community:")
    for comm in communities:
        community_nodes = [node for node in G.nodes() if partition[node] == comm]
        top_nodes = sorted(community_nodes, key=lambda n: G.degree(n), reverse=True)[:5]
        print(f"Community {comm}: {', '.join(top_nodes)}")
    
    return True

if __name__ == "__main__":
    original_file = "entity_relationships.gexf"
    
    print("Extracting and analyzing entity nodes...")
    nodes = extract_nodes_from_xml(original_file)
    
    if nodes:
        print("Creating clustered visualization...")
        create_clustered_graph(nodes, "entity_clusters.png")
    else:
        print("Could not extract nodes from the file.")