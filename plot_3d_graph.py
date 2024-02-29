import networkx as nx
import plotly.graph_objects as go
import seaborn as sns
import numpy as np
import json
import plotly.offline as py_offline


def network_3d_plotly(confusion_matrix, subgenre_to_genre, subgenre_order):
    num_groups = len(set(subgenre_to_genre.values()))
    subgenre_to_genre = {id: subgenre_to_genre[id] for id in subgenre_order}
    # Generate distinct colors for each top-level genre
    sns_colors = sns.color_palette("hsv", num_groups)
    # Map each genre to a specific color
    genre_to_color = {genre: f'rgb({int(r*255)}, {int(g*255)}, {int(b*255)})'
                      for genre, (r, g, b) in zip(sorted(set(subgenre_to_genre.values())), sns_colors)}

    # Create a directed graph
    G = nx.DiGraph()

    # Calculate total predictions for each subgenre to influence node size
    total_predictions = np.sum(confusion_matrix, axis=1)
    max_predictions = np.max(total_predictions)
    # Normalize total predictions for node size (scaled between 5 and 15 for visibility)
    node_sizes = 5 + 10 * (total_predictions / max_predictions)

    # Add nodes with class (subgenre) information, including size and color based on their top-level genre
    for subgenre_id, genre_id in subgenre_to_genre.items():
        size = node_sizes[list(subgenre_to_genre.keys()).index(subgenre_id)]  # Get node size
        color = genre_to_color[genre_id]  # Get color for the genre
        G.add_node(subgenre_id, label=f'{subgenre_id}', group=genre_id, size=size, color=color)

    # Mapping subgenre IDs to indices in the confusion matrix
    true_positives = np.diag(confusion_matrix)

    # Assuming id_to_index maps subgenre IDs to their correct indices in the confusion matrix
    id_to_index = {subgenre_id: index for index, subgenre_id in enumerate(subgenre_order)}

    save_ratios = []
    for src_id in subgenre_order:
        for tgt_id in subgenre_order:
            if src_id != tgt_id:
                i, j = id_to_index[src_id], id_to_index[tgt_id]
                predictions = confusion_matrix[i, j]
                src_true_positives = true_positives[i]  # true positives for src_id
                if src_true_positives != 0:  # Ensure no division by zero
                    ratio = predictions / src_true_positives
                    G.add_edge(src_id, tgt_id, weight=ratio)

                    ratio_dict = {
                        f"{src_id}|{tgt_id}": ratio
                    }
                    save_ratios.append(ratio_dict)

    # Use spring layout for positioning
    pos = nx.spring_layout(G, dim=3, seed=42)

    # Create traces for the Plotly graph
    traces = []
    for genre_id, color in genre_to_color.items():
        # Filter nodes by genre for positions, sizes, and labels
        x_nodes = [pos[subgenre_id][0] for subgenre_id in G.nodes if subgenre_to_genre[subgenre_id] == genre_id]
        y_nodes = [pos[subgenre_id][1] for subgenre_id in G.nodes if subgenre_to_genre[subgenre_id] == genre_id]
        z_nodes = [pos[subgenre_id][2] for subgenre_id in G.nodes if subgenre_to_genre[subgenre_id] == genre_id]
        sizes = [G.nodes[subgenre_id]['size'] for subgenre_id in G.nodes if subgenre_to_genre[subgenre_id] == genre_id]
        labels = [f'{subgenre_id}' for subgenre_id in G.nodes if subgenre_to_genre[subgenre_id] == genre_id]

        trace = go.Scatter3d(
            x=x_nodes, y=y_nodes, z=z_nodes,
            mode='markers',
            marker=dict(size=sizes, color=color),
            name=f'{genre_id}',
            text=labels,
            hoverinfo='text'
        )
        traces.append(trace)

    # Plot configuration
    fig = go.Figure(data=traces)
    fig.update_layout(
        title="3D Network Graph of Confusion Matrix",
        showlegend=True,
        scene=dict(
            xaxis=dict(
                showgrid=False, zeroline=False, showticklabels=False,
                backgroundcolor='rgba(0,0,0,0)', visible=False),  # Make x-axis fully transparent and not visible
            yaxis=dict(
                showgrid=False, zeroline=False, showticklabels=False,
                backgroundcolor='rgba(0,0,0,0)', visible=False),  # Make y-axis fully transparent and not visible
            zaxis=dict(
                showgrid=False, zeroline=False, showticklabels=False,
                backgroundcolor='rgba(0,0,0,0)', visible=False),  # Make z-axis fully transparent and not visible
            # Optional: make the background of the scene itself fully transparent
            bgcolor='rgba(0,0,0,0)'
        ),
        # Optional: make the plot's background fully transparent
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    py_offline.plot(fig, filename='data/mood_plot.html', auto_open=True)
    with open("data/mood_ratios.json", "w") as f:
        json.dump(save_ratios, f)


"""
conf_matrix = np.load('data/confusion_matrix.npy')
with open("data/genre_mapping.json") as f:
    genre_mapping = json.load(f)

with open("data/finished_dataset.json", "r") as f:
    data = json.load(f)

# Convert the list of dictionaries to a DataFrame
df = pd.DataFrame([{'subgenre': item['metadata']['subgenre'], **item['moods']} for item in data])
order = df['subgenre'].tolist()
network_3d_plotly(conf_matrix, genre_mapping, order)
"""