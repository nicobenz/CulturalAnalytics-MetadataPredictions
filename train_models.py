import json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from plot_3d_graph import network_3d_plotly
import numpy as np


def train_rf():
    with open("data/finished_dataset.json", "r") as f:
        data = json.load(f)
    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame([{'subgenre': item['metadata']['subgenre_name'], **item['moods']} for item in data])

    # Separate features and labels
    X = df.drop('subgenre', axis=1)
    y = df['subgenre']

    # Initialize the Random Forest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42, verbose=1)

    # Define a KFold strategy
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # Use cross_val_predict to make predictions on each test fold
    y_pred = cross_val_predict(clf, X, y, cv=cv)

    # Calculate the F1 score
    f1 = f1_score(y, y_pred, average='weighted')
    print(f"F1 Score Subgenres: {f1}")

    # Generate the confusion matrix
    conf_matrix = confusion_matrix(y, y_pred)
    np.save('data/confusion_matrix.npy', conf_matrix)
    true_positives = np.diag(conf_matrix)
    ratio_matrix = np.zeros_like(conf_matrix, dtype=float)
    for i in range(conf_matrix.shape[0]):  # Iterate over rows (actual classes)
        for j in range(conf_matrix.shape[1]):  # Iterate over columns (predicted classes)
            if true_positives[j] > 0:  # Ensure division by zero is handled
                ratio_matrix[i, j] = conf_matrix[i, j] / true_positives[j]
            else:
                # Decide how to handle 0 true positives, e.g., set to 0 or some other value
                ratio_matrix[i, j] = 0

    annot_matrix = np.empty(ratio_matrix.shape, dtype=object)
    for i in range(ratio_matrix.shape[0]):  # Iterate over rows
        for j in range(ratio_matrix.shape[1]):  # Iterate over columns
            if ratio_matrix[i, j] == 1:
                annot_matrix[i, j] = ""  # Set to empty string for values of 1
            else:
                # Format values to remove leading zero if less than 1 but not zero, keep as is otherwise
                annot_value = ratio_matrix[i, j]
                if annot_value == 0:
                    annot_matrix[i, j] = "0"
                else:
                    annot_matrix[i, j] = f"{ratio_matrix[i, j]:.2f}".lstrip('0')
    # Print the confusion matrix
    #print("Subgenre Confusion Matrix:")
    #print(conf_matrix)

    print("Subgenre Classification Report:")
    print(classification_report(y, y_pred))

    with open("data/genre_mapping.json") as f:
        genre_mapping = json.load(f)



    # Ensure class_labels are sorted uniquely to maintain consistent order
    class_labels = np.unique(np.concatenate((y, y_pred)))
    # Map each unique subgenre label to its corresponding top-level genre
    subgenre_to_top_genre = [genre_mapping[label] for label in class_labels]
    # Create a unique list of top-level genres in the order they appear
    top_genres = sorted(set(subgenre_to_top_genre), key=subgenre_to_top_genre.index)
    # Create a mapping from top-level genres to a new index
    top_genre_to_new_index = {genre: i for i, genre in enumerate(top_genres)}
    # Initialize an empty confusion matrix for top-level genres
    top_level_conf_matrix = np.zeros((len(top_genres), len(top_genres)), dtype=int)
    # Aggregate the confusion matrix values from subgenres to top-level genres
    for i, row_label in enumerate(class_labels):
        for j, col_label in enumerate(class_labels):
            # Map the subgenre indices to top-level genre indices
            top_i = top_genre_to_new_index[genre_mapping[row_label]]
            top_j = top_genre_to_new_index[genre_mapping[col_label]]
            # Aggregate the values
            top_level_conf_matrix[top_i, top_j] += conf_matrix[i, j]

    network_3d_plotly(conf_matrix, genre_mapping, class_labels)

    # Convert y and y_pred to pandas Series for easy mapping
    y_series = pd.Series(y)
    y_pred_series = pd.Series(y_pred)

    # Map subgenre labels to top-level genre labels
    y_top = y_series.map(genre_mapping)
    y_pred_top = y_pred_series.map(genre_mapping)

    f1 = f1_score(y_top, y_pred_top, average='weighted')  # You can adjust the averaging method as needed

    print(f"F1 Score Genres: {f1}")

    # Generate a classification report
    report = classification_report(y_top, y_pred_top)
    print("Genre Classification Report:")
    print(report)
    #print(y)
    #print(y_pred)

    # Optional: Plot the confusion matrix for better visualization
    matrices = [
        {
            "data": conf_matrix,
            "title": "Confusion Matrix",
            "file_name": "data/rf_subgenre_confusion_matrix.pdf",
            "formating": "d",
            "annotations": True,
            "labels": class_labels,
            "adjust": (0.16, 0.95, 0.95, 0.22)
         },
        {
            "data": ratio_matrix,
            "title": "Ratio Matrix",
            "file_name": "data/rf_subgenre_ratio_matrix.pdf",
            "formating": "",
            "annotations": annot_matrix,
            "labels": class_labels,
            "adjust": (0.16, 0.95, 0.95, 0.22)
        },
        {
            "data": top_level_conf_matrix,
            "title": "Genre Aggregated Confusion Matrix",
            "file_name": "data/rf_genre_confusion_matrix.pdf",
            "formating": "d",
            "annotations": True,
            "labels": top_genres,
            "adjust": (0.125, 0.9, 0.9, 0.1)
        },
    ]
    for m in matrices:
        plt.figure(figsize=(10, 7))
        sns.heatmap(m["data"], annot=m["annotations"], fmt=m["formating"], cmap="Blues",
                    xticklabels=m["labels"], yticklabels=m["labels"])
        plt.title(m["title"])
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.subplots_adjust(left=m["adjust"][0], right=m["adjust"][1], top=m["adjust"][2], bottom=m["adjust"][3])
        plt.savefig(m["file_name"])


train_rf()
