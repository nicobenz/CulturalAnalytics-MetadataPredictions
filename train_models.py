import json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
from plot_3d_graph import network_3d_plotly
import numpy as np
from tqdm import tqdm


def train_model(model="rf"):
    def calculate_roc():
        classes = np.unique(np.concatenate((y, y_pred)))

        y_bin = label_binarize(y, classes=np.unique(y))
        n_classes = y_bin.shape[1]
        y_pred_probs = cross_val_predict(clf, X, y, cv=cv, method='predict_proba')

        sns.set_theme()
        sns.set_context("paper")

        tab20b_cmap = plt.get_cmap('tab20b')
        tab20c_cmap = plt.get_cmap('tab20c')

        tab20b_indices = np.linspace(0, 1, 20)
        tab20c_indices = np.linspace(0, 1, 20)[:4]

        tab20b_colors = tab20b_cmap(tab20b_indices)
        tab20c_colors = tab20c_cmap(tab20c_indices)

        color_palette = np.vstack((tab20b_colors, tab20c_colors))
        sns.set_palette(color_palette)

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_pred_probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Plot all ROC curves
        plt.figure(figsize=(9, 9))
        for i, color, label in zip(range(n_classes), color_palette, classes):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label='{0} (area = {1:0.2f})'.format(label, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('')
        plt.legend(loc="lower right")
        plt.savefig(f"results/{model}_roc_curve.pdf")
        plt.clf()

        precision = dict()
        recall = dict()
        average_precision = dict()
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(y_bin[:, i], y_pred_probs[:, i])
            average_precision[i] = average_precision_score(y_bin[:, i], y_pred_probs[:, i])

        plt.figure(figsize=(9, 9))
        for i, color, label in zip(range(n_classes), color_palette, classes):
            plt.plot(recall[i], precision[i], color=color, lw=2,
                     label='{0} (AP = {1:0.2f})'.format(label, average_precision[i]))

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('')
        plt.legend(loc="upper right", fontsize='small')
        plt.savefig(f"results/{model}_precision_recall_curve.pdf")
        plt.close()

    with open("data/finished_dataset.json", "r") as f:
        data = json.load(f)
    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame([{'subgenre': item['metadata']['subgenre_name'], **item['moods']} for item in data])

    # Separate features and labels
    X = df.drop('subgenre', axis=1)
    y = df['subgenre']

    if model == "rf":
        clf = RandomForestClassifier(n_estimators=100, random_state=42, verbose=0)
    elif model == "svm":
        clf = SVC(probability=True, random_state=42, verbose=0)
    else:
        raise ValueError("Invalid model name. Use 'rf' or 'svm'.")

    # Define a KFold strategy
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # Use cross_val_predict to make predictions on each test fold
    y_pred = cross_val_predict(clf, X, y, cv=cv)

    calculate_roc()

    # Generate the confusion matrix
    conf_matrix = confusion_matrix(y, y_pred)
    np.save('data/confusion_matrix.npy', conf_matrix)
    true_positives = np.diag(conf_matrix)
    ratio_matrix = np.zeros_like(conf_matrix, dtype=float)

    for i in range(conf_matrix.shape[0]):  # Iterate over rows (actual classes)
        for j in range(conf_matrix.shape[1]):  # Iterate over columns (predicted classes)
            if i != j:  # Skip diagonal (true positives)
                if true_positives[i] > 0 and true_positives[j] > 0:
                    ratio_i_to_j = conf_matrix[i, j] / true_positives[i]
                    ratio_j_to_i = conf_matrix[j, i] / true_positives[j]
                    ratio_matrix[i, j] = (ratio_i_to_j + ratio_j_to_i) / 2
                else:
                    # If either true positive count is 0, handle accordingly. Here, we simply set it to 0.
                    ratio_matrix[i, j] = 0
            else:
                # Optionally handle diagonal elements differently since they represent true positives, not confusion.
                # For the purpose of proximity calculation, these can remain 0 or be set to a specific value if desired.
                ratio_matrix[i, j] = 1

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


    report = classification_report(y, y_pred, output_dict=True)
    with open(f"results/{model}_report_subgenre.json", "w") as f:
        json.dump(report, f, indent=4, sort_keys=True, ensure_ascii=False)

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
    true_positives_genre = np.diag(top_level_conf_matrix)
    ratio_genre_matrix = np.zeros_like(top_level_conf_matrix, dtype=float)

    for i in range(top_level_conf_matrix.shape[0]):  # Iterate over rows (actual classes)
        for j in range(top_level_conf_matrix.shape[1]):  # Iterate over columns (predicted classes)
            if i != j:  # Skip diagonal (true positives)
                if true_positives_genre[i] > 0 and true_positives_genre[j] > 0:
                    ratio_i_to_j = top_level_conf_matrix[i, j] / true_positives_genre[i]
                    ratio_j_to_i = top_level_conf_matrix[j, i] / true_positives_genre[j]
                    ratio_genre_matrix[i, j] = (ratio_i_to_j + ratio_j_to_i) / 2
                else:
                    # If either true positive count is 0, handle accordingly. Here, we simply set it to 0.
                    ratio_genre_matrix[i, j] = 0
            else:
                # Optionally handle diagonal elements differently since they represent true positives.
                # For the purpose of proximity calculation, these can remain 0 or be set to a specific value if desired.
                ratio_genre_matrix[i, j] = 1

    annot_genre_matrix = np.empty(ratio_genre_matrix.shape, dtype=object)
    for i in range(ratio_genre_matrix.shape[0]):  # Iterate over rows
        for j in range(ratio_genre_matrix.shape[1]):  # Iterate over columns
            if ratio_genre_matrix[i, j] == 1:
                annot_genre_matrix[i, j] = ""  # Set to empty string for values of 1
            else:
                # Format values to remove leading zero if less than 1 but not zero, keep as is otherwise
                annot_genre_value = ratio_genre_matrix[i, j]
                if annot_genre_value == 0:
                    annot_genre_matrix[i, j] = "0"
                else:
                    annot_genre_matrix[i, j] = f"{ratio_genre_matrix[i, j]:.2f}".lstrip('0')

    network_3d_plotly(conf_matrix, genre_mapping, class_labels, model)

    # Convert y and y_pred to pandas Series for easy mapping
    y_series = pd.Series(y)
    y_pred_series = pd.Series(y_pred)

    # Map subgenre labels to top-level genre labels
    y_top = y_series.map(genre_mapping)
    y_pred_top = y_pred_series.map(genre_mapping)

    # Generate a classification report
    report = classification_report(y_top, y_pred_top, output_dict=True)
    with open(f"results/{model}_report_genre.json", "w") as f:
        json.dump(report, f, indent=4, sort_keys=True, ensure_ascii=False)

    # Optional: Plot the confusion matrix for better visualization
    matrices = [
        {
            "data": conf_matrix,
            "title": "Subgenre Confusion Matrix",
            "file_name": f"results/{model}_subgenre_confusion_matrix.pdf",
            "formating": "d",
            "annotations": True,
            "labels": class_labels,
            "adjust": (0.16, 0.95, 0.95, 0.22)
        },
        {
            "data": ratio_matrix,
            "title": "Subgenre Ratio Matrix",
            "file_name": f"results/{model}_subgenre_ratio_matrix.pdf",
            "formating": "",
            "annotations": annot_matrix,
            "labels": class_labels,
            "adjust": (0.16, 0.95, 0.95, 0.22)
        },
        {
            "data": top_level_conf_matrix,
            "title": "Genre Confusion Matrix",
            "file_name": f"results/{model}_genre_confusion_matrix.pdf",
            "formating": "d",
            "annotations": True,
            "labels": top_genres,
            "adjust": (0.125, 0.9, 0.9, 0.1)
        },
        {
            "data": ratio_genre_matrix,
            "title": "Subgenre Ratio Matrix",
            "file_name": f"results/{model}_genre_ratio_matrix.pdf",
            "formating": "",
            "annotations": annot_genre_matrix,
            "labels": top_genres,
            "adjust": (0.125, 0.9, 0.9, 0.1)
        }
    ]
    for m in matrices:
        plt.figure(figsize=(10, 7))
        sns.heatmap(m["data"], annot=m["annotations"], fmt=m["formating"], cmap="Blues",
                    xticklabels=m["labels"], yticklabels=m["labels"])
        plt.title("")
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.subplots_adjust(left=m["adjust"][0], right=m["adjust"][1], top=m["adjust"][2], bottom=m["adjust"][3])
        plt.savefig(m["file_name"])
        plt.close()


models = ["rf", "svm"]

with tqdm(total=2) as pbar:
    for mod in models:
        train_model(model=mod)
        pbar.update(1)

