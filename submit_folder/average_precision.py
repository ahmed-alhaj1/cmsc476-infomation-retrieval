import numpy as np
import os
from sklearn.metrics import label_ranking_average_precision_score


def compute_average_precision_score(test_codes, test_labels, learned_codes, y_train, n_samples):
    # For each n_samples (=number of retrieved images to assess) we store the corresponding labels and distances
    out_labels = []
    out_distances = []


    #For each query image feature we compute the closest images from training dataset
    for i in range(len(test_codes)):
        distances = []
        # Compute the euclidian distance for each feature from training dataset
        for code in learned_codes:
            distance = np.linalg.norm(code - test_codes[i])
            distances.append(distance)

        # Store the computed distances and corresponding labels from training dataset
        distances = np.array(distances)

        # Scoring function needs to replace similar labels by 1 and different ones by 0
        labels = np.copy(y_train).astype('float32')
        labels[labels != test_labels[i]] = -1
        labels[labels == test_labels[i]] = 1
        labels[labels == -1] = 0
        distance_with_labels = np.stack((distances, labels), axis=-1)
        sorted_distance_with_labels = distance_with_labels[distance_with_labels[:, 0].argsort()]

        # The distances are between 0 and 28. The lesser the distance the bigger the relevance score should be
        sorted_distances = 28 - sorted_distance_with_labels[:, 0]
        sorted_labels = sorted_distance_with_labels[:, 1]

        # We keep only n_samples closest elements from the images retrieved
        out_distances.append(sorted_distances[:n_samples])
        out_labels.append(sorted_labels[:n_samples])

    out_labels = np.array(out_labels)
    out_labels_file_name = os.path.join(os.getcwd(),'final_project\\computed_data\\out_labels_{}'.format(n_samples))
    np.save(out_labels_file_name, out_labels)

    out_distances_file_name = os.path.join(os.getcwd(),'final_project\\computed_data\\out_distances_{}'.format(n_samples))
    out_distances = np.array(out_distances)
    np.save(out_distances_file_name, out_distances)

    # Score the model based on n_samples first images retrieved
    score = label_ranking_average_precision_score(out_labels, out_distances)
    #scores.append(score)
    return score
