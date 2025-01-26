from constants import BINARY_CRITIC_PREDICTIONS_PATH, DEVIGN_TEST_DATASET_PATH
import jsonlines

def get_Labels(labels_path):
    labels = []
    with jsonlines.open(labels_path, mode="r") as file:
        for line in file:
            labels.append(line["target"])

    return labels


def get_Predictions(predictions_path):
    predictions = []
    with open(predictions_path, "r") as file:
        for line in file: 
            predictions.append(int(line[:-1]))

    return predictions


def main():

    predictions_path = BINARY_CRITIC_PREDICTIONS_PATH
    labels_path = DEVIGN_TEST_DATASET_PATH

    predictions = get_Predictions(predictions_path)
    labels = get_Labels(labels_path)

    total = len(predictions)
    i = 0
    for a, p in zip(labels, predictions):
        if a == p:
            i += 1

    accuracy = i / total
    print(f" accuracy: {accuracy}")

if __name__ == "__main__":
    main()