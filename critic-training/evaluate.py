import argparse
import jsonlines

parser = argparse.ArgumentParser()

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

    parser.add_argument("--pred_path", required=True,
                    help="The path to previous model predictions")
    parser.add_argument("--labels_path", required=True,
                    help="The path to previous model predictions")

    args = parser.parse_args()

    predictions_path = args.pred_path
    labels_path = args.labels_path

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