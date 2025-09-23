import argparse

parser = argparse.ArgumentParser()

def calculate_accuracy(file_path):
    correct = 0
    total = 0

    with open(file_path, "r") as file:
        for line in file:
            pred, label = line.strip().split("\t") 
            pred, label = int(pred), int(label)

            if pred == label:
                correct += 1
            total += 1

    accuracy = correct / total if total > 0 else 0
    print(f"Accuracy: {accuracy:.4f} ({correct}/{total} correct)")
    return accuracy


def main():
    parser.add_argument("--file_path", required=True,
                    help="The path to previous model predictions")

    args = parser.parse_args()
    file_path = args.file_path

    calculate_accuracy(file_path)

if __name__ == "__main__":
    main()
