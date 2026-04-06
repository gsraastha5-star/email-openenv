def grade(predictions, true_labels):
    correct = sum([p == t for p, t in zip(predictions, true_labels)])
    return correct / len(true_labels)