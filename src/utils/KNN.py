from collections import Counter
import math


class KNN:
    def __init__(self, k=3):
        self.k = k
        self.train_data = []
        self.train_labels = []

    def fit(self, data, labels):
        self.train_data = data
        self.train_labels = labels

    def calculate_distance(self, point1, point2):
        # Simple Euclidean distance
        total = 0
        for i in range(len(point1)):
            total += (point1[i] - point2[i]) ** 2
        return math.sqrt(total)

    def predict_one(self, test_point):
        distances = []

        # Calculate distance from the test point to all training points
        for i in range(len(self.train_data)):
            dist = self.calculate_distance(test_point, self.train_data[i])
            distances.append((dist, self.train_labels[i]))

        # Sort by distance and take the first k
        distances.sort()
        k_nearest = distances[: self.k]

        # Count labels
        labels = [label for _, label in k_nearest]
        most_common = Counter(labels).most_common(1)[0][0]
        return most_common

    def predict(self, test_data):
        predictions = []
        for point in test_data:
            pred = self.predict_one(point)
            predictions.append(pred)
        return predictions

    def score(self, test_data, test_labels):
        correct = 0
        predictions = self.predict(test_data)
        for i in range(len(predictions)):
            if predictions[i] == test_labels[i]:
                correct += 1
        return correct / len(test_labels)
