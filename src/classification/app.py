import pandas as pd
from plot_class_distribution import plot_class_distribution
from load_data import load_data
from analyze_class_distribution import analyze_class_distribution
from prepare_data import prepare_data
from train_and_evaluate_model import train_and_evaluate_model
from generate_classification_report import generate_classification_report
from plot_confusion_matrix import plot_confusion_matrix
from analyze_class_errors import analyze_class_errors
from calculate_baseline import calculate_baseline


def main():
    """Основная функция."""
    # 1. Визуализация распределения классов
    stats = plot_class_distribution(y_data_path="y_data.npy", figsize=(14, 8))
    print("Статистика классов:", stats)

    # 2. ML пайплайн
    X, y = load_data()

    unique_classes, class_counts = analyze_class_distribution(y)

    X_train, X_test, y_train, y_test, _ = prepare_data(X, y)

    model, accuracy, _ = train_and_evaluate_model(X_train, y_train, X_test, y_test)

    y_pred, _ = generate_classification_report(model, X_test, y_test, unique_classes)

    plot_confusion_matrix(y_test, y_pred, unique_classes)

    analyze_class_errors(y_test, y_pred, unique_classes)

    baseline_accuracy, _ = calculate_baseline(y_test, class_counts, unique_classes)

    print("Accuracy:", accuracy)
    print("Baseline accuracy:", baseline_accuracy)


if __name__ == "__main__":
    main()
