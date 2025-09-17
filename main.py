"""Simple Iris Visualization

This script loads the classic Iris dataset using scikit‑learn, converts it to a
pandas DataFrame and creates a scatter plot of sepal length versus sepal
width. The generated plot is saved to disk as ``iris_scatter.png``.

Running this script demonstrates data loading, manipulation and basic
visualization in Python.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets

def main() -> None:
    """Load the Iris dataset and generate a scatter plot."""
    # Load the iris dataset
    iris = datasets.load_iris()
    # Create a DataFrame with feature data and target names
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

    # Create a scatter plot of sepal length vs sepal width
    plt.figure(figsize=(8, 6))
    for species, group in df.groupby('species'):
        plt.scatter(
            group['sepal length (cm)'],
            group['sepal width (cm)'],
            label=species,
            alpha=0.7
        )
    plt.title('Iris Dataset: Sepal Length vs Sepal Width')
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Sepal Width (cm)')
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    # Save figure
    output_path = 'iris_scatter.png'
    plt.savefig(output_path)
    print(f'Saved scatter plot to {output_path}')

if __name__ == '__main__':
    main()
