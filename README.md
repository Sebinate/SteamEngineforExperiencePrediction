# TF-IDF + Multinomial Naive Bayes Text Classifier (PySpark)

A distributed text classification pipeline built on PySpark MLlib, trained on the Steam 100M Reviews dataset. The dataset contains over 100 million user-written game reviews sourced from the Steam platform, making single-machine frameworks impractical. PySpark is used throughout to distribute preprocessing, feature extraction, and model training across a cluster.

To run the full pipeline including exploratory data analysis, open and execute `experiments.ipynb`. The notebook walks through data loading, EDA, pipeline training, and evaluation in a single place.

---

## Requirements

- Python 3.10+
- Apache Spark 3.3+
- Java 11

```bash
pip install -r requirements.txt
```

---

## Pipeline Stages
1. `Text Cleaning`
2. `Tokenizer` - Tokenizes Review text
3. `StopWordsRemover` - Strips common stop words
4. `HashingTF` - Maps tokens to a sparse frequency vector
5. `IDF` - Re-weights term frequencies by inverse document frequency
6. `NaiveBayes` (multinomial) - Fits the classifier on TF-IDF vectors

---

## Usage

### Train and Test the Pipeline

- Run using the experiments/experiments.ipynb file
- Ensure that the Steam 100 Million Reviews dataset is present in   ```data/ ```

### Input Schema

The input Parquet file must contain:
- `review` (string) - raw document text
- `price` (float)- game price
- `playtime_at_review` (float) - total playtime of the user when the review was made
- `recieved_free` (bool) - inidicates whether the user recieved the game for free
- `written_during_early_access` (bool) - indicates whether the user wrote the review before the game was official released
- `num_reviews` (int) - number of reviews made by the user at the time of writing the review

**Target Variable**
- `voted_up` (bool) - indicates whether the review is positive (recommend) or negative (does not recommend), 
---

## Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `num_features` | `2^12` | HashingTF bucket size |
| `smoothing` | `1.0` | Laplace smoothing for Naive Bayes |
| `test_size` | `0.2` | Holdout fraction for evaluation |

---

## Evaluation

Metrics computed on the test split using `MulticlassClassificationEvaluator`:

- Accuracy, F1-Score, Precision, Recall
    - Ran at the end of the `experiments/experiments.ipynb` file 
---

## License

MIT