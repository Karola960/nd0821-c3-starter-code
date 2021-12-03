# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Gradient Boosted Classifier model implemented.

## Intended Use
Model predicts the salary based on specified features.

## Training Data
Selected 80% of data for training phase.

## Evaluation Data
Selected 20% of data for evaluation phase.

## Metrics
The model was evaluated using fbeta, precision and recall scores.

## Ethical Considerations
Used publicly available Census Bureau data https://archive.ics.uci.edu/ml/datasets/census+income. The dataset contains data that could potentially discriminate against people, such as related race, education, gender and country of origin.

## Caveats and Recommendations
It should be consider to use more data from a different source. I recommend to use other models and improve training parameters to get better results. 