import json

from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, make_scorer, fbeta_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold


def train_model(model, param_grid, X_train, y_train):
    """
    Trains the model with hyperparameters and returns the trained model.

    Parameters
    -----------
    model : Scikit's model

    param_grid : dict
        hyperparameters required by GridSearchCV

    X_train: dataframe
    		dataset's features for traing

    y_train: dataframe
    		dataset's label for training

    Returns
    -------
    base : RandomizedSearchCV
    	an RandomizedSearchCV object
    """
    k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=50)
    base = RandomizedSearchCV(
        estimator=model, param_distributions=param_grid,
        n_iter=50, scoring=make_scorer(fbeta_score, beta=0.5),
        n_jobs=-1, verbose=3, cv=k_fold, refit='precision',
    )

    base.fit(X_train, y_train)

    return base


def tune_threshold(model, x_data, y_data):
    model_name = model.__class__.__name__
    y_probabilities = model.predict_proba(x_data)[:, 1]

    scores = []
    thresholds = []

    for threshold in range(1, 101):
        threshold = threshold / 100
        y_predictions = (y_probabilities >= threshold).astype(int)

        score = fbeta_score(y_data, y_predictions, beta=0.5)

        scores.append(score)
        thresholds.append(threshold)

    calculated_thresholds = {
        'model': model_name,
        'metric': 'precision',
        'thresholds': thresholds,
        'scores': scores,
    }

    plt.figure(figsize=(12, 8))
    plt.title(f'Precision Scores for {model}')
    plt.plot(thresholds, scores, label='Precision Scores')
    plt.xlabel('Threshold')
    plt.ylabel('Precision Score')
    plt.legend()
    plt.grid()
    plt.show()

    return calculated_thresholds


def calculate_best_threshold(thresholds):
    best_threshold = thresholds.pop()
    for threshold in thresholds:
        if threshold['highest_score'] > best_threshold['highest_score']:
            best_threshold = threshold

    return best_threshold


def get_best_threshold(thresholds, path):
    calculated_threshold = []
    for threshold in thresholds:
        model = threshold['model']
        metric = threshold['metric']
        thresholds_ = threshold['thresholds']
        scores = threshold['scores']

        # find the best threshold from the highest score
        highest_score = max(scores)
        index_of_highest_score = scores.index(highest_score)
        best_threshold_ = thresholds_[index_of_highest_score]

        calculated_threshold.append({
            'model': model,
            'highest_score': highest_score,
            'threshold': best_threshold_,
            'metric': metric,
        })

    best_threshold = calculate_best_threshold(calculated_threshold)
    with open(path, 'w') as f:
        json.dump(best_threshold, f)

    return best_threshold


def evaluate_best_model(model, threshold, x_data, y_data):
    print(" Threshold : ", threshold)

    threshold = threshold['threshold']
    y_predicted = (model.predict_proba(x_data)[:, 1] >= threshold).astype(int)

    print(classification_report(y_data, y_predicted))