import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from collections import Counter




class VotingClassifier:
    def __init__(self, estimators, voting='hard', weights=None):
        self.estimators = estimators
        self.voting = voting
        self.weights = weights if weights is not None else [1] * len(estimators)
        self.fitted_models = []


    def fit(self, X, y):
        self.fitted_models = []

        for _, model in self.estimators:
            cloned_model = model.__class__(**model.get_params())
            cloned_model.fit(X, y)
            self.fitted_models.append(cloned_model)

        return self
    
    def predict_hard(self, X):
        predictions = np.array([model.predict(X) for model in self.fitted_models])

        n_samples = X.shape[0]
        final_preds = np.zeros(n_samples, dtype=int)

        for i in range(n_samples):
            sample_preds = predictions[: , i]
            vote_counts  = Counter()
            for pred, weight in zip(sample_preds, self.weights):
                vote_counts[pred] += weight

            final_preds[i] = vote_counts.most_common(1)[0][0]
        return final_preds
    
    def predict_soft(self, X):
            probas = np.array([model.predict_proba(X) for model in self.fitted_models])
            

            n_samples = X.shape[0]
            n_classes = probas.shape[2]
            
            weighted_probas = np.zeros((n_samples, n_classes))
            for i, (proba, weight) in enumerate(zip(probas, self.weights)):
                weighted_probas += weight * proba
            weighted_probas /= sum(self.weights)  
            

            final_preds = np.argmax(weighted_probas, axis=1)
            return final_preds

    def predict(self, X):
        """Predict based on voting type."""
        if self.voting == 'hard':
            return self.predict_hard(X)
        elif self.voting == 'soft':
            return self.predict_soft(X)
        else:
            raise ValueError("Voting must be 'hard' or 'soft'")

    def score(self, X, y):
        """Calculate accuracy."""
        preds = self.predict(X)
        return np.mean(preds == y)

if __name__ == "__main__":

    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    clf1 = LogisticRegression(random_state=42)
    clf2 = SVC(probability=True, random_state=42) 
    clf3 = DecisionTreeClassifier(random_state=42)


    estimators = [('lr', clf1), ('svc', clf2), ('dt', clf3)]

    hard_voter = VotingClassifier(estimators, voting='hard', weights=[2, 1.5, 0.5])
    hard_voter.fit(X_train, y_train)
    hard_score = hard_voter.score(X_test, y_test)
    print(f"Hard Voting Accuracy: {hard_score:.4f}")


    soft_voter = VotingClassifier(estimators, voting='soft', weights=[1, 1.5, 0.5])
    soft_voter.fit(X_train, y_train)
    soft_score = soft_voter.score(X_test, y_test)
    print(f"Soft Voting Accuracy: {soft_score:.4f}")   
        