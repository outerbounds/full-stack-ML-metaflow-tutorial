from metaflow import FlowSpec, step, card, conda
import json


class Branch_Flow_Cloud(FlowSpec):
    """
    train multiple tree based methods
    """

    @conda(libraries={"scikit-learn": "1.0.2"})
    @card
    @step
    def start(self):
        """
        Load the data
        """
        # Import scikit-learn dataset library
        from sklearn import datasets

        # Load dataset
        self.iris = datasets.load_iris()
        self.X = self.iris["data"]
        self.y = self.iris["target"]
        self.next(self.rf_model, self.xt_model, self.dt_model)

    @conda(libraries={"scikit-learn": "1.0.2"})
    @step
    def rf_model(self):
        """
        build random forest model
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score

        self.clf = RandomForestClassifier(
            n_estimators=10, max_depth=None, min_samples_split=2, random_state=0
        )
        self.scores = cross_val_score(self.clf, self.X, self.y, cv=5)
        self.next(self.choose_model)

    @conda(libraries={"scikit-learn": "1.0.2"})
    @step
    def xt_model(self):
        """
        build extra trees classifier
        """
        from sklearn.ensemble import ExtraTreesClassifier
        from sklearn.model_selection import cross_val_score

        self.clf = ExtraTreesClassifier(
            n_estimators=10, max_depth=None, min_samples_split=2, random_state=0
        )

        self.scores = cross_val_score(self.clf, self.X, self.y, cv=5)
        self.next(self.choose_model)

    @conda(libraries={"scikit-learn": "1.0.2"})
    @step
    def dt_model(self):
        """
        build decision tree classifier
        """
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.model_selection import cross_val_score

        self.clf = DecisionTreeClassifier(
            max_depth=None, min_samples_split=2, random_state=0
        )

        self.scores = cross_val_score(self.clf, self.X, self.y, cv=5)

        self.next(self.choose_model)

    @conda(libraries={"scikit-learn": "1.0.2"})
    @step
    def choose_model(self, inputs):
        """
        find 'best' model
        """
        import numpy as np

        def score(inp):
            return inp.clf, np.mean(inp.scores)

        self.results = sorted(map(score, inputs), key=lambda x: -x[1])
        self.model = self.results[0][0]
        self.next(self.end)

    @conda(libraries={"scikit-learn": "1.0.2"})
    @step
    def end(self):
        """
        End of flow, yo!
        """
        print("Scores:")
        print("\n".join("%s %f" % res for res in self.results))


if __name__ == "__main__":
    Branch_Flow_Cloud()
