from metaflow import FlowSpec, step, card


class TFlow5(FlowSpec):
    """
    hyperparameter search in your ML workflow
    """

    @card
    @step
    def start(self):
        """
        Load the data & train model
        """
        import pandas as pd

        self.df = ____
        self.next(self.data_prep)

    @step
    def data_prep(self):
        """
        prep data for tree-based model
        """
        import numpy as np
        import pandas as pd

        # Store target variable of training data in a safe place
        survived = ____

        #
        df = self.df.drop(["Survived"], axis=1)

        # Impute missing numerical variables
        df["Age"] = df.Age.fillna(df.Age.median())
        df["Fare"] = df.Fare.fillna(df.Fare.median())

        df = pd.get_dummies(df, columns=["Sex"], drop_first=True)
        df = df[["Sex_male", "Fare", "Age", "Pclass", "SibSp"]]

        X = ____
        y = ____

        from sklearn.model_selection import train_test_split

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.33, random_state=42
        )

        self.grid_points = np.arange(1, 9)

        self.next(self.model3, foreach=____)

    @step
    def model3(self):
        """
        make predictions
        """
        from sklearn import tree

        # Instantiate model and fit to data
        self.clf = tree.DecisionTreeClassifier(max_depth=self.input)
        self.clf.fit(____)
        self.score = ____

        self.next(self.choose_model)

    @step
    def choose_model(self, inputs):
        """
        find 'best' model
        """
        import numpy as np

        def score(inp):
            return inp.clf, inp.score

        self.results = ____
        self.model = self.results[0][0]

        self.next(self.end)

    @step
    def end(self):
        """
        End of flow!
        """
        pass


if __name__ == "__main__":
    TFlow5()
