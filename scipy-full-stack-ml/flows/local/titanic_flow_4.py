from metaflow import FlowSpec, step, card


class TFlow4(FlowSpec):
    """
    train 3 models, including data preparation
    """

    @card
    @step
    def start(self):
        """
        Load the data & train model
        """
        import pandas as pd

        self.df = pd.read_csv("data/titanic.csv")
        self.next(self.model1, self.model2, self.data_prep)

    @step
    def model1(self):
        """
        make predictions
        """
        import pandas as pd
        from sklearn.metrics import accuracy_score

        self.clf = "model_1"
        self.df["model"] = 0
        self.score = accuracy_score(self.df["Survived"], self.df["model"])

        self.next(self.choose_model)

    @step
    def model2(self):
        """
        make predictions
        """
        import pandas as pd
        from sklearn.metrics import accuracy_score

        self.clf = "model_2"
        self.df["model"] = self.df.Sex == "female"
        self.score = accuracy_score(self.df["Survived"], self.df["model"])

        self.next(self.choose_model)

    @step
    def data_prep(self):
        """
        prep data for tree-based model
        """
        import pandas as pd

        # Store target variable of training data in a safe place
        survived = self.df.Survived

        #
        df = self.df.drop(["Survived"], axis=1)

        # Impute missing numerical variables
        df["Age"] = df.Age.fillna(df.Age.median())
        df["Fare"] = df.Fare.fillna(df.Fare.median())

        df = pd.get_dummies(df, columns=["Sex"], drop_first=True)
        df = df[["Sex_male", "Fare", "Age", "Pclass", "SibSp"]]

        X = df.values
        y = survived.values

        from sklearn.model_selection import train_test_split

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.33, random_state=42
        )

        self.next(self.model3)

    @step
    def model3(self):
        """
        make predictions
        """
        from sklearn import tree

        # Instantiate model and fit to data
        self.clf = tree.DecisionTreeClassifier(max_depth=3)
        self.clf.fit(self.X_train, self.y_train)
        self.score = self.clf.score(self.X_test, self.y_test)

        self.next(self.choose_model)

    @step
    def choose_model(self, inputs):
        """
        find 'best' model
        """
        import numpy as np

        def score(inp):
            return inp.clf, inp.score

        self.results = sorted(map(score, inputs), key=lambda x: -x[1])
        self.model = self.results[0][0]

        self.next(self.end)

    @step
    def end(self):
        """
        End of flow!
        """
        print("Scores:")
        print("\n".join("%s %f" % res for res in self.results))
        print("Best model:")
        print(self.model)
        print("TFlow4 is all done.")


if __name__ == "__main__":
    TFlow4()
