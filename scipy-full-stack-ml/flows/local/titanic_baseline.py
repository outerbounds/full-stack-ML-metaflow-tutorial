from metaflow import FlowSpec, step, card


class TFlow1(FlowSpec):
    """
    train a boosted tree
    """

    @card
    @step
    def start(self):
        """
        Load the data & train model
        """
        import pandas as pd

        # NOTE: The data path in read_csv is relative to where you run command from.
        self.df = pd.read_csv("./data/titanic.csv")
        self.next(self.predict)

    @step
    def predict(self):
        """
        make predictions
        """
        import pandas as pd
        from sklearn.metrics import accuracy_score

        self.df["model_1"] = 0
        self.score1 = accuracy_score(self.df["Survived"], self.df["model_1"])

        self.next(self.end)

    @step
    def end(self):
        """
        End of flow!
        """
        print("Score = %s" % self.score1)


if __name__ == "__main__":
    TFlow1()
