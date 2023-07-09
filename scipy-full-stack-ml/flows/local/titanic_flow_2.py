from metaflow import FlowSpec, step, card


class TFlow2(FlowSpec):
    """
    train a rule-based method
    """

    @card
    @step
    def start(self):
        """
        Load the data & train model
        """
        import pandas as pd

        self.df = pd.read_csv("data/titanic.csv")
        self.next(self.predict)

    @step
    def predict(self):
        """
        make predictions
        """
        import pandas as pd
        from sklearn.metrics import accuracy_score

        self.df["model_2"] = self.df.Sex == "female"
        self.score2 = accuracy_score(self.df["Survived"], self.df["model_2"])

        self.next(self.end)

    @step
    def end(self):
        """
        End of flow!
        """
        print("Score = %s" % self.score2)

        print("TFlow2 is all done.")


if __name__ == "__main__":
    TFlow2()
