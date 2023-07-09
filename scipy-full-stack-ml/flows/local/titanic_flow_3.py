from metaflow import FlowSpec, step, card


class TFlow3(FlowSpec):
    """
    train two models for titanic data
    """

    @card
    @step
    def start(self):
        """
        Load the data & train model
        """
        import pandas as pd

        self.df = pd.read_csv("data/titanic.csv")
        self.next(self.model1, self.model2)

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
        print("TFlow3 is all done.")


if __name__ == "__main__":
    TFlow3()
