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

        self.df = ____
        self.next(self.model1, self.model2)

    @step
    def model1(self):
        """
        make predictions
        """
        import pandas as pd
        from sklearn.metrics import accuracy_score

        self.clf = "model_1"
        self.df["model"] = ____
        self.score = ____

        self.next(self.choose_model)

    @step
    def model2(self):
        """
        make predictions
        """
        import pandas as pd
        from sklearn.metrics import accuracy_score

        self.clf = "model_2"
        self.df["model"] = ____
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
        self.model = ____

        self.next(self.end)

    @step
    def end(self):
        """
        End of flow!
        """
        pass


if __name__ == "__main__":
    TFlow3()
