from metaflow import FlowSpec, step, card, conda_base, project, IncludeFile


@conda_base(libraries={"conda-forge::pandas": "2.0.1", "scikit-learn": "1.1.2"})
@project(name="titanic_survival_prediction")
class TitanicSurvivalPredictor(FlowSpec):

    """
    baseline
    """

    data = IncludeFile('d', default="data/titanic.csv")

    @card
    @step
    def start(self):
        """
        Load the data & train model
        """
        import pandas as pd
        from io import StringIO

        # NOTE: The data path in read_csv is relative to where you run command from.
        self.df = pd.read_csv(StringIO(self.data))
        self.next(self.predict)

    @step
    def predict(self):
        """
        make predictions
        """
        import pandas as pd
        from sklearn.metrics import accuracy_score

        self.df["preds"] = 0
        self.score = accuracy_score(self.df["Survived"], self.df["preds"])

        self.next(self.end)

    @step
    def end(self):
        """
        End of flow!
        """
        self.model_type = "baseline"
        print("Score = %s" % self.score)

if __name__ == "__main__":
    TitanicSurvivalPredictor()
