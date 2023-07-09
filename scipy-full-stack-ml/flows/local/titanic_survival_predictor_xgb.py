from metaflow import FlowSpec, step, card, conda_base, project, IncludeFile, S3
import pandas as pd


@conda_base(libraries={"conda-forge::xgboost": '1.5.1', "conda-forge::scikit-learn": '1.1.2', "conda-forge::pandas": '1.4.2', "conda-forge::pyarrow": '11.0.0'})
@project(name="titanic_survival_prediction")
class TitanicSurvivalPredictor(FlowSpec):
    """
    train a boosted tree
    """

    data = IncludeFile('d', default="data/titanic.csv")
    max_depth = 6
    eta = 1

    @card
    @step
    def start(self):
        """
        Load the data & train model
        """
        from io import StringIO
        # NOTE: The data path in read_csv is relative to where you run command from.
        self.df = pd.read_csv(StringIO(self.data))
        self.next(self.predict)

    def featurize(self, df):
        TARGET='Survived'
        IGNORE_COLS = ['Name', 'Ticket']
        CATEGORICALS = ['Sex', 'Cabin', 'Embarked', 'Pclass', 'SibSp', 'Parch'] 
        df = pd.get_dummies(df, columns = CATEGORICALS)
        return df.drop(columns=[TARGET] + IGNORE_COLS), df[TARGET]

    @step
    def predict(self):

        """
        make predictions
        """

        import xgboost as xgb
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score

        features, labels = self.featurize(self.df)
        self.cols = features.columns
        train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2, random_state=42)

        self.model = xgb.XGBClassifier(max_depth=self.max_depth, eta=self.eta, objective='binary:logistic', nthread=8, eval_metric='auc', use_label_encoder=False)
        self.model.fit(train_features, train_labels, early_stopping_rounds=5, eval_set=[(test_features, test_labels)])
        self.preds = self.model.predict(test_features)
        self.score = accuracy_score(test_labels, self.preds)

        self.next(self.end)

    @step
    def end(self):
        """
        End of flow!
        """
        self.model_type = "xgboost"
        print("Score = %s" % self.score)

if __name__ == "__main__":
    TitanicSurvivalPredictor()
