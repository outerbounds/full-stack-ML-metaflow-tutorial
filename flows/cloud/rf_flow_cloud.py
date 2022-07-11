
from metaflow import FlowSpec, step, card, conda
import json





class RF_Flow_cloud(FlowSpec):
    """
    train a random forest
    """
    @conda(libraries={'scikit-learn':'1.0.2'}) 
    @card
    @step
    def start(self):
        """
        Load the data
        """
        #Import scikit-learn dataset library
        from sklearn import datasets

        #Load dataset
        self.iris = datasets.load_iris()
        self.X = self.iris['data']
        self.y = self.iris['target']
        self.next(self.rf_model)
        
    @conda(libraries={'scikit-learn':'1.0.2'})
    @step
    def rf_model(self):
        """
        build random forest model
        """
        from sklearn.ensemble import RandomForestClassifier
        
        self.clf = RandomForestClassifier(n_estimators=10, max_depth=None,
            min_samples_split=2, random_state=0)
        self.next(self.train)

        
    @conda(libraries={'scikit-learn':'1.0.2'})       
    @step
    def train(self):
        """
        Train the model
        """
        from sklearn.model_selection import cross_val_score
        self.scores = cross_val_score(self.clf, self.X, self.y, cv=5)
        self.next(self.end)
        
        
    @step
    def end(self):
        """
        End of flow!
        """
        print("ClassificationFlow is all done.")


if __name__ == "__main__":
    RF_Flow_cloud()
