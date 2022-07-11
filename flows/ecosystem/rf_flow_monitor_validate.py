

from metaflow import FlowSpec, step, card
import json

class Combination_Flow(FlowSpec):
    """
    train a random forest
    """
    @card 
    @step
    def start(self):
        """
        Load the data
        """
        #Import scikit-learn dataset library
        from sklearn import datasets
        from sklearn.model_selection import train_test_split

        #Load dataset
        self.iris = datasets.load_iris()
        self.X = self.iris['data']
        self.y = self.iris['target']
        self.labels = self.iris['target_names']

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2)
        self.next(self.data_validation)
        

    @step
    def data_validation(self):
        """
        Perform data validation with great_expectations
        """
        import pandas as pd
        from utils import validate



        
        from sklearn import datasets
        iris = datasets.load_iris()
        df = pd.DataFrame(data=iris['data'], columns=iris['feature_names'])
        df["target"] = iris['target']
        df["petal length (cm)"][0] = -1

        validate(df)


        self.next(self.rf_model)
        
        
    @step
    def rf_model(self):
        """
        build random forest model
        """
        from sklearn.ensemble import RandomForestClassifier
        
        
        self.clf = RandomForestClassifier(n_estimators=10, max_depth=None,
            min_samples_split=2, random_state=0)
        self.next(self.train)

        
        
    @step
    def train(self):
        """
        Train the model
        """
        from sklearn.model_selection import cross_val_score
        self.clf.fit(self.X_train, self.y_train)
        self.y_pred = self.clf.predict(self.X_test)
        self.y_probs = self.clf.predict_proba(self.X_test)
        self.next(self.monitor)
        

    
        
    @step
    def monitor(self):
        """
        plot some things using an experiment tracker
        
        """
        import wandb
        wandb.init(project="mf-rf-wandb", entity="hugobowne", name="mf-tutorial-iris")

        wandb.sklearn.plot_class_proportions(self.y_train, self.y_test, self.labels)
        wandb.sklearn.plot_learning_curve(self.clf, self.X_train, self.y_train)
        wandb.sklearn.plot_roc(self.y_test, self.y_probs, self.labels)
        wandb.sklearn.plot_precision_recall(self.y_test, self.y_probs, self.labels)
        wandb.sklearn.plot_feature_importances(self.clf)

        wandb.sklearn.plot_classifier(self.clf, 
                              self.X_train, self.X_test, 
                              self.y_train, self.y_test, 
                              self.y_pred, self.y_probs, 
                              self.labels, 
                              is_binary=True, 
                              model_name='RandomForest')

        wandb.finish()
        self.next(self.end)
    
    @step
    def end(self):
        """
        End of flow!
        """
        print("Combination_Flow is all done.")


if __name__ == "__main__":
    Combination_Flow()
