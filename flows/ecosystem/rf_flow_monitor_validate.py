
from metaflow import FlowSpec, step, card
import json

class ClassificationFlow(FlowSpec):
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
        from ruamel import yaml
        import great_expectations as ge
        from great_expectations.core.batch import RuntimeBatchRequest

        context = ge.get_context()

        
        from sklearn import datasets
        iris = datasets.load_iris()
        df = pd.DataFrame(data=iris['data'], columns=iris['feature_names'])
        df["target"] = iris['target']
        # df["sepal length (cm)"][0] = -1


        checkpoint_config = {
            "name": "flowers-test-flow-checkpoint",
            "config_version": 1,
            "class_name": "SimpleCheckpoint",
            "run_name_template": "%Y%m%d-%H%M%S-flower-power",
            "validations": [
                {
                    "batch_request": {
                        "datasource_name": "flowers",
                        "data_connector_name": "default_runtime_data_connector_name",
                        "data_asset_name": "iris",
                    },
                    "expectation_suite_name": "flowers-testing-suite",
                }
            ],
        }
        context.add_checkpoint(**checkpoint_config)


        results = context.run_checkpoint(
            checkpoint_name="flowers-test-flow-checkpoint",
            batch_request={
                "runtime_parameters": {"batch_data": df},
                "batch_identifiers": {
                    "default_identifier_name": "<YOUR MEANINGFUL IDENTIFIER>"
                },
            },
        )
        context.build_data_docs()
        context.open_data_docs()

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
        print("ClassificationFlow is all done.")


if __name__ == "__main__":
    ClassificationFlow()