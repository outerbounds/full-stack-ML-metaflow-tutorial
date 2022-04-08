
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
        #df["petal length (cm)"][0] = -1

        # configuration for data validation checkpoint
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

        # results of data validation
        # then build and view docs
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

        self.next(self.end)

    
    @step
    def end(self):
        """
        End of flow!
        """
        print("ClassificationFlow is all done.")


if __name__ == "__main__":
    ClassificationFlow()
