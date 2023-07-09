"""

Template for writing Metaflows

"""

from metaflow import FlowSpec, step, current, card


class Template_Flow(FlowSpec):
    """
    Template for Metaflows.
    You can choose which steps suit your workflow.
    We have included the following common steps:
    - Start
    - Process data
    - Data validation
    - Model configuration
    - Model training
    - Model deployment
    """

    @card
    @step
    def start(self):
        """
        Start Step for a Flow;
        """
        print("flow name: %s" % current.flow_name)
        print("run id: %s" % current.run_id)
        print("username: %s" % current.username)

        # Call next step in DAG with self.next(...)
        self.next(self.process_raw_data)

    @step
    def process_raw_data(self):
        """
        Read and process data
        """
        print("In this step, you'll read in and process your data")

        self.next(self.data_validation)

    @step
    def data_validation(self):
        """
        Perform data validation
        """
        print("In this step, you'll write your data validation code")

        self.next(self.get_model_config)

    @step
    def get_model_config(self):
        """
        Configure model + hyperparams
        """
        print("In this step, you'll configure your model + hyperparameters")
        self.next(self.train_model)

    @step
    def train_model(self):
        """
        Train your model
        """
        print("In this step, you'll train your model")

        self.next(self.deploy)

    @step
    def deploy(self):
        """
        Deploy model
        """
        print("In this step, you'll deploy your model")

        self.next(self.end)

    @step
    def end(self):
        """
        DAG is done! Congrats!
        """
        print("DAG ended! Woohoo!")


if __name__ == "__main__":
    Template_Flow()
