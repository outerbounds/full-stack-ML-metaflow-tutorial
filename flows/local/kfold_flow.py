from metaflow import FlowSpec, step, Parameter

class KFoldFlow(FlowSpec):
    """
    Template for Metaflows.
    You can choose which steps suit your workflow.
    We have included the following common steps:
    - Start
    - Build and Score models
    """ 
    
    # pass Parameter assignment to the flow at runtime
    k = Parameter("k", default=10)

    @step
    def start(self):
        from sklearn.datasets import make_classification 
        self.features, self.labels = make_classification(n_samples = 100)
        self.next(self.kfold_split)
        
    @step
    def kfold_split(self):
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=self.k) # split into "k" folds based on Parameter defined at runtime
        self.split = [(train_idxs, valid_idx) for train_idxs, valid_idx in kf.split(self.features)]
        self.next(self.build_and_score_model, foreach="split") 

    @step
    def build_and_score_model(self):
        from sklearn.linear_model import LogisticRegression
        train_x, valid_x = self.features[self.input[0]], self.features[self.input[1]]
        train_y, valid_y = self.labels[self.input[0]], self.labels[self.input[1]]
        lr = LogisticRegression().fit(train_x, train_y)
        self.score = lr.score(valid_x, valid_y)
        self.next(self.join)
        
    @step
    def join(self, inputs):
        import numpy as np
        self.results = [i.score for i in inputs]
        out_msg = f"{round(np.mean(self.results), 3)} +/- {round(np.std(self.results), 3)}"
        print(f"mean +/- std across folds is " + out_msg)
        self.next(self.end)
    
    @step
    def end(self):
        pass


if __name__ == "__main__":
    KFoldFlow()
