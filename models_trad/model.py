from sklearn.pipeline import FeatureUnion, Pipeline

from .methods import METHODS


class Model:
    def __init__(self, pipeline, unions):
        self.steps = self._create_steps(pipeline, unions)

    def create(self):
        self.model = Pipeline(steps=self.steps)
        return self.model

    def _create_steps(self, pipeline, unions):
        steps = list()
        for model_name in pipeline:
            # add features from pipeline
            if model_name in METHODS.keys():
                step = self._make_step(model_name)
                steps.append(step)

            # add combined features
            elif model_name in unions.keys():
                steps_cf = list()
                for model_name_cf in unions[model_name]:
                    if model_name_cf in METHODS.keys():
                        step = self._make_step(model_name_cf)
                        steps_cf.append(step)
                if steps_cf:
                    steps.append([model_name, FeatureUnion(steps_cf)])

            else:
                # if method not found
                steps.append([model_name, None])
        return steps

    def _make_step(self, model_name):
        if isinstance(METHODS[model_name], type):
            step = [model_name, METHODS[model_name]()]
        else:
            # if already initialized
            step = [model_name, METHODS[model_name]]
        return step
