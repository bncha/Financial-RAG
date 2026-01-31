from zenml import pipeline
from config.settings import Settings
from steps.evaluate.get_testset_data import get_testset_data
from steps.evaluate.evaluate_answer import evaluate_answer
from steps.evaluate.generate_answers import generate_answers
from steps.evaluate.store_evaluation import store_evaluation

@pipeline(enable_cache=False)
def evaluate(settings: Settings):
    dataset = get_testset_data(settings)

    answers_dataset = generate_answers(dataset)

    evaluate_dataset = evaluate_answer(answers_dataset, settings)

    store_evaluation(evaluate_dataset, settings)



