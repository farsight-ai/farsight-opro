import pytest
import os
from sklearn.model_selection import train_test_split
import json
from opro import (
    FarsightOPRO,
)

"""
To run tests run the following commands: 

echo "export OPENAI_API_KEY='yourkey'" >> ~/.zshrc
source ~/.zshrc
echo $OPENAI_API_KEY

"""
openai_key = os.environ["OPENAI_API_KEY"]
client = FarsightOPRO(openai_key=openai_key)


dataset_path = "./bbh/movie_recommendation.json"
with open(dataset_path, "r") as file:
    data = json.load(file)

train_set, test_set = train_test_split(
    data["examples"],
    train_size=0.02,
)


# Short Tests
def test_generate_optimized_prompts_basic():
    prompts_and_scores = client.generate_optimized_prompts(
        train_set, num_iterations=1, prompts_generated_per_iteration=25
    )
    assert isinstance(prompts_and_scores, list), "Output should be a list"
    assert len(prompts_and_scores) > 0, "Output list should not be empty"
    assert len(prompts_and_scores) <= 20, "Output list should not be greater than 20"


def test_generate_optimized_prompts_edge_case_empty_dataset():
    with pytest.raises(ValueError):
        client.generate_optimized_prompts(
            [], num_iterations=3, prompts_generated_per_iteration=2
        )

# Run the tests
if __name__ == "__main__":
    pytest.main([__file__])
