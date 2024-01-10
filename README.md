
## Revolutionize your prompt engineering with Farsight's OPRO SDK  
Stop wasting time with prompt engineering, tailored to your unique inputs and targets, our sdk effortlessly identifies the optimal system prompt for you.
 
<img src="opro/src/images/readme_cartoon.png" alt="cartoon" width="500"/>




## Features

- **Python Compatibility:** Seamlessly integrates with Python environments.
- **Efficient Prompt Optimization:** Streamlines the prompt optimization process in a single step.
- **Automated Testing:** Robust testing features for reliability and accuracy.
- **User-Friendly Design:** Intuitive and straightforward to use, regardless of your expertise level.

## Installation

Install the Farsight OPRO SDK with ease:

```bash
pip install farsight-opro
```

## Usage

Dive into the world of optimized prompt engineering with Farsight's OPRO SDK, an implementation inspired by the innovative [OPRO paper](https://arxiv.org/abs/2309.03409). Whether you're working on a small project or a large-scale application, our SDK adapts to your needs.

Begin optimizing promptly with this simple setup. For comprehensive guidance, visit our detailed [documentation](https://api.farsight-ai.com/farsight-opro/).

```bash
from opro import FarsightOPRO

# Define your datasets
train_set, examples, test_set = # ...

# Initialize with your OpenAI key
farsight = FarsightOPRO(openai_key=OPEN_AI_KEY)

# Get optimized prompts
prompts_and_scores = farsight.get_prompts(train_set, examples, test_set)
```

### Full Example:

```bash
from opro import FarsightOPRO
import json
from sklearn.model_selection import train_test_split

# Set your OpenAI credentials
farsight = FarsightOPRO(openai_key="<openai_key>")

# Load your dataset
dataset_path = "/opro/src/bbh/movie_recommendation.json"
with open(dataset_path, "r") as file:
    data = json.load(file)

# Split the dataset
examples = data["examples"][:3]  # Choose 3 examples for the prompt
train_set, test_set = train_test_split(data["examples"], train_size=0.20)

# Train and get prompts with scores
prompts_and_scores = farsight.train(train_set, examples, test_set)

# Output example
#  [{
#    "prompt": "Select a movie matching the genres, popularity, critical acclaim, and quality of provided examples for accurate recommendations.",
#    "train_score": 0.94,
#    "test_score": 0.88
#   }, ...]
```

## Contributing

#### Bug Reports & Feature Requests

Encounter an issue or have an idea? Share your feedback on our [issue tracker](https://github.com/farsight-ai/farsight-opro/issues).

#### Development Contributions

Your contributions are welcome! Join us in refining and enhancing our prompt optimization library.
