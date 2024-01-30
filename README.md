
## Revolutionize your prompt engineering with Farsight's OPRO SDK   
Stop wasting time with prompt engineering, tailored to your unique inputs and targets, our sdk effortlessly identifies the optimal system prompt for you.
 
<img src="images/readme_cartoon.png" alt="cartoon" width="500"/>




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

## Cost
When you utilize this package, please be aware that it will incur costs to your OpenAI account.

## Usage

Dive into the world of optimized prompt engineering with Farsight's OPRO SDK, an implementation inspired by the innovative [OPRO paper](https://arxiv.org/abs/2309.03409). Whether you're working on a small project or a large-scale application, our SDK adapts to your needs.

Begin optimizing promptly with this simple setup. For comprehensive guidance, visit our detailed [documentation](https://api.farsight-ai.com/farsight-opro/).

```bash
from opro import FarsightOPRO

# Define your datasets
dataset = # ...

# Initialize with your OpenAI key
farsight = FarsightOPRO(openai_key=OPEN_AI_KEY)

# Get optimized prompts
prompts_and_scores = farsight.generate_optimized_prompts(dataset)
```

### Full Example:

```bash
import json
import random
from sklearn.model_selection import train_test_split
from opro import FarsightOPRO

# replace with your openAI credentials
OPEN_AI_KEY = "<openai_key>"
farsight = FarsightOPRO(openai_key=OPEN_AI_KEY)

# load dataset
dataset_path = "/content/movie_recommendation.json"
with open(dataset_path, "r") as file:
    data = json.load(file)

# split dataset
dataset, test_set = train_test_split(
    data["examples"],
    train_size=0.4
)

##################### For a short test run, try this #####################

# dataset = full_dataset = [
#    {'input': 'Find a movie similar to Batman, The Mask, The Fugitive, Pretty Woman:\nOptions:\n(A) The Front Page\n(B) Maelstrom\n(C) The Lion King\n(D) Lamerica','target': '(C)'},
#    {'input': 'Find a movie similar to The Sixth Sense, The Matrix, Forrest Gump, The Shawshank Redemption:\nOptions:\n(A) Street Fighter II The Animated Movie\n(B) The Sheltering Sky\n(C) The Boy Who Could Fly\n(D) Terminator 2 Judgment Day',
# 'target': '(D)'},
#    {'input': "Find a movie similar to Schindler's List, Braveheart, The Silence of the Lambs, Tombstone:\nOptions:\n(A) Orlando\n(B) Guilty of Romance\n(C) Forrest Gump\n(D) All the Real Girls", 'target': '(C)'},
#    {'input': "Find a movie similar to Terminator 2 Judgment Day, The Fugitive, The Shawshank Redemption, Dead Man Walking:\nOptions:\n(A) Walk\n(B) Don't Run\n(C) Shaun the Sheep Movie\n(D) Rocky IV\n(E) Braveheart", 'target': '(E)'},
#    {'input': "Find a movie similar to Braveheart, The Mask, The Fugitive, Batman:\nOptions:\n(A) Club Dread\n(B) George Washington\n(C) Schindler's List\n(D) Once Upon a Time in America", 'target': '(C)'},
#    {'input': 'Find a movie similar to Heat, The Fugitive, Forrest Gump, The Silence of the Lambs:\nOptions:\n(A) Death Race 2\n(B) Cannonball Run II\n(C) Independence Day\n(D) Slumber Party Massacre II', 'target': '(C)'},
#    {'input': 'Find a movie similar to Raiders of the Lost Ark, The Shawshank Redemption, Inception, Pulp Fiction:\nOptions:\n(A) Beyond the Poseidon Adventure\n(B) The Chorus\n(C) Forrest Gump\n(D) Scouts Guide to the Zombie Apocalypse', 'target': #'(C)'},
#    {'input': "Find a movie similar to Schindler's List, Pulp Fiction, Braveheart, The Usual Suspects:\nOptions:\n(A) 12 Angry Men\n(B) Mo' Better Blues\n(C) Mindhunters\n(D) The Shawshank Redemption", 'target': '(D)'},
#    {'input': "Find a movie similar to Jurassic Park, The Silence of the Lambs, Schindler's List, Braveheart:\nOptions:\n(A) A Hard Day's Night\n(B) Showtime\n(C) Forrest Gump\n(D) Eddie the Eagle", 'target': '(C)'},
#    {'input': "Find a movie similar to The Lord of the Rings The Two Towers, The Lord of the Rings The Fellowship of the Ring, Star Wars Episode IV - A New Hope, The Matrix:\nOptions:\n(A) The Return\n(B) The Hidden Fortress\n(C) Schindler's 
# List\n(D) The End of the Affair", 'target': '(C)'}]
# prompts_and_scores = farsight.generate_optimized_prompts(dataset, prompts_generated_per_iteration=2, num_iterations=3)
# print(prompts_and_scores)

########################################################################


# get optimized prompts
prompts_and_scores = farsight.generate_optimized_prompts(dataset, test_set)
print(prompts_and_scores) 
#  [{
#        "prompt": "Choose the movie option that aligns with the given movies' genres, popularity, critical acclaim, and overall quality to provide the most accurate and comprehensive recommendation."
#        "score": 0.94,
#        "test_score": 0.88
#
#   },
#   {
#        "prompt": "Choose the movie option that aligns with the genres, themes, popularity, critical acclaim, and overall quality of the given movies to provide the most accurate and comprehensive recommendation."
#        "score": 0.9,
#        "test_score": 0.86
#   }, ...
```

## Contributing

#### Bug Reports & Feature Requests

Encounter an issue or have an idea? Share your feedback on our [issue tracker](https://github.com/farsight-ai/farsight-opro/issues).

#### Development Contributions

Your contributions are welcome! Join us in refining and enhancing our prompt optimization library.
