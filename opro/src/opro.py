import json
import matplotlib.pyplot as plt
import pandas as pd
import random
import time
from openai import OpenAI, AzureOpenAI
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class FarsightOPRO:
    def __init__(self, openai_key: str, azure_endpoint=None, api_version=None):
        self.key = openai_key
        if azure_endpoint and api_version:
            self.client = AzureOpenAI(
                api_key=openai_key,
                azure_endpoint=azure_endpoint,
                api_version=api_version,
            )
        else:
            self.client = OpenAI(openai_key)

    """
    Evaluates the accuracy of a response based on the given output and target pair. 

    params: 
        - output: str, output to be evaluated
        - target: str, expected output
    returns: 
        - score: int, 0 being incorrect and 1 being correct
    """
    def llm_evaluator(self, input: str, output: str, target: str):
        prompt = f"""
            You are an LLM evaluator, given this input "{input}" and this target output "{target}", is this output "{output}" correct?  
            Please answer yes or no.
        """

        chatCompletion = self.client.chat.completions.create(
            model="gpt-35-turbo",
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=1.0,
        )
        # update token counts and get output
        content = chatCompletion.choices[0].message.content
        print("target: ", target)
        print("output: ", output)
        print("evaluation: ", content)
        if "yes" in content or "Yes" in content:
            print("score: ", 1)
            return 1
        else:
            print("score: ", 0)
            return 0

    def generate_prompt(self, previous_prompts_and_scores, dataset):
        # Get the top 20 best prompts
        sorted_list = sorted(
            previous_prompts_and_scores, key=lambda x: x["score"], reverse=True
        )
        previous_prompts_and_scores = sorted_list[:20]

        # Get the average accorss all best prompts for this iteration
        best_prompts_avg = 0
        for pair in previous_prompts_and_scores:
            best_prompts_avg += pair["score"]

        print(
            "Average score accross 20 best prompts:",
            best_prompts_avg / len(previous_prompts_and_scores),
        )

        previous_instructions_str = ""
        for pair in previous_prompts_and_scores:
            prompt = pair["prompt"]
            score = pair["score"]
            previous_instructions_str += f"""
                text:
                {prompt}
                score:
                {score}
                """
        prompt = f"""Your task is to generate the instruction <INS>. Below are some previous instructions with their scores.
        The score ranges from 0 to 100.
        {previous_instructions_str}
        Below are some problems.
        Problem:
        Q: {dataset[0]['input']}
        A: <INS>
        Ground truth answer: {dataset[0]['target']}

        Q: {dataset[1]['input']}
        A: <INS>
        Ground truth answer: {dataset[1]['target']}

        Q: {dataset[2]['input']}
        A: <INS>
        Ground truth answer: {dataset[2]['target']}

        Generate an instruction that is different from all the instructions <INS> above, and has a higher score
        than all the instructions <INS> above. The instruction should begin with <INS> and end with </INS>.
        The instruction should be concise, effective, and generally applicable to all problems above."""

        return prompt, previous_prompts_and_scores

    """
    Given a system prompt, iterate over the entire dataset and gets the average of all output scores. 

    params: 
        - system_prompt: str, system prompt to be tested
        - dataset: list[tuple(input, target)], dataset to be tested
        - eval_function: function(output, target), function that returns a store from 0-1 based on how close the output is to the target output, with 0 being the worst score and 1 being the best score
    
    returns: 
        - score: float, average score between 0-1
    """

    def get_average_score(self, system_prompt, dataset, eval_function):
        score = 0
        global completion_tokens
        global prompt_tokens
        for i in tqdm(range(len(dataset)), desc=" evaluation progress", leave=False):
            test_input = dataset[i]["input"]
            target = dataset[i]["target"]
            input = f"""
            {system_prompt}
            {test_input}"""
            chatCompletion = self.client.chat.completions.create(
                model="gpt-35-turbo",
                messages=[
                    {"role": "user", "content": input},
                ],
                temperature=0.0,
            )
            output = chatCompletion.choices[0].message.content
            if eval_function:
                score += eval_function(input, output, target)
            elif target in output:
                score += 1
        score = score / len(dataset)
        return score

    """
    Using the OPRO methodology with a fraction of the iteration cycles, generates the top 20 best prompts iteratively for the given dataset.

    params: 
        - dataset: list of tuples in the format (input, expected_output)
        - examples: list of 3 tuples in the format (input, expected_output) to be used as the examples in the prompt
        - eval_function: function that takes in (input, output, and expected_output) and returns a score from 0-1, 0 being the worst possible score 
            and 1 being the best possible score
    """

    def train(
        self,
        dataset,
        examples,
        num_iterations=40,
        prompts_generated_per_iteration=8,
        eval_function=None,
    ):
        scorer = eval_function if eval_function else self.llm_evaluator

        # initialize price tracking variables
        completion_tokens = 0
        prompt_tokens = 0
        # train the opro package
        # ORPO starts each prompt optimization with the same first prompt
        first_step_prompt = "Let’s solve the problem"

        # generate score for starting prompt and add to dictionary
        first_step_score = self.get_average_score(first_step_prompt, dataset, scorer)
        previous_prompts_and_scores = [
            {"prompt": first_step_prompt, "score": first_step_score}
        ]

        # initialize variables
        x = []
        y = []
        start_time = time.time()

        # start prompt optimization iterations
        for i in range(num_iterations):
            print(
                "------------------------------------------------------------------------------------"
            )
            print("Iteration: ", i)

            prompt, previous_prompts_and_scores = self.generate_prompt(
                previous_prompts_and_scores, examples
            )
            avg_score = 0
            print(prompt)

            # generate multiple system prompts each iteration
            for j in range(prompts_generated_per_iteration):
                chatCompletion = self.client.chat.completions.create(
                    model="gpt-35-turbo",
                    messages=[
                        {"role": "user", "content": prompt},
                    ],
                    temperature=1.0,
                )
                # update token counts and get output
                output = chatCompletion.choices[0].message.content
                completion_tokens += chatCompletion.usage.completion_tokens
                prompt_tokens += chatCompletion.usage.prompt_tokens

                generated_system_prompt = output.replace("<INS>", "")
                generated_system_prompt = generated_system_prompt.replace("</INS>", "")

                # get score and update variables
                score = self.get_average_score(generated_system_prompt, dataset, scorer)
                previous_prompts_and_scores.append(
                    {"prompt": generated_system_prompt, "score": score}
                )
                avg_score += score

                # update plot
                x.append(i)
                y.append(score)

            # plot accuracy graph
            plt.scatter(x, y)
            plt.title("Prompt Optimization Accuracy")
            plt.xlabel("Iterations")
            plt.ylabel("Percent Accurate")
            # plt.show()

            print("Average score across 8 generated prompts:", avg_score / 8)

        # Calculate Cost
        input_cost = prompt_tokens / 1000 * 0.001
        output_cost = completion_tokens / 1000 * 0.002
        total_cost = input_cost + output_cost
        # plt.show()

        print(
            "------------------------------------------------------------------------------------"
        )
        print("Total Training Time: --- %s seconds ---" % (time.time() - start_time))
        print("Total Cost: $", total_cost)
        for pair in previous_prompts_and_scores:
            print("Score: ", pair["score"], "Prompt: ", pair["prompt"])
        return previous_prompts_and_scores
