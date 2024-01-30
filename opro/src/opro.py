import random
from openai import OpenAI, AzureOpenAI
from tqdm import tqdm


class FarsightOPRO:
    def __init__(
        self, openai_key: str, azure_endpoint=None, api_version=None, model=None
    ):
        self.key = openai_key
        self.model = model if model else "gpt-3.5-turbo"
        if azure_endpoint and api_version:
            self.client = AzureOpenAI(
                api_key=openai_key,
                azure_endpoint=azure_endpoint,
                api_version=api_version,
            )
        else:
            self.client = OpenAI(api_key=openai_key)

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
            model=self.model,
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=1.0,
        )
        # update token counts and get output
        content = chatCompletion.choices[0].message.content
        if "yes" in content or "Yes" in content:
            return 1
        else:
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

    def get_average_score(self, system_prompt, dataset, eval_function, sample_evals):
        score = 0
        global completion_tokens
        global prompt_tokens
        samples = []
        for i in range(len(dataset)):
            test_input = dataset[i]["input"]
            target = dataset[i]["target"]
            input = f"""
            {system_prompt}
            {test_input}"""
            chatCompletion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": input},
                ],
                temperature=0.0,
            )
            output = chatCompletion.choices[0].message.content
            if eval_function:
                sample_score = eval_function(input, output, target)
            elif target in output:
                sample_score = 1
            score += sample_score
            if sample_evals:
                samples.append(
                    {
                        "sample": test_input,
                        "target": target,
                        "output": output,
                        "score": sample_score,
                    }
                )
        score = score / len(dataset)
        return score, samples

    """
    Using the OPRO methodology with a fraction of the iteration cycles, generates the top 20 best prompts iteratively for the given dataset.

    params: 
        - dataset: list[dict] required: This should include pairs of "input" (the query or request) and "target" (the desired output). We recommend at least 50 samples for robust results.
        - test_dataset: similar to dataset, used for testing the efficacy of the generated prompts.
        - num_iterations: The total number of iterations for prompt optimization. Default is 40.
        - num_prompts_per_iteration: The number of different prompts generated per iteration. Default is 8.
        - sample_evaluations: If True, includes results of each sample evaluated in results. Default is False
        - eval_function: A custom scoring function that evaluates (input, output, expected_output) and returns a score between 0 and 1, with 1 being the best.
    output: 
        This includes up to 20 top system prompts, depending on your iteration settings. Each entry contains:
        - "prompt": always included, The generated system prompt.
        - "overall_score": always included,The average performance score across the dataset.
        - "test_score":  included if a test dataset is provided. Reflects the prompt's efficacy on the test set.
        - "sample_evals": included if sample evaluations is set to True. Contains evaluation results in the format list[dict], with the "sample" and the corresponding "score". 
    """

    def generate_optimized_prompts(
        self,
        dataset,
        test_set=None,
        num_iterations=40,
        prompts_generated_per_iteration=8,
        sample_evaluations=False,
        custom_score_function=None,
    ):
        if len(dataset) < 3:
            raise ValueError(
                "Training dataset must have at least 3 items. We recommend around 50 items."
            )
        scorer = custom_score_function if custom_score_function else self.llm_evaluator

        # initialize price tracking variables
        completion_tokens = 0
        prompt_tokens = 0
        # train the opro package
        # ORPO starts each prompt optimization with the same first prompt
        first_step_prompt = "Let’s solve the problem"

        # generate score for starting prompt and add to dictionary
        first_step_score, samples = self.get_average_score(
            first_step_prompt, dataset, scorer, sample_evaluations
        )
        if sample_evaluations:
            previous_prompts_and_scores = [
                {
                    "prompt": first_step_prompt,
                    "score": first_step_score,
                    "sample_evals": samples,
                }
            ]

        else:
            previous_prompts_and_scores = [
                {"prompt": first_step_prompt, "score": first_step_score}
            ]

        # start prompt optimization iterations
        for i in tqdm(range(num_iterations), desc=" evaluation progress", leave=True):
            examples = random.sample(dataset, 3)
            prompt, previous_prompts_and_scores = self.generate_prompt(
                previous_prompts_and_scores, examples
            )
            avg_score = 0

            # generate multiple system prompts each iteration
            for j in range(prompts_generated_per_iteration):
                chatCompletion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": prompt},
                    ],
                    temperature=1.0,
                )
                # update token counts and gets output
                output = chatCompletion.choices[0].message.content
                completion_tokens += chatCompletion.usage.completion_tokens
                prompt_tokens += chatCompletion.usage.prompt_tokens

                generated_system_prompt = output.replace("<INS>", "")
                generated_system_prompt = generated_system_prompt.replace("</INS>", "")

                # get score and update variables
                score, samples = self.get_average_score(
                    generated_system_prompt, dataset, scorer, sample_evaluations
                )
                results = {"prompt": generated_system_prompt, "score": score}

                if sample_evaluations:
                    results["sample_evals"] = samples

                previous_prompts_and_scores.append(results)
                avg_score += score
        if test_set:
            avg_test_score = 0

            for pair in previous_prompts_and_scores:
                prompt = pair["prompt"]

                (
                    test_score,
                    test_samples,
                ) = self.get_average_score(prompt, test_set, scorer, sample_evaluations)
                avg_test_score += test_score
                pair["test_score"] = test_score
                pair["sample_evals"] = test_samples

            # Get the top 20 best prompts
            sorted_list = sorted(
                previous_prompts_and_scores, key=lambda x: x["test_score"], reverse=True
            )
            previous_prompts_and_scores = sorted_list[:20]

        else:
            # Get the top 20 best prompts
            sorted_list = sorted(
                previous_prompts_and_scores, key=lambda x: x["score"], reverse=True
            )
            previous_prompts_and_scores = sorted_list[:20]
        return previous_prompts_and_scores
