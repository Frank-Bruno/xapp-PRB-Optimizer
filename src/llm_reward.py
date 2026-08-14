import re
from openai import OpenAI, api_key
import os                  
from dotenv import load_dotenv 
import traceback
import importlib.util
from pathlib import Path
import numpy as np
from crewai import Agent, Crew, Process, Task, LLM


env_path = Path("/tmp/src/.env") #path to .env file with OPENAI_API_KEY
load_dotenv(dotenv_path=env_path)

class LLMAgent:
    
    def __init__(self):
        self.code_name = "/opt/rl-models/reward_function.py" #"./reward_function.py"
        self.code_path = Path(self.code_name).resolve()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Error: the environment variable 'OPENAI_API_KEY' was not found.")

        self.client = OpenAI(api_key=api_key)
        self.MODEL = "gpt-5.2"
    
    def llm_reward_agent(self, descricao_logica_recompensa: str):
        """
        Receives a reward logic description, queries the LLM,
        and returns the corresponding Python code line.

        Args:
            descricao_logica_recompensa (str): The text describing the logic.
                                              Example: "penalize based on the average percentage deficit"

        Returns:
            str: The generated Python code line, or None if it fails.
        """

        prompt = self.criar_prompt(descricao_logica_recompensa)
        #prompt = self.prompt_test(descricao_logica_recompensa)

        try:
            response = self.client.chat.completions.create(
                model=self.MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )

            content = None
            try:
                choice = response.choices[0]
                msg = getattr(choice, "message", None) or (choice.get("message") if isinstance(choice, dict) else None)
                content = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else None)
            except Exception:
                content = None

            if not content:
                print("Error: could not extract content from the LLM response.")
                print("Raw response:", response)
                return None

            #Extract the Python code from the response
            match = re.search(r'```(?:python\n)?(.*?)\n?```', content, re.DOTALL)

            if match:
                codigo_gerado = match.group(1).strip()
            else:
                codigo_gerado = content.strip()

            codigo_final = codigo_gerado.split('\n')[0].strip()

            if not codigo_final:
                print("Error: the LLM returned an empty or invalid response.")
                return None

            print(f"Generated reward code:\n{codigo_gerado}")
            return codigo_final

        except Exception as e:
            print(f"An error occurred: {e}")
            print(f"Raw response received: {content if 'content' in locals() else 'N/A'}")
            return None

    def criar_prompt(self, descricao_logica: str):
        """
        Creates the structured prompt for the LLM by inserting the desired reward logic.
        """

        prompt_template = f"""
        You are an expert in Reinforcement Learning applied to Radio Access Networks (RAN) and network slicing.
        Your task is to determine which use case to use according to the provided logic description
        and then generate only the Python code that implements a reward calculation logic.

        **Variables Available in Scope:**
        * `reward` (float): The reward variable (initialized to 0.0).
        * `slice_obs` (np.ndarray): The current observation array.
        * `slice_req` (np.ndarray): The minimum requirement array.
        * `k` (int): Index of the element to be maximized when necessary.

        **Use Cases**
        1 - The first resource allocation use case aims to satisfy all requirements, possibly exceeding the requirements of each network slice. In this case, the reward function should focus on penalizing the agent for not meeting the requirement but does not consider whether the slices exceed the requirement.

        2 - The second resource allocation use case aims to maximize the performance of one of the slices while only meeting the requirements of the others, without exceeding the required value. To do this, the reward function should penalize the agent if the exact required value is not observed, and should reward the agent if the prioritized slice has the highest observed value possible. The index of the slice to be prioritized must be obtained from the logical description.

        3 - The third resource allocation use case aims to save resources by meeting only the requirement of all slices. In this case, the reward function should penalize the agent for any deviation between the observed values and the requirements so that excess resources are saved.

        4 - The fourth resource allocation use case aims to optimize the performance of all slices based on their performance metric. To do this, the reward function should reward the agent proportionally to the performance relative to the requirement for each slice. In addition, the function should assign a weight (+1) to throughput slices and (-1) to latency slices.

        **Task:**
        Given the logical description, map the description to one of the use cases and generate a reward function for a reinforcement learning agent for resource allocation in that case.

        For the first case, the reward function should be given by a variable reward that is the sum of the minimum between 0 and the difference between two arrays, slice_obs and slice_req, normalized by slice_req. The result of this calculation should be normalized by the system buffer variable.

        For the second case, the reward function should be expressed as a function of a delta variable calculated from the difference between the arrays slice_obs and slice_req. The variable k is also used to indicate the position in slice_obs of the slice to be prioritized. Iterating over the array elements, it checks simultaneously whether the current element is different from k and whether delta is less than zero, between 0 and 2, or greater than 2. If it is less than zero and the current element differs from k, the reward is given by -3 times delta normalized by a buffer times the hyperbolic tangent of the normalized delta. If delta is between 0 and 2 and the current element differs from k, the reward value is zero. If delta is greater than 2 and the current element differs from k, the reward is given by -1 times delta (normalized) times the hyperbolic tangent of the normalized delta. Finally, if the current element equals k, the reward is given by the absolute value of delta (normalized) times the hyperbolic tangent of the normalized delta. The reward function should be the sum of the rewards at each position of the slice_obs and slice_req vectors.

        For the third case, the reward function should be defined from simultaneous iteration over the slice_obs and slice_req arrays, calculating for each position i the error as the difference between the observed value and the required value. The reward assignment logic follows three conditions: if the observed value is strictly greater than the requirement plus two units, the reward for that index is the product of the negative error and the hyperbolic tangent of the error minus two units; if the observed value is greater than the requirement but does not exceed this two-unit limit, the reward added is zero; if the observed value is equal to or less than the requirement, the reward is calculated by multiplying the constant 3 by the hyperbolic tangent of the error and the negative of that same error. The function should return the sum of all accumulated rewards.

        For the fourth case, the reward function should be given by a variable reward that is the weighted sum of the ratio between two arrays, slice_obs and slice_req, where the weights are given by an array k, which contains only elements +1 or -1, indicated by the slice metric type. The result of this calculation should be normalized by the system buffer variable.

        **Reward Logic Requirements:**
        "{descricao_logica}"

        **Output Format:**
        Your output must be only the Python code line and nothing else. Do not include "python", backticks (```), or explanations.
        """
        return prompt_template.strip()

    def k_indice(self, descricao_logica: str):
        """
        Receives a DESCRIPTION of the reward logic, queries the LLM 
        and returns the corresponding k index.
        
        Returns:
            int: The generated k index, or None in case of failure.
        """
        prompt = f"""
        From the following phrase, indicate, if any, which index in a numpy array would the slice to have maximized performance:
        "{descricao_logica}"
        
        Say only the number (index). Do not say anything else. As it is a numpy array, if it is the first slice, return '0', if it is the second slice, return '1', and so on.
        If there is no slice to be maximized, answer 'None'.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )

            content = None
            try:
                choice = response.choices[0]
                msg = getattr(choice, "message", None) or (choice.get("message") if isinstance(choice, dict) else None)
                content = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else None)
            except Exception:
                content = None

            if not content:
                print("Error: could not extract content from LLM response for k_indice.")
                return None
            
            # Clean the response and try to convert to int
            # The LLM may respond "2" or "index 2" or "The index is 2."
            # We use regex to get the first number that appears.
            match = re.search(r'\d+', content)
            if match:
                indice_k = int(match.group(0))
                #print(f"K Index extracted: {indice_k}")
                return indice_k
            else:
                print(f"Error: Could not extract a number from the response: {content}")
                return None

        except Exception as e:
            print(f"An error occurred when calling the LLM for k_indice: {e}")
            return None
        
    def k_array(self, descricao_logica: str, slice_req):
        """
        Receives a DESCRIPTION of the reward logic, queries the LLM 
        and returns the corresponding k array.
        
        Returns:
            np.ndarray: The generated k array, or None in case of failure. Has the dimension of the number of slices
        """
        prompt = f"""
        From the following sentence, indicate, if any, what the k array of weights (+1 or -1) is for each slice in a numpy array, where +1 indicates throughput slices and -1 indicates latency slices:
        "{descricao_logica}"

        Say only the array in numpy format, for example: np.array([1, 1, -1, -1]), in the case of a scenario with 4 slices. Adapt the vector size to the number of slices. Say nothing else.
        If there is no slice to be maximized, respond 'None'.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )

            content = None
            try:
                choice = response.choices[0]
                msg = getattr(choice, "message", None) or (choice.get("message") if isinstance(choice, dict) else None)
                content = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else None)
            except Exception:
                content = None

            if not content:
                print("Error: could not extract content from LLM response for k_array.")
                return None
            
            array_match = re.search(r'np\.array\(\s*\[([-\d,\s]+)\]\s*\)', content)
            if array_match:
                array_str = array_match.group(1)
                array_values = [int(x.strip()) for x in array_str.split(',')]
                k_array = np.array(array_values)
                return k_array
            else:
                print(f"Error: Could not extract an array from the response: {content}")
                return None

        except Exception as e:
            print(f"An error occurred when calling the LLM for k_array: {e}")
            return None
    
    def k_type(self, descricao_logica: str, slice_req):
        """
        Receives the logical description and determines the type of k.
        """
        k = self.k_indice(descricao_logica)
        if type(k) is int:
            print("DEBUG: k_type:", k)
            return k
        k_arr = self.k_array(descricao_logica, slice_req)
        if isinstance(k_arr, np.ndarray):
            print("DEBUG: k_type:", k_arr)
            return k_arr
        print("DEBUG: k_type: None")
        return None
        
    def create_reward_function(self, intent, slice_req, k, output_file="reward_function.py"):
        """
        Creates the reward function based on the provided intent and saves it to a file.
        """

        try:
            codigo_str = self.llm_reward_agent(intent)
            #requirements_code = self.get_requiriments(intent)

            if codigo_str is None:
                raise ValueError("Error: 'codigo_str' is None. Check the output of 'llm_reward_agent'.")
            codigo_indentado = "\n".join("    " + line if line.strip() != "" else "" for line in codigo_str.splitlines())
            
            #k = self.k_type(intent) #Verifica o tipo de k a ser usado
            if isinstance(k, np.ndarray):
                k_arr = list(k)
                codigo_python = f"""import numpy as np
def reward_function(slice_obs, slice_req, buffer, k={k_arr}):
    reward=0.0
    slice_req = {list(slice_req)}
    k = np.array(k)
{codigo_indentado}
    return reward
"""

            elif isinstance(k, int):
                codigo_python = f"""import numpy as np
def reward_function(slice_obs, slice_req, buffer, k={k}):
    reward=0.0
{codigo_indentado}
    return reward
"""
            else:
                codigo_python = f"""import numpy as np
def reward_function(slice_obs, slice_req, buffer, k=None):
    reward=0.0
{codigo_indentado}
    return reward
"""

            with open(output_file, "w") as f:
                f.write(codigo_python)
            print(f"DEBUG: Generated reward function:\n{codigo_python}")
            return codigo_python, k
        except Exception as e:
            print(f"Error creating the reward function: {e}")

    def run_reward_function(self, slice_obs, slice_req, buffer):
        """
        Executes the reward function by loading the generated module and calling the function.
        Returns the reward value (or None in case of error).
        """
        #if not self.existing_code():
        #    print(f"Arquivo '{self.code_path}' não encontrado ou está vazio.")
        #    return None

        try:
            spec = importlib.util.spec_from_file_location("reward_function_module", str(self.code_path))
            if spec is None or spec.loader is None:
                raise FileNotFoundError(f"Could not load module spec from {self.code_path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            if not hasattr(module, "reward_function"):
                print("The generated module does not contain 'reward_function'.")
                return None

            # Garantir que a função receba tipos compatíveis (numpy arrays) para indexação/mascara
            obs_arr = np.array(slice_obs)
            slice_req_arr = np.array(slice_req)

            print(f"DEBUG: obs_arr={obs_arr}")
            print(f"DEBUG: slice_req_arr={slice_req_arr}")

            resultado = module.reward_function(obs_arr, slice_req_arr, buffer)
            return resultado

        except FileNotFoundError:
            print("Error: reward function file not found.")
        except Exception as e:
            print(f"An error occurred while executing the reward function: {e}")
            traceback.print_exc()
            return None

    def existing_code(self):
        """
        Verifies whether the code already exists.
        """
        print(f"DEBUG: Code path: {self.code_path}")
        if self.code_path.is_file() and self.code_path.stat().st_size > 0:
            print("DEBUG: The code exists and is not empty.")
            return True
        else:
            print("DEBUG: The code does not exist or is empty.")
            return False
        
    def get_classification_prompt(self, descricao_logica: str):
        """
        Classifies the logical description into one of 4 use cases.
        """
        prompt = f"""
        You are an expert in Radio Access Networks (RAN) and network slicing.
        Your task is to determine which use case to use according to the provided logical description.
        Analyze the operator request below and classify it into one of the 4 use cases.

        **USE CASE DEFINITIONS:**

        Case 1 - The first resource allocation use case aims to satisfy all requirements, possibly exceeding the requirements of each network slice. In this case, the reward function should focus on penalizing the agent for not meeting the requirement but does not consider whether the slices exceed the requirement.

        Case 2 - The second resource allocation use case aims to maximize the performance of one of the slices while only meeting the requirements of the others, without exceeding the required value. To do this, the reward function should penalize the agent if the exact required value is not observed, and should reward the agent if the prioritized slice has the highest observed value possible. The index of the slice to be prioritized must be obtained from the logical description.

        Case 3 - The third resource allocation use case aims to save resources by meeting only the requirement of all slices. In this case, the reward function should penalize the agent for any deviation between the observed values and the requirements so that excess resources are saved.

        Case 4 - The fourth resource allocation use case aims to optimize the performance of all slices based on their performance metric. To do this, the reward function should reward the agent proportionally to the performance relative to the requirement for each slice. In addition, the function should assign a weight (+1) to throughput slices and (-1) to latency slices.

        **OPERATOR REQUEST:**
        "{descricao_logica}"

        **OUTPUT:**
        Respond ONLY with the case number (1, 2, 3, or 4). Do not write anything else.
        """

        try:
            response = self.client.chat.completions.create(
                model=self.MODEL,
                messages=[
                    {"role": "system", "content": "Respond with only the class number."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
            )
            raw = response.choices[0].message.content.strip() if response.choices[0].message.content else ""
            clean = ''.join(filter(str.isdigit, raw))
            case_num = (int(clean) if clean else -1)
            print(f"DEBUG: Classified case: {case_num}")
            return case_num
        except Exception:
            print("Error while classifying the logical description.")

    def get_requiriments(self, descricao_logica: str):
        """
        Extracts the slice requirements from the logical description.
        """
        prompt = f"""
        You are an expert in Radio Access Networks (RAN) and network slicing.
        Your task is to extract the throughput and latency requirements of the slices from the provided logical description.

        **OPERATOR REQUEST:**
        "{descricao_logica}"

        **RULES:**
        1. When the metric is throughput, extract the values in Mbps.
        2. When the metric is latency, extract the values in s.
        3. When there is no explicit mention of latency, use the default value np.nan to indicate that this measure was not specified.
        4. When there is no explicit mention of throughput, use the default value np.nan to indicate that this measure was not specified.

        **OUTPUT:**
        Respond with two separate arrays, one for throughput and one for latency, in the format:
        [100, 200], [10, 30]
        """

        try:
            response = self.client.chat.completions.create(
                model=self.MODEL,
                messages=[
                    {"role": "system", "content": "Respond only with the two arrays separated."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
            )
            content = response.choices[0].message.content.strip() if response.choices[0].message.content else ""
            print(f"DEBUG: Extracted requirements: {content}")

            match = re.match(r'\[([^\]]+)\],\s*\[([^\]]+)\]', content)
            if match:
                throughput = np.array([
                    float(x.strip()) if x.strip() != "np.nan" else np.nan
                    for x in match.group(1).split(',')
                ])
                latency = np.array([
                    float(x.strip()) if x.strip() != "np.nan" else np.nan
                    for x in match.group(2).split(',')
                ])
                slice_requirements = np.concatenate((throughput, latency))
                
                # Formatar o array para retornar com "np.nan" ao invés de "nan"
                formatted = np.array2string(slice_requirements, separator=', ')
                formatted = formatted.replace('nan', 'np.nan')
                return slice_requirements
            else:
                print("DEBUG: Unexpected response format.")
                return None
        except Exception as e:
            print(f"DEBUG: Error extracting the requirements from the logical description: {e}")
            return None
        
        ##### TESTING NEW STRUCTURE ######
    def agent_pipeline(self, descricao_logica: str):
        """
        Agent pipeline to generate, validate, and format the reward function."""

        llm = LLM(model=self.MODEL, temperature=0.2, api_key=api_key)
        specialist_rl = Agent(
            role='Specialist in Reinforcement Learning and Telecommunications',
            goal='Translate network operator requests (RAN) into precise reward function logic in Python.',
            backstory='You are a telecommunications engineer specializing in 5G/6G Network Slicing and Reinforcement Learning. You deeply understand the 4 resource allocation use cases (1: Meet requirements, 2: Maximize one slice, 3: Save resources, 4: Optimize all with weights).',
            verbose=True,
            llm=llm
        )

        qa_reviewer = Agent(
            role='RL Code Validation Engineer',
            goal='Ensure the generated function strictly follows the variable rules, identifies conceptual reward-shaping issues, and has the correct signature.',
            backstory='You are a meticulous QA engineer. Your role is to receive the RL specialist code and verify that: 1) The function is named "reward_function". 2) It receives exactly the parameters (slice_obs, slice_req, buffer). 3) The "reward" variable is initialized to 0.0 and returned at the end. If there is an error, you fix the code.',
            verbose=True,
            llm=llm
        )

        output_formatter = Agent(
            role='Python Script Formatter',
            goal='Extract only functional Python code, removing any model conversation.',
            backstory='You are the guardian of the final file. You do not evaluate logic. Your only job is to remove greetings, explanations, and Markdown markers (such as ```python and ```). The result must be pure code.',
            verbose=True,
            llm=llm
        )

        generate_reward_task = Task(
            description='''Based on the operator request: "{descricao_logica}", identify which of the 4 Network Slicing use cases applies and create the logic.

        Use Cases:
            1 - The first resource allocation use case aims to satisfy all requirements, possibly exceeding the requirements of each network slice. In this case, the reward function should focus on penalizing the agent for not meeting the requirement but should not consider whether the slices exceed the requirement.
            2 - The second resource allocation use case aims to maximize the performance of one slice while only meeting the requirements of the others, without exceeding the required value. To do this, the reward function should penalize the agent if the required value is not observed and reward the agent if the prioritized slice has the highest observed value possible.
            3 - The third resource allocation use case aims to save resources by meeting only the requirement of all slices. In this case, the reward function should penalize the agent for any deviation between the observed values and the requirements so that excess resources are saved.
            4 - The fourth resource allocation use case aims to optimize the performance of all slices based on their performance metric. To do this, the reward function should reward the agent proportionally to the performance relative to the requirement for each slice. In addition, the function should assign a weight (+1) to throughput slices and (-1) to latency slices.

        Variables available for the equation:**
            * `reward` (float): The reward variable (initialized to 0.0).
            * `slice_obs` (np.ndarray): The current throughput observation array.
            * `slice_req` (np.ndarray): The minimum requirement array.
        Generate the Python function.''',
            expected_output='The initial Python code implementing the reward function.',
            agent=specialist_rl
        )

        validate_code_task = Task(
            description='''Analyze the function generated by the previous task.
        Mandatory rules:
        - The function name MUST be: def reward_function(slice_obs, slice_req, buffer):
        - The code must initialize reward = 0.0 and return reward at the end.
        - Verify that the mathematical logic reflects one of the 4 use cases.
        - Verify that the function uses a descending gradient as a way to facilitate convergence, meaning the code penalizes the agent proportionally to the error committed, in a non-linear way.
        - Verify that the function is sensitive to deviations, meaning the agent should be penalized more severely for deviations below requirements than for deviations above requirements to encourage compliance.
        - Do not normalize the data in the function; use raw values.
        - Consider tolerance values (eps) of 2 Mbps, only above the requirements, to facilitate convergence, meaning if the slice is within 2 Mbps of the requirement, consider it as meeting the requirement.
        - Check common reward-shaping errors, such as not penalizing the agent for not meeting requirements.
        Correct any anomalies and return the full corrected code.''',
            expected_output='The validated and corrected Python code.',
            agent=qa_reviewer
        )

        format_output_task = Task(
            description='''Take the validated code. Remove Markdown formatting and extra text. Leave only executable Python code and comments.
            Only the reward_function function and its dependencies/comments should remain.''',
            expected_output='Only the Python code lines.',
            agent=output_formatter,
            output_file=self.code_name
        )

        rl_team = Crew(
            agents=[specialist_rl, qa_reviewer, output_formatter],
            tasks=[generate_reward_task, validate_code_task, format_output_task],
            process=Process.sequential
        )

        print("Starting reward function generation...")

        resultado = rl_team.kickoff(inputs={"descricao_logica": descricao_logica})

if __name__ == "__main__":

    agent = LLMAgent()

    test_case_description = """
    Consider a situation with three network slices, each with its own performance requirements. The requirements for each slice are 50, 20, and 10 Mbps, respectively.
    The goal is to meet the requirements of the first and second slices without exceeding or falling below them, while maximizing the performance of the third slice, where the highest performance is desired.
    """
    slice_requirements = agent.get_requiriments(test_case_description)
    print(f"\n[Result] Extracted requirements code: {slice_requirements}")
    resultado = agent.agent_pipeline(test_case_description)
