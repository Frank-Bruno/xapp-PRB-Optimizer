import re
import json
from openai import OpenAI, api_key
import os                  
from dotenv import load_dotenv 
import traceback
import importlib.util
from pathlib import Path
import numpy as np
from crewai import Agent, Crew, Process, Task, LLM
from crewai.tools import tool


env_path = Path("/tmp/src/.env") #path to .env file with OPENAI_API_KEY
load_dotenv(dotenv_path=env_path)


class LLMAgent:
    
    def __init__(self):
        self.code_name = "/opt/rl-models/reward_function.py" #"./reward_function.py"
        self.code_path = Path(self.code_name).resolve()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Error: the environment variable 'OPENAI_API_KEY' was not found.")

        self.api_key = api_key
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

    def run_reward_function(self, slice_obs, slice_req):
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

            resultado = module.reward_function(obs_arr, slice_req_arr)
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
        
        # =========================================================================
        # AUDITORIA MATEMÁTICA DE RECOMPENSA E FERRAMENTAS DO AGENTE DE SLA
        # =========================================================================

    @staticmethod
    def audit_reward_landscape(reward_func, slice_req: list, case_num: int = 2, target_slice_idx: int = 0) -> dict:
        """
        Executes mathematical stress tests on the candidate reward function according to the
        identified Use Case (1, 2, 3, or 4) and target prioritized slice index.

        Parameters:
            reward_func (callable): The compiled reward_function(slice_obs, slice_req).
            slice_req (list): List of slice requirements in Mbps (e.g. [25.0, 10.0, 5.0]).
            case_num (int): Use case identifier (1 to 4).
            target_slice_idx (int): 0-based index of prioritized slice for Case 2.

        Returns:
            dict: Mathematical vulnerability and robustness report.
        """
        req = np.array(slice_req, dtype=float)
        n = len(req)
        if n < 2:
            return {
                "case_tested": int(case_num),
                "reward_compliant": 0.0,
                "risk_detected": False,
                "verdict": "ROBUST",
                "details": "Insufficient slices to perform multi-slice stress tests."
            }

        # 1. Compliant state (ideal baseline where all slices meet exact targets)
        obs_compliant = req.copy()
        try:
            r_compliant = reward_func(obs_compliant, req)
        except Exception as e:
            return {
                "case_tested": int(case_num),
                "reward_compliant": 0.0,
                "risk_detected": True,
                "verdict": "VULNERABLE",
                "error": f"Function crashed on compliant baseline input: {str(e)}"
            }

        vulnerabilities = []
        metrics = {
            "case_tested": int(case_num),
            "reward_compliant": round(float(r_compliant), 4),
        }

        # Check basic numerical validity
        try:
            r_zero = reward_func(np.zeros_like(req), req)
            if not np.isfinite(r_zero):
                vulnerabilities.append("Non-finite reward (NaN or Inf) returned on zero observations.")
        except Exception as e:
            vulnerabilities.append(f"Function raised exception on zero observations: {str(e)}")

        if case_num == 1:
            # Case 1: Satisfy requirements; exceeding is allowed / not penalized.
            for i in range(n):
                obs_def = req.copy()
                obs_def[i] = max(0.0, req[i] * 0.8)
                r_def = reward_func(obs_def, req)
                if r_def >= r_compliant - 1e-4:
                    vulnerabilities.append(f"Slice {i} deficit is not penalized (r_deficit={r_def:.4f} >= r_compliant={r_compliant:.4f}).")

            for i in range(n):
                obs_exc = req.copy()
                obs_exc[i] = req[i] * 1.5
                r_exc = reward_func(obs_exc, req)
                if r_exc < r_compliant - 1e-3:
                    vulnerabilities.append(f"Slice {i} is penalized for exceeding requirements in Case 1 (r_exceed={r_exc:.4f} < r_compliant={r_compliant:.4f}).")

        elif case_num == 2:
            # Case 2: Maximize prioritized slice while other slices meet requirements without waste.
            prio_idx = int(target_slice_idx) if (0 <= int(target_slice_idx) < n) else 0
            metrics["prioritized_slice"] = prio_idx

            cannibalized_rewards = []
            waste_rewards = []

            for i in range(n):
                if i == prio_idx:
                    continue

                # A) Symmetric exchange: divert ~20% (up to 2.0 Mbps) from slice i to prioritized slice
                steal_sym = min(2.0, max(0.5, req[i] * 0.2))
                obs_sym = req.copy()
                obs_sym[prio_idx] += steal_sym
                obs_sym[i] = max(0.0, obs_sym[i] - steal_sym)
                cannibalized_rewards.append(reward_func(obs_sym, req))

                # B) Small deficit: divert ~10% (up to 1.0 Mbps) from slice i to prioritized slice
                steal_small = min(1.0, max(0.2, req[i] * 0.1))
                obs_small = req.copy()
                obs_small[prio_idx] += 1.0
                obs_small[i] = max(0.0, obs_small[i] - steal_small)
                cannibalized_rewards.append(reward_func(obs_small, req))

                # C) Asymmetric gain: +5.0 Mbps on prioritized slice
                obs_asym = req.copy()
                obs_asym[prio_idx] += 5.0
                obs_asym[i] = max(0.0, obs_asym[i] - steal_sym)
                cannibalized_rewards.append(reward_func(obs_asym, req))

                # D) Waste / Over-provisioning on constrained slice
                waste_val = max(5.0, req[i] * 0.5)
                obs_waste = req.copy()
                obs_waste[i] += waste_val
                waste_rewards.append(reward_func(obs_waste, req))

            r_cannibalized = max(cannibalized_rewards) if cannibalized_rewards else r_compliant
            r_waste = max(waste_rewards) if waste_rewards else r_compliant

            metrics["reward_cannibalized"] = round(float(r_cannibalized), 4)
            metrics["reward_waste"] = round(float(r_waste), 4)

            # Check 1: SLA cannibalization
            if r_cannibalized >= r_compliant - 1e-4:
                vulnerabilities.append(f"Cannibalization profitable: sacrificing SLA of a constrained slice gives r={r_cannibalized:.4f} >= r_compliant={r_compliant:.4f}.")

            # Check 2: Waste on constrained slices
            if r_waste >= r_compliant - 1e-4:
                vulnerabilities.append(f"Waste/over-provisioning on constrained slices is not penalized (r_waste={r_waste:.4f} >= r_compliant={r_compliant:.4f}).")

            # Check 3: Active gradient / non-flatness on prioritized slice
            obs_boost_20 = req.copy()
            obs_boost_20[prio_idx] += max(5.0, req[prio_idx] * 0.2)
            r_boost_20 = reward_func(obs_boost_20, req)

            obs_boost_50 = req.copy()
            obs_boost_50[prio_idx] += max(10.0, req[prio_idx] * 0.5)
            r_boost_50 = reward_func(obs_boost_50, req)

            metrics["reward_prio_boost_20pct"] = round(float(r_boost_20), 4)
            metrics["reward_prio_boost_50pct"] = round(float(r_boost_50), 4)

            if r_boost_20 <= r_compliant + 1e-4:
                vulnerabilities.append("Prioritized slice reward is FLAT/unresponsive: increasing prioritized throughput yields zero reward gain.")
            elif (r_boost_50 - r_boost_20) < 0.02:
                vulnerabilities.append("Prioritized slice reward prematurely SATURATED: increasing prioritized throughput from +20% to +50% yields practically zero marginal reward (<0.02).")

        elif case_num == 3:
            # Case 3: Save resources (penalize deviations both below and above requirements)
            for i in range(n):
                obs_def = req.copy()
                obs_def[i] = max(0.0, req[i] * 0.8)
                r_def = reward_func(obs_def, req)
                if r_def >= r_compliant - 1e-4:
                    vulnerabilities.append(f"Slice {i} deficit is not penalized in resource saving (Case 3).")

                obs_exc = req.copy()
                obs_exc[i] = req[i] + max(3.0, req[i] * 0.3)
                r_exc = reward_func(obs_exc, req)
                if r_exc >= r_compliant - 1e-4:
                    vulnerabilities.append(f"Slice {i} excess throughput is not penalized in resource saving (Case 3).")

        elif case_num == 4:
            # Case 4: Weighted optimization across slices
            obs_boost = req * 1.3
            r_boost = reward_func(obs_boost, req)
            metrics["reward_boosted"] = round(float(r_boost), 4)
            if not np.isfinite(r_boost):
                vulnerabilities.append("Non-finite reward computed in Case 4.")

        is_vulnerable = len(vulnerabilities) > 0
        metrics["risk_detected"] = is_vulnerable
        metrics["verdict"] = "VULNERABLE" if is_vulnerable else "ROBUST"
        if vulnerabilities:
            metrics["vulnerabilities"] = vulnerabilities

        return metrics

    @staticmethod
    def extract_python_code(content: str) -> str:
        """Extrai código Python funcional de respostas de LLM (removendo markdown)."""
        if not content:
            return ""
        match = re.search(r'```(?:python)?\s*(.*?)\s*```', content, re.DOTALL)
        if match:
            return match.group(1).strip()
        return content.strip()

    @staticmethod
    def compile_reward_function(code_content: str):
        """Compila o código gerado em uma função chamável reward_function."""
        clean_code = LLMAgent.extract_python_code(code_content)
        local_scope = {
            'np': np,
            'numpy': np,
        }
        exec(clean_code, local_scope)
        if 'reward_function' not in local_scope:
            raise ValueError("A função 'reward_function' não foi encontrada no código fornecido.")
        return local_scope['reward_function'], clean_code

    # =========================================================================
    # PIPELINE MULTI-AGENTE (CREWAI)
    # =========================================================================
    def agent_pipeline(self, descricao_logica: str, slice_req: list = None):
        """
        Agent pipeline to generate, validate, audit against SLA cannibalization, and format the reward function.
        """
        api_key = getattr(self, "api_key", None) or os.getenv("OPENAI_API_KEY")
        llm = LLM(model=self.MODEL, temperature=0.1, api_key=api_key)

        # Definição da ferramenta de auditoria para o sla_auditor
        latest_audit_result = {}

        @tool("audit_reward_landscape_tool")
        def audit_reward_landscape_tool(
            reward_code: str, 
            slice_requirements: str = "",
            case_num: int = 2,
            target_slice_idx: int = 0
        ) -> str:
            """Executes mathematical stress tests on the candidate reward function according to the specific use case.
            
            Parameters:
              - reward_code: Python code string containing def reward_function(slice_obs, slice_req)
              - slice_requirements: Slice requirements as string or list (e.g. '[25.0, 10.0, 5.0]')
              - case_num: Identified use case number (1, 2, 3, or 4).
              - target_slice_idx: 0-based index of the prioritized slice for Case 2 (e.g. 0 for first slice).
            """
            target_req = slice_req if slice_req is not None else [50.0, 20.0, 10.0]
            if slice_requirements:
                nums = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", str(slice_requirements))
                if len(nums) >= 2:
                    target_req = [float(x) for x in nums]

            try:
                c_num = int(case_num)
            except Exception:
                c_num = 2

            try:
                t_idx = int(target_slice_idx)
            except Exception:
                t_idx = 0

            try:
                reward_func, _ = self.compile_reward_function(reward_code)
                report = self.audit_reward_landscape(reward_func, target_req, case_num=c_num, target_slice_idx=t_idx)
            except Exception as e:
                report = {
                    "case_tested": c_num,
                    "reward_compliant": 0.0,
                    "risk_detected": True,
                    "verdict": "VULNERABLE",
                    "error": f"Erro de compilação ou execução da função de recompensa: {str(e)}"
                }

            latest_audit_result.clear()
            latest_audit_result.update(report)
            return json.dumps(report, indent=2)

        # 1. Agentes da Pipeline
        specialist_rl = Agent(
            role='Specialist in Reinforcement Learning and Telecommunications',
            goal='Translate network operator requests (RAN) into precise reward function logic in Python.',
            backstory='You are a telecommunications engineer specializing in 5G/6G Network Slicing and Reinforcement Learning. You deeply understand the 4 resource allocation use cases (1: Meet requirements, 2: Maximize one slice, 3: Save resources, 4: Optimize all with weights) and apply distinct, unpolluted reward architectures for each case.',
            verbose=True,
            llm=llm
        )

        qa_reviewer = Agent(
            role='RL Code Validation Engineer',
            goal='Ensure the generated function strictly follows the variable rules, identifies conceptual reward-shaping issues, and enforces strict use-case isolation.',
            backstory='You are a meticulous QA engineer. Your role is to receive the RL specialist code and verify that it strictly conforms to its specific use case without cross-case contamination (e.g., never adding over-provisioning penalties to Case 1, never adding maximization terms to Case 3, and enforcing the exact penalty hierarchies required for each case). If there is an error, you fix the code.',
            verbose=True,
            llm=llm
        )

        sla_auditor = Agent(
            role='Reward Landscape and SLA Auditor',
            goal='Audit candidate reward functions using the audit_reward_landscape_tool to identify any mathematical incentive to violate SLAs to maximize other slices (cannibalization).',
            backstory='You are a mathematical validation and stress-testing specialist for reinforcement learning in RAN slicing. You use the audit_reward_landscape_tool to analyze whether candidate reward functions allow reward hacking, SLA cannibalization, or unpenalized waste. If vulnerabilities are detected, you provide clear mathematical diagnostics so the reviewer can correct the formulation.',
            tools=[audit_reward_landscape_tool],
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

        # 2. Definição do slice_req para teste de estresse
        if slice_req is None:
            extracted = self.get_requiriments(descricao_logica)
            if extracted is not None:
                valid = [float(x) for x in extracted if not np.isnan(x)]
                if len(valid) >= 2:
                    slice_req = valid
        elif isinstance(slice_req, np.ndarray):
            valid = [float(x) for x in slice_req if not np.isnan(x)]
            slice_req = valid if len(valid) >= 2 else slice_req.tolist()

        if slice_req is None or len(slice_req) < 2:
            slice_req = [50.0, 20.0, 10.0]

        # 3. Geração e Validação Inicial (Especialista + Revisor)
        generate_reward_task = Task(
            description=f'''Based on the operator request: "{descricao_logica}", identify which of the 4 Network Slicing use cases applies and create the logic.

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
        - The function name MUST be: def reward_function(slice_obs, slice_req):
        - The code must initialize reward = 0.0 and return reward at the end.
        - Verify that the mathematical logic reflects one of the 4 use cases.
        - Verify that the function uses a descending gradient as a way to facilitate convergence, meaning the code penalizes the agent proportionally to the error committed, in a non-linear way.
        - Verify that penalty weights enforce constraint dominance: the penalty for falling below any slice's SLA requirement must be strictly greater than the marginal reward gained by diverting those resources to the maximized slice, preventing constraint cannibalization (reward hacking where the agent starves constrained slices to boost the prioritized slice).
        - Ensure shortfall errors below requirements are normalized directly against the target requirement (error / req) rather than diluted by tolerance margins, ensuring an accurate and strong signal when SLA is violated.
        - Account for discrete PRB allocation granularity and radio channel variance by allowing an adaptive tolerance margin above requirements (typically between 10% and 20% of the requirement, e.g., max(2.0, 0.15 * req)), ensuring small natural fluctuations around the target are not penalized as waste.
        - Verify that penalty weights remain bounded and numerically stable: under-provisioning should dominate over-provisioning and maximization gains, while avoiding runaway coefficients that destabilize policy gradients.
        - Check common reward-shaping errors, such as not penalizing the agent for not meeting requirements.
        Correct any anomalies and return the full corrected code.''',
            expected_output='The validated and corrected Python code.',
            agent=qa_reviewer
        )

        initial_crew = Crew(
            agents=[specialist_rl, qa_reviewer],
            tasks=[generate_reward_task, validate_code_task],
            process=Process.sequential
        )

        print("Starting reward function generation...")
        generation_output = initial_crew.kickoff(inputs={"descricao_logica": descricao_logica})
        current_code = str(generation_output)

        # 4. Execução do Agente Auditor com Loop de Correção pelo Revisor
        max_audit_iterations = 3
        iteration = 0

        while iteration < max_audit_iterations:
            # Garante que o auditor sempre receba o código mais recente
            clean_code = self.extract_python_code(current_code)
            print(f"\n--- [SLA Auditor] Executing audit iteration {iteration + 1} with audit_reward_landscape_tool ---")
            
            # Reset do resultado antes de cada execução da auditoria
            latest_audit_result.clear()

            # Tarefa do Auditor: invocar a ferramenta audit_reward_landscape_tool e avaliar
            audit_task = Task(
                description=f'''You must audit the candidate reward function against mathematical vulnerabilities and reward-shaping flaws using your tool `audit_reward_landscape_tool`.

Operator Request:
"{descricao_logica}"

Current Reward Function Code:
```python
{clean_code}
```

Slice Requirements to test:
{slice_req}

Instructions:
1. Inspect the Operator Request and the candidate code to identify:
   - Which of the 4 Use Cases applies (1: Meet requirements, 2: Maximize one slice while meeting others, 3: Save resources, 4: Optimize all with weights).
   - If Use Case 2 applies, identify which slice index (0-based) must be prioritized (e.g. 0 for the first slice, 1 for the second slice, etc.). If not applicable, use 0.
2. Call your tool `audit_reward_landscape_tool` passing:
   - `reward_code`: the Python code of the reward function.
   - `slice_requirements`: "{slice_req}"
   - `case_num`: the identified use case number (1, 2, 3, or 4).
   - `target_slice_idx`: the zero-based index of the slice to prioritize (for Case 2).
3. Inspect the tool output JSON metrics and verdict.
4. Evaluate the mathematical soundness:
   - In Case 2: verify that SLA cannibalization is not profitable, that constrained slice waste is penalized, AND that the prioritized slice has an active, non-flat, non-prematurely-saturated reward gradient.
   - In other cases: verify compliance with that specific use case's objectives without cross-case contamination.
   - If vulnerabilities are reported, explain the exact mathematical flaws and provide concrete instructions for the QA Reviewer to repair the formula.
   - If ROBUST, confirm its mathematical robustness.
5. Conclude clearly with your diagnosis and verdict (ROBUST or VULNERABLE).
''',
                expected_output='Audit diagnosis citing the audit_reward_landscape_tool results and giving actionable recommendations if vulnerable.',
                agent=sla_auditor
            )

            auditor_crew = Crew(
                agents=[sla_auditor],
                tasks=[audit_task],
                process=Process.sequential
            )
            audit_output = auditor_crew.kickoff()
            audit_analysis = str(audit_output)

            # Fallback se a tool não registrou durante o kickoff
            if not latest_audit_result:
                try:
                    fallback_case = self.get_classification_prompt(descricao_logica) or 2
                    fallback_prio = self.k_indice(descricao_logica)
                    fallback_prio = fallback_prio if fallback_prio is not None else 0
                    reward_func, clean_code = self.compile_reward_function(current_code)
                    latest_audit_result.update(self.audit_reward_landscape(reward_func, slice_req, case_num=fallback_case, target_slice_idx=fallback_prio))
                except Exception as e:
                    latest_audit_result.update({
                        "reward_compliant": 0.0,
                        "risk_detected": True,
                        "verdict": "VULNERABLE",
                        "error": str(e)
                    })

            audit_report = dict(latest_audit_result)
            print(f"[SLA Auditor] Stress test metrics obtained: {audit_report}")

            has_problem = (
                audit_report.get("risk_detected", False)
                or audit_report.get("verdict") in ["VULNERABLE", "VULNERABLE_TO_CANNIBALIZATION", "VULNERAVEL_A_CANIBALIZACAO"]
                or "error" in audit_report
            )

            if not has_problem:
                print("[SLA Auditor] Function is ROBUST! No cannibalization incentive detected.")
                break

            print(f"[SLA Auditor] Cannibalization / vulnerability detected! Passing back to QA Reviewer to fix...")
            iteration += 1
            if iteration >= max_audit_iterations:
                print("[SLA Auditor] Max audit iterations reached. Proceeding to format final code.")
                break

            # Tarefa do Revisor para consertar a função com base no feedback do auditor
            review_fix_task = Task(
                description=f'''The previously reviewed reward function FAILED the stress test of audit_reward_landscape.

Operator Request:
"{descricao_logica}"

Current Code:
```python
{clean_code}
```

Stress Test Data from audit_reward_landscape:
- Test Report: {audit_report}
- Verdict: {audit_report.get('verdict')}
{f"- Vulnerabilities: {audit_report.get('vulnerabilities')}" if 'vulnerabilities' in audit_report else ""}
{f"- Error: {audit_report.get('error')}" if 'error' in audit_report else ""}

Auditor Analysis & Feedback:
{audit_analysis}

CRITICAL FIX REQUIRED:
Resolve all reported vulnerabilities:
1. Ensure the function strictly adheres to the requested use case without cross-case contamination.
2. If Case 2: penalty for falling below any SLA requirement must strictly dominate any marginal reward gained from exceeding requirements or maximizing another slice; AND the prioritized slice must maintain an active, non-saturated reward gradient across operating throughput ranges.
3. Keep penalty weights and objective rewards balanced so neither completely masks the other under real-world measurement fluctuations.
4. Ensure the function signature remains: def reward_function(slice_obs, slice_req):
5. Ensure reward is initialized to 0.0 and returned at the end.

Correct the code and return the full corrected Python code.''',
                expected_output='The corrected Python code resolving the identified vulnerabilities.',
                agent=qa_reviewer
            )

            reviewer_crew = Crew(
                agents=[qa_reviewer],
                tasks=[review_fix_task],
                process=Process.sequential
            )
            review_fix_output = reviewer_crew.kickoff()
            current_code = str(review_fix_output)
            clean_code = self.extract_python_code(current_code)

        # 5. Formatação Final e Gravação do Arquivo
        format_output_task = Task(
            description=f'''Take the validated and audited code below. Remove Markdown formatting and extra text. Leave only executable Python code and comments.
Only the reward_function function and its dependencies/comments should remain.

Code to format:
{current_code}''',
            expected_output='Only the Python code lines.',
            agent=output_formatter,
            output_file=self.code_name
        )

        formatter_crew = Crew(
            agents=[output_formatter],
            tasks=[format_output_task],
            process=Process.sequential
        )
        print("Formatting and saving final reward function...")
        formatter_result = formatter_crew.kickoff()
        final_code = str(formatter_result)

        # Fallback de gravação direta se necessário
        try:
            clean_final = self.extract_python_code(final_code)
            with open(self.code_name, "w") as f:
                f.write(clean_final)
        except Exception as e:
            print(f"DEBUG: Note writing to {self.code_name}: {e}")

        print(f"Pipeline complete! Function saved to '{self.code_name}'.")
        return final_code

# Aliases de compatibilidade a nível de módulo
audit_reward_landscape = LLMAgent.audit_reward_landscape
extract_python_code = LLMAgent.extract_python_code
compile_reward_function = LLMAgent.compile_reward_function

if __name__ == "__main__":

    agent = LLMAgent()

    test_case_description = """
    Consider a situation with three network slices, each with its own performance requirements. The requirements for each slice are 50, 20, and 10 Mbps, respectively.
    The goal is to meet the requirements of the first and second slices without exceeding or falling below them, while maximizing the performance of the third slice, where the highest performance is desired.
    """
    slice_requirements = agent.get_requiriments(test_case_description)
    print(f"\n[Result] Extracted requirements code: {slice_requirements}")
    resultado = agent.agent_pipeline(test_case_description)
