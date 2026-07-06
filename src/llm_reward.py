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
        self.code_name = "./reward_function.py"
        self.code_path = Path(self.code_name).resolve()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Erro: A variável de ambiente 'OPENAI_API_KEY' não foi encontrada.")

        self.client = OpenAI(api_key=api_key)
        self.MODEL = "gpt-5.2"
    
    def llm_reward_agent(self, descricao_logica_recompensa: str):
        """
        Recebe uma DESCRIÇÃO da lógica de recompensa, consulta o LLM 
        e retorna a linha de código Python correspondente.
        
        Args:
            descricao_logica_recompensa (str): O texto que descreve a lógica.
                                                Ex: "penalizar pela média do déficit percentual"
        
        Returns:
            str: A linha de código Python gerada, ou None em caso de falha.
        """

        #prompt = self.criar_prompt(descricao_logica_recompensa)
        prompt = self.prompt_test(descricao_logica_recompensa)

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
                print("Erro: não foi possível extrair conteúdo da resposta do LLM.")
                print("Resposta bruta:", response)
                return None
            
            # Extrair o código Python da resposta
            match = re.search(r'```(?:python\n)?(.*?)\n?```', content, re.DOTALL)
            
            if match:
                codigo_gerado = match.group(1).strip()
            else:
                # Se não houver ```, assume que a resposta inteira é o código
                codigo_gerado = content.strip()

            # Pega a primeira linha do código gerado (conforme lógica original)
            codigo_final = codigo_gerado.split('\n')[0].strip()
            
            if not codigo_final:
                 print("Erro: O LLM retornou uma resposta vazia ou inválida.")
                 return None

            print(f"Código de recompensa gerado:\n{codigo_gerado}")
            return codigo_final

        except Exception as e:
            print(f"Ocorreu um erro: {e}")
            print(f"Resposta bruta recebida: {content if 'content' in locals() else 'N/A'}")
            return None

    def criar_prompt(self, descricao_logica: str):
        """
        Cria o prompt estruturado para o LLM, inserindo a lógica
        de recompensa desejada.
        """
        
        prompt_template = f"""
        Você é um especialista em Reinforcement Learning aplicado a Redes de Acesso Rádio (RAN) e network slicing.
        Sua tarefa é definir qual caso de uso utilizar de acordo com a lógica de descrição fornecida
        e em seguida gerar **apenas** o código Python que implementa uma lógica de cálculo de recompensa.
        
        **Variáveis Disponíveis no Escopo:**
        * `reward` (float): A variável de recompensa (inicializada em 0.0).
        * `slice_obs` (np.ndarray): O array de observações atuais.
        * `slice_req` (np.ndarray): O array de requisitos mínimos.
        * `k` (int): Indice do elemento a ser maximizado, quando necessário.

        **Casos de uso**
        1 - O primeiro caso de uso de alocação de recursos tem por objetivo cumprir todos os requisitos, podendo ultrapassar os requisitos de cada slice de rede. Nesse caso, a função de reward deve focar em punir o agente não cumprir o requisito mas não leva em consideração se os slices ultrapassam o requisito.

        2 - O segundo caso de uso de alocação de recursos tem por objetivo maximizar o desempenho de um dos slices e apenas cumprir o requisito dos outros, sem ultrapassar o valor do requisito. Para isso, a função de reward deve punir o agente se não for observado o exato valor do requisito, e deve recompensar o agente se o slice priorizado tiver o maior valor observado possível. Deve-se obter o índice do slice a ser priorizado a partir da descrição lógica.

        3 - O terceiro caso de uso de alocação de recursos tem por objetivo economizar recursos e assim cumprir apenas o requisito de todos os slices. Nesse caso, a função de reward deve punir o agente por qualquer desvio entre o observado e os requisitos, a fim de que o excedente de recursos seja economizado.
        
        4 - O quarto caso de uso de alocação de recursos tem por objetivo otimizar o desempenho de todos os slices a partir de sua métrica de desempenho. Para isso, a função de reward deve recompensar o agente proporcialmente ao desempenho em relação ao requisito para cada slice. Além disso, a função deve atribuir um peso (+1) para slices de vazão e (-1) para slices de latência. 

        
        **Tarefa:**
        Dada a descrição lógica, faça o mapeamento da descrição em um dos casos de uso e gere uma função de reward para um agente de apredizado por reforço para alocação de recursos desse caso.
        
        Para o primeiro caso, a função de reward deve ser dada por uma variável reward que seja dada pela soma do mínimo entre 0 e a diferença ente dois arrays, slice_obs e slice_req, normalizada pelo slice_req. O resultado desse cálculo deve ser normalizado pela variável buffer do sistema.
        
        Para o segundo caso, a função de reward deve ser dada em função de uma variável delta, calculada pela diferença entre os arrays slice_obs e slice_req. Também utiliza-se a variável k para indicar a posição em slice_obs do slice a ser priorizado. Iterando pelos elementos dos arrays, verifica-se simulataneamente se o elemento atual é diferente de k e se delta é menor que zero, entre 0 e 2 ou maior que dois. Caso seja menor que zero e o elemento atual diferente de k, a reward é dada por -3 vezes delta normalizado por um beuffer vezes a tangenete hiperbólica de delta (normalizado). Caso delta esteja entre 0 e 2 e o elemento atual seja diferente de k, o valor de reward é zero. Caso seja delta seja maior que 2 e o elemento atual diferente de k, a reward é dada por -1 vezes delta (normalizado) vezes tangente hiperbólica de delta (normalizado). Por fim, no caso de do elemento atual ser igual a k, a reward é dada pelo módulo de delta (normalizado) vezes a tangente hiperbólica de delta (normalizado). A função de reward feve ser dada pela soma dos rewards em cada posição dos vetores slice_obs e slice_req.
                
        Para o terceiro caso, a função de recompensa deve ser definida a partir da iteração simultânea sobre os arrays slice_obs e slice_req, calculando-se, para cada posição i, o erro como a diferença entre o valor observado e o valor requerido. A lógica de atribuição da recompensa segue três condições: se o valor observado for estritamente maior que o requisito somado a duas unidades, a recompensa para aquele índice é o produto do erro negativo pela tangente hiperbólica do erro subtraído de duas unidades; se o valor observado for maior que o requisito, mas não ultrapassar esse limite de duas unidades adicionais, o valor de recompensa adicionado é zero; caso o valor observado seja igual ou inferior ao requisito, a recompensa é calculada multiplicando-se a constante 3 pela tangente hiperbólica do erro e pelo valor negativo desse mesmo erro. A função deve retornar a soma total das recompensas acumuladas em todas as iterações.
        
        Para o quarto caso, a função de reward deve ser dada por uma variável reward que seja dada pela soma ponderada da razão entre dois arrays. slice_obs e slice_req, na qual os pesos são dados por um array k, o qual é só possui elementos +1 ou -1, indicados pelo tipo de métrica do slice. O resultado desse cálculo deve ser normalizado pela variável buffer do sistema.

        
        **Requisitos da Lógica de Recompensa:**
        "{descricao_logica}"

        **Formato da Saída:**
        Sua saída deve ser **apenas a linha de código Python** e nada mais. Não inclua "python", acentos graves (```), ou explicações.
        """
        return prompt_template.strip()
    

    def k_indice(self, descricao_logica: str):
        """
        Recebe uma DESCRIÇÃO da lógica de recompensa, consulta o LLM 
        e retorna o índice k correspondente.
        
        Returns:
            int: O índice k gerado, ou None em caso de falha.
        """
        prompt = f"""
        A partir da frase seguinte, indique , se houver, qual o índice em um numpy array estaria o slice a ter o desempenho maximizado:
        "{descricao_logica}"
        
        Diga apenas o número (índice). Não diga mais nada. Como é um numpy array, se for o primeiro slice, retorne '0', se for o segundo slice, retorne '1', e assim por diante.
        Se não houver slice a ser maximizado, responda 'None'.
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
                print("Erro: não foi possível extrair conteúdo da resposta do LLM para k_indice.")
                return None
            
            # Limpar a resposta e tentar converter para int
            # O LLM pode responder "2" ou "índice 2" ou "O índice é 2."
            # Usamos regex para pegar o primeiro número que aparecer.
            match = re.search(r'\d+', content)
            if match:
                indice_k = int(match.group(0))
                #print(f"Índice K extraído: {indice_k}")
                return indice_k
            else:
                print(f"Erro: Não foi possível extrair um número da resposta: {content}")
                return None

        except Exception as e:
            print(f"Ocorreu um erro ao chamar o LLM para k_indice: {e}")
            return None
        
    def k_array(self, descricao_logica: str, slice_req):
        """
        Recebe uma DESCRIÇÃO da lógica de recompensa, consulta o LLM 
        e retorna o array k correspondente.
        
        Returns:
            np.ndarray: O array k gerado, ou None em caso de falha. Possui a dimensão do número de slices
        """
        prompt = f"""
        A partir da frase seguinte, indique , se houver, qual o array k de pesos (+1 ou -1) para cada slice em um numpy array, onde +1 indica slices de vazão e -1 indica slices de latência:
        "{descricao_logica}"

        Diga apenas o array no formato numpy, por exemplo: np.array([1, 1, -1, -1]), no caso de um cenário com 4 slices. Adapte o tamanho do vetor para o número de slices. Não diga mais nada.
        Se não houver slice a ser maximizado, responda 'None'.
        """
        #O vetor abaixo contém os requisitos de throughput (em Mbps) e latência (em s) para os slices descritos, sendo que a primeira metade do array corresponde aos requisitos de throughput e a segunda metade aos requisitos de latência:
        #"{slice_req}"
        
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
                print("Erro: não foi possível extrair conteúdo da resposta do LLM para k_array.")
                return None
            
            # Tentar avaliar o array retornado
            array_match = re.search(r'np\.array\(\s*\[([-\d,\s]+)\]\s*\)', content)
            if array_match:
                array_str = array_match.group(1)
                array_values = [int(x.strip()) for x in array_str.split(',')]
                k_array = np.array(array_values)
                #print(f"Array K extraído: {k_array}")
                return k_array
            else:
                print(f"Erro: Não foi possível extrair um array da resposta: {content}")
                return None

        except Exception as e:
            print(f"Ocorreu um erro ao chamar o LLM para k_array: {e}")
            return None
    
    def k_type(self, descricao_logica: str, slice_req):
        """
        Recebe a descrição lógica e determina o tipo de k.
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
        Cria a função de recompensa com base no intent fornecido e salva em um arquivo.
        """

        try:
            #Gera o código python a partir do intent
            codigo_str = self.llm_reward_agent(intent)
            #requirements_code = self.get_requiriments(intent)

            if codigo_str is None:
                raise ValueError("Erro: 'codigo_str' é None. Verifique a saída de 'llm_reward_agent'.")
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
            #print(f"Função de recompensa salva como '{output_file}'.")
            print(f"DEBUG: Função de recompensa gerada:\n{codigo_python}")
            return codigo_python, k
        except Exception as e:
            print(f"Erro ao criar a função de recompensa: {e}")

    
    def run_reward_function(self, slice_obs, slice_req, buffer):
        """
        Executa a função de recompensa carregando o módulo gerado e chamando a função.
        Retorna o valor de reward (ou None em caso de erro).
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
                print("O módulo gerado não contém 'reward_function'.")
                return None

            # Garantir que a função receba tipos compatíveis (numpy arrays) para indexação/mascara
            obs_arr = np.array(slice_obs)
            slice_req_arr = np.array(slice_req)
            
            print(f"DEBUG: obs_arr={obs_arr}")
            print(f"DEBUG: slice_req_arr={slice_req_arr}")

            resultado = module.reward_function(obs_arr, slice_req_arr, buffer)
            #print("DEBUG: Resultado da função de recompensa:", resultado)
            return resultado

        except FileNotFoundError:
            print("Erro: arquivo de função não encontrado.")
        except Exception as e:
            print(f"Ocorreu um erro ao executar a função de recompensa: {e}")
            traceback.print_exc()
            return None
        
    def existing_code(self):
        """
        Verifica se o código já existe.
        """
        #print(f"DEBUG: Verificando existência do código em '{self.code_path}'")
        print(f"DEBUG: Caminho do código: {self.code_path}")
        if self.code_path.is_file() and self.code_path.stat().st_size > 0:
        #    print("DEBUG: O código existe e não está vazio.")
            return True
        else:
        #    print("DEBUG: O código não existe ou está vazio.")
            return False
        
    def get_classification_prompt(self, descricao_logica: str):
        """
        Classifica a descrição lógica em um dos 4 casos de uso.
        """
        prompt = f"""
        Você é um especialista em Redes de Acesso Rádio (RAN) e network slicing.
        Sua tarefa é definir qual caso de uso utilizar de acordo com a lógica de descrição fornecida.
        Analise a solicitação do operador abaixo e classifique-a em um dos 4 casos de uso.

        **DEFINIÇÕES DOS CASOS:**

        Caso 1 - O primeiro caso de uso de alocação de recursos tem por objetivo cumprir todos os requisitos, podendo ultrapassar os requisitos de cada slice de rede. Nesse caso, a função de reward deve focar em punir o agente não cumprir o requisito mas não leva em consideração se os slices ultrapassam o requisito.

        Caso 2 - O segundo caso de uso de alocação de recursos tem por objetivo maximizar o desempenho de um dos slices e apenas cumprir o requisito dos outros, sem ultrapassar o valor do requisito. Para isso, a função de reward deve punir o agente se não for observado o exato valor do requisito, e deve recompensar o agente se o slice priorizado tiver o maior valor observado possível. Deve-se obter o índice do slice a ser priorizado a partir da descrição lógica.

        Caso 3 - O terceiro caso de uso de alocação de recursos tem por objetivo economizar recursos e assim cumprir apenas o requisito de todos os slices. Nesse caso, a função de reward deve punir o agente por qualquer desvio entre o observado eos requisitos, a fim de que o excedente de recursos seja economizado.

        Caso 4 - O quarto caso de uso de alocação de recursos tem por objetivo otimizar o desempenho de todos os slices a partir de sua métrica de desempenho. Para isso, a função de reward deve recompensar o agente proporcialmente ao desempenho em relação ao requisito para cada slice. Além disso, a função deve atribuir um peso (+1) para slices de vazão e (-1) para slices de latência. 

        **SOLICITAÇÃO DO OPERADOR:**
        "{descricao_logica}"

        **SAÍDA:**
        Responda APENAS com o número do caso (1, 2, 3 ou 4). Não escreva nada além do número.
        """

        try:
            response = self.client.chat.completions.create(
                model=self.MODEL,
                messages=[
                    {"role": "system", "content": "Responda apenas com o número da classe."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
            )
            raw = response.choices[0].message.content.strip() if response.choices[0].message.content else ""
            clean = ''.join(filter(str.isdigit, raw))
            case_num = (int(clean) if clean else -1)
            print(f"DEBUG: Caso classificado: {case_num}")
            return case_num
            #if case_num == 3:
            #    return True
            #else:
            #    return False
        except Exception:
            print("Erro ao classificar a descrição lógica.")
            
    def get_requiriments(self, descricao_logica: str):
        """
        Extrai os requisitos dos slices a partir da descrição lógica.
        """
        prompt = f"""
        Você é um especialista em Redes de Acesso Rádio (RAN) e network slicing.
        Sua tarefa é extrair os requisitos de throughput e latência dos slices a partir da descrição lógica fornecida.

        **SOLICITAÇÃO DO OPERADOR:**
        "{descricao_logica}"
        
        **REGRAS:**
        1. Quando a métrica for vazão (throughput), extraia os valores em Mbps.
        2. Quando a métrica for latência (latency), extraia os valores em s.
        3. Quando não houver menção explicita da latencia, use o valor padrão de np.nan, para indicar que essa medida não foram especificada.
        4. Quando não houver menção explicita da vazão, use o valor padrão de np.nan, para indicar que essa medida não foram especificada.
        
        **SAÍDA:**
        Responda dois arrays separados, um para throughput e outro para latência, no formato:
        [100, 200], [10, 30]
        """

        try:
            response = self.client.chat.completions.create(
                model=self.MODEL,
                messages=[
                    {"role": "system", "content": "Responda apenas com os dois arrays separados."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
            )
            content = response.choices[0].message.content.strip() if response.choices[0].message.content else ""
            print(f"DEBUG: Requisitos extraídos: {content}")
            
            # Parse the response to extract throughput and latency arrays
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
                print("DEBUG: Formato inesperado na resposta.")
                return None#, None
        except Exception as e:
            print(f"DEBUG: Erro ao extrair os requisitos da descrição lógica: {e}")
            return None#, None
        
        ##### TESTANDO NOVA ESTRUTURA ######
    def agent_pipeline(self, descricao_logica: str):
        """
        Pipeline de agentes para gerar, validar e formatar a função de recompensa."""
        
        llm = LLM(model=self.MODEL, temperature=0.2, api_key=api_key)
        # 1. CONFIGURAÇÃO DOS AGENTES
        especialista_rl = Agent(
            role='Especialista em Reinforcement Learning e Telecomunicações',
            goal='Traduzir solicitações de operadores de rede (RAN) em lógicas precisas de cálculo de recompensa (Reward Functions) em Python.',
            backstory='Você é um engenheiro de telecomunicações especialista em 5G/6G Network Slicing e Reinforcement Learning. Você entende profundamente os 4 casos de uso de alocação de recursos (1: Cumprir requisito, 2: Maximizar um slice, 3: Economizar recursos, 4: Otimizar todos com pesos).',
            verbose=True,
            llm=llm
        )

        revisor_qa = Agent(
            role='Engenheiro de Validação de Código RL',
            goal='Garantir que a função gerada atende estritamente às regras de variáveis, buscar erros conceituais de reward shaping e possui a assinatura correta.',
            backstory='Você é um QA meticuloso. Sua função é receber o código do especialista em RL e verificar se: 1) A função se chama "reward_function". 2) Ela recebe exatamente os parâmetros (slice_obs, slice_req, buffer). 3) A variável "reward" é inicializada em 0.0 e retornada no final. Se houver erro, você corrige o código.',
            verbose=True,
            llm=llm
        )

        formatador_saida = Agent(
            role='Formatador de Scripts Python',
            goal='Extrair apenas o código Python funcional, removendo qualquer conversa do modelo.',
            backstory='Você é o guardião do arquivo final. Você não avalia lógica. Sua única função é remover saudações, explicações e marcadores Markdown (como ```python e ```). O resultado deve ser apenas o código puro.',
            verbose=True,
            llm=llm
        )

        # 2. CONFIGURAÇÃO DAS TAREFAS
        # A tarefa 1 recebe as regras lógicas e usa o placeholder {descricao_logica}
        tarefa_gerar_reward = Task(
            description='''A partir da solicitação do operador: "{descricao_logica}", identifique qual dos 4 casos de uso de Network Slicing se aplica e crie a lógica.

        Casos de Uso:
            1 - O primeiro caso de uso de alocação de recursos tem por objetivo cumprir todos os requisitos, podendo ultrapassar os requisitos de cada slice de rede. Nesse caso, a função de reward deve focar em punir o agente não cumprir o requisito mas não leva em consideração se os slices ultrapassam o requisito.
            2 - O segundo caso de uso de alocação de recursos tem por objetivo maximizar o desempenho de um dos slices e apenas cumprir o requisito dos outros, sem ultrapassar o valor do requisito. Para isso, a função de reward deve punir o agente se não for observado o valor do requisito, e assim recompensar o agente se o slice priorizado tiver o maior valor observado possível. 
            3 - O terceiro caso de uso de alocação de recursos tem por objetivo economizar recursos e assim cumprir apenas o requisito de todos os slices. Nesse caso, a função de reward deve punir o agente por qualquer desvio entre o observado e os requisitos, a fim de que o excedente de recursos seja economizado.
            4 - O quarto caso de uso de alocação de recursos tem por objetivo otimizar o desempenho de todos os slices a partir de sua métrica de desempenho. Para isso, a função de reward deve recompensar o agente proporcialmente ao desempenho em relação ao requisito para cada slice. Além disso, a função deve atribuir um peso (+1) para slices de vazão e (-1) para slices de latência. 

        Variáveis Disponíveis para a equação:**
            * `reward` (float): A variável de recompensa (inicializada em 0.0).
            * `slice_obs` (np.ndarray): O array de observações atuais de throughput.
            * `slice_req` (np.ndarray): O array de requisitos mínimos.
        Gere a função Python.''',
            expected_output='O código Python inicial implementando a função de reward.',
            agent=especialista_rl
        )

        tarefa_validar_codigo = Task(
            description='''Analise a função gerada pela tarefa anterior.
        Regras obrigatórias:
        - O nome da função DEVE ser: def reward_function(slice_obs, slice_req, buffer):
        - O código deve inicializar reward = 0.0 e retornar reward no final.
        - Verifique se a lógica matemática reflete um dos 4 casos de uso.
        - Verifique se a função utiliza de gradiente descendente como uma forma de facilitar a convergência do agente, ou seja, se o código pune o agente proporcionalmente ao erro cometido,ou seja, de forma não linear.
        - Verifique se a função é sensível a desvios, ou seja, o agente deve punir mais severamente os desvios abaixo dos requisitos do que os desvios acima dos requisitos, a fim de incentivar o agente a cumprir os requisitos.
        - Não utilize normalização dos dados na função, use valores brutos.
        - Considere valores de tolerância (eps) de 2Mbps, *apenas acima dos requisitos*, para facilitar a convergência do agente, ou seja, se o slice estiver dentro de 2Mbps do requisito, considere que o slice está cumprindo o requisito.
        - Buscar erros comuns de reward shaping, como: não punir o agente por não cumprir requisitos.
        Corrija qualquer anomalia e retorne o código completo corrigido.''',
            expected_output='O código Python validado e corrigido.',
            agent=revisor_qa
        )

        tarefa_limpar_formato = Task(
            description='''Pegue o código validado. Remova formatações Markdown e textos extras. Deixe apenas o código Python executável e comentários.
            Apenas a função reward_function e suas dependências/comentários devem permanecer.''',
            expected_output='Apenas as linhas de código Python.',
            agent=formatador_saida,
            output_file=self.code_name # Salva automaticamente
        )

        # 3. CRIAÇÃO DA EQUIPE E EXECUÇÃO
        equipe_rl = Crew(
            agents=[especialista_rl, revisor_qa, formatador_saida],
            tasks=[tarefa_gerar_reward, tarefa_validar_codigo, tarefa_limpar_formato],
            process=Process.sequential
        )

        print("Iniciando a geração da função de reward...")

        # O dicionário 'inputs' mapeia a variável para dentro das Tasks
        resultado = equipe_rl.kickoff(inputs={"descricao_logica": descricao_logica}
        )
        
if __name__ == "__main__":
    
    agent = LLMAgent()
    
    descricao_teste_caso = """
    Considere uma situação onde tem-se três slices de rede, cada um com seus próprios requisitos de desempenho. Os requisitos para cada slice são: 50, 20 e 10 Mbps, respectivamente.
    Deseja-se cumprir os requisitos do primeiro e segundo slice, sem ultrapassá-los ou ficar abaixo, para maximizar o desempenho no terceiro slice, no qual é desejado que o desempeno seja o maior possível.
    """
    #generate_code = agent.llm_reward_agent(descricao_teste_caso)
    #print(f"{generate_code}")
    #create_code = agent.create_reward_function(descricao_teste_caso, [50, 20, 10], 2)
    slice_requirements = agent.get_requiriments(descricao_teste_caso)
    print(f"\n[Resultado] Código de requisitos extraído: {slice_requirements}")
    resultado = agent.agent_pipeline(descricao_teste_caso)
    #get_case = agent.get_classification_prompt(descricao_teste_caso)
    #print(f"\n[Resultado] Código de requisitos extraído: {slice_requirements}")
    #k = agent.k_type(descricao_teste_caso, slice_requirements)
    #print(f"\n[Resultado] k extraído: {k}")
    #codigo_recompensa, k_indice = agent.create_reward_function(descricao_teste_caso, slice_requirements, k)
    #case_num = agent.get_classification_prompt(descricao_teste_caso)
    #print(f"\n[Resultado] Código gerado: {codigo_recompensa}")
    #print(f"\n[Resultado] Índice k gerado: {type(k_indice)}")
    #agent.run_reward_function([80, 150], [100, 200], k_indice)
    

    #descricao_teste_caso_3 = "economizar recursos e assim cumprir apenas o requisito de todos os slices"
    #
    #print("--- Testando llm_reward_agent (Caso 3) ---")
    #codigo_recompensa_3 = agent.create_reward_function(descricao_teste_caso_3)
    #if codigo_recompensa_3:
    #    print(f"\n[Resultado] Código gerado: {codigo_recompensa_3}")