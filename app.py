import gradio as gr
import os
with open(os.path.join('./keys/', 'openai.key'), 'r') as f:
    os.environ["OPENAI_API_KEY"] = f.read().strip()
with open(os.path.join('./keys/', 'serpapi.key'), 'r') as f:
    os.environ["SERPAPI_API_KEY"] = f.read().strip()

from algos.PWS import *
from utils.util import *

def process(tools, model, input_text):
    method = PWS_Base(planner_model=model, solver_model=model,
                  fewshot=fewshots.TRIVIAQA_PWS, available_tools=tools)
    response = method.run(input_text)
    plan = response["planner_log"].split(input_text)[1].strip('\n')
    solve = response["solver_log"].split(input_text)[1].split("Now begin to solve the task")[0].strip('\n')
    return plan, solve, response["output"]


tools = gr.components.CheckboxGroup(['Wikipedia', 'Google', 'LLM', 'WolframAlpha', 'Calculator'],label="Tools")
model = gr.components.Dropdown(["text-davinci-003", "gpt-3.5-turbo"], label="Model")
input_text = gr.components.Textbox(lines=2, placeholder="Input Here...", label="Input")
planner = gr.components.Textbox(lines=4, label="Planner")
solver = gr.components.Textbox(lines=4, label="Solver")
output = gr.components.Textbox(label="Output")

iface = gr.Interface(
    fn=process,
    inputs=[tools, model, input_text],
    outputs=[planner, solver, output],
    examples=[
        [["Wikipedia", "LLM"], "gpt-3.5-turbo", "American Callan Pinckney’s eponymously named system became a best-selling (1980s-2000s) book/video franchise in what genre?"],
        [['Google', 'LLM'], "gpt-3.5-turbo", "What is the recent paper ReWOO: Decoupling Reasoning from Observations for Efficient Augmented Language Models about?"],
        [["Calculator","WolframAlpha"], "gpt-3.5-turbo", "the car can accelerate from 0 to 27.8 m/s in a time of 3.85 seconds. Determine the acceleration of this car in m/s/s."],
    ],
    title="ReWOO Demo 🤗",
    description="""
    Demonstraing our recent work -- ReWOO: Decoupling Reasoning from Observations for Efficient Augmented Language Models.
    Note that this demo is only a conceptual impression of our work, we use a zero-shot set up and not optimizing the run time.
    """
)

iface.launch()