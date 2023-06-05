import argparse
import os
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Eval Arguments.')
parser.add_argument('--method',
                    type=str,
                    choices=['direct', 'cot', 'react', 'rewoo'],
                    help='Paradigm to use')

parser.add_argument('--exemplar',
                    type=str,
                    help='Input exemplar')

parser.add_argument('--toolset',
                    nargs='+',
                    default=['Google', 'Wikipedia', 'WolframAlpha', 'Calculator', 'LLM'],
                    help='Tools available to ALMs.')

parser.add_argument('--base_lm',
                    type=str,
                    default='text-davinci-003',
                    help='Base language model to use. Can be text-davinci-003, gpt-3.5-turbo or directory to alpca-lora')

parser.add_argument('--planner_lm',
                    type=str,
                    help='Base LM for Planner. Default to base_lm')

parser.add_argument('--solver_lm',
                    type=str,
                    help='Base LM for Solver. Default to base_lm')

parser.add_argument('--print_trajectory',
                    action='store_true',
                    help='Print reasoning traces to stdout (Only for ALMs)')

parser.add_argument('--key_path',
                    type=str,
                    default='./keys/',
                    help='Path where you store your openai.key and serpapi.key. Default to ./key/')

args = parser.parse_args()

with open(os.path.join(args.key_path, 'openai.key'), 'r') as f:
    os.environ["OPENAI_API_KEY"] = f.read().strip()
with open(os.path.join(args.key_path, 'serpapi.key'), 'r') as f:
    os.environ["SERPAPI_API_KEY"] = f.read().strip()

from algos.PWS import *
from algos.notool import IO, CoT
from algos.react import ReactExtraTool
from utils.util import *


def main(args):
    task = input("Ask a question or give a task: ")
    if args.method == 'direct':
        method = IO(model_name=args.base_lm)
        response = method.run(task)
    elif args.method == 'cot':
        method = CoT(model_name=args.base_lm, fewshot=DEFAULT_EXEMPLARS_COT[args.dataset])
        response = method.run(task)
    elif args.method == 'react':
        if args.exemplar is None:
            args.exemplar = fewshots.DEFAULT_REACT
        method = ReactExtraTool(model_name=args.base_lm, available_tools=args.toolset,
                                fewshot=args.exemplar, verbose=args.print_trajectory)
        response = method.run(task)
    elif args.method == 'rewoo':
        if args.planner_lm is None:
            args.planner_lm = args.base_lm
        if args.solver_lm is None:
            args.solver_lm = args.base_lm
        if args.exemplar is None:
            args.exemplar = fewshots.TRIVIAQA_PWS
        method = PWS_Base(planner_model=args.planner_lm, solver_model=args.solver_lm,
                          fewshot=args.exemplar, available_tools=args.toolset)
        response = method.run(task)
        if args.print_trajectory:
            print("=== Planner ===" + '\n\n' + response["planner_log"] + '\n' + "=== Solver ===" + '\n\n' + response[
                "solver_log"])
    else:
        raise NotImplementedError

    print(response["output"])


if __name__ == '__main__':
    main(args)
