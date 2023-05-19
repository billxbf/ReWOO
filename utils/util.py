from prompts import fewshots

OPENAI_COMPLETION_MODELS = ["text-davinci-003"]
OPENAI_CHAT_MODELS = ["gpt-3.5-turbo"]
LLAMA_WEIGHTS = ["tloen/alpaca-lora-7b", "alpaca-lora/lora-alpaca-planner", "billxbf/specialized-rewoo-planner-7b"]

DEFAULT_EXEMPLARS_COT = {"hotpot_qa": fewshots.HOTPOTQA_COT,
                         "trivia_qa": fewshots.TRIVIAQA_COT,
                         "gsm8k": fewshots.GSM8K_COT,
                         "physics_question": fewshots.TRIVIAQA_COT,
                         "sports_understanding": fewshots.TRIVIAQA_COT,
                         "strategy_qa": fewshots.TRIVIAQA_COT,
                         "sotu_qa": fewshots.TRIVIAQA_COT}

DEFAULT_EXEMPLARS_REACT = {"hotpot_qa": fewshots.HOTPOTQA_REACT,
                           "trivia_qa": fewshots.TRIVIAQA_REACT,
                           "gsm8k": fewshots.GSM8K_REACT,
                           "physics_question": fewshots.GSM8K_REACT,
                           "sports_understanding": fewshots.GSM8K_REACT,
                           "strategy_qa": fewshots.GSM8K_REACT,
                           "sotu_qa": fewshots.GSM8K_REACT}

DEFAUL_EXEMPLARS_PWS = {"hotpot_qa": fewshots.HOTPOTQA_PWS_BASE,
                        "trivia_qa": fewshots.TRIVIAQA_PWS,
                        "gsm8k": fewshots.GSM8K_PWS,
                        "physics_question": fewshots.GSM8K_PWS,
                        "sports_understanding": fewshots.GSM8K_PWS,
                        "strategy_qa": fewshots.GSM8K_PWS,
                        "sotu_qa": fewshots.GSM8K_PWS}


def get_token_unit_price(model):
    if model in OPENAI_COMPLETION_MODELS:
        return 0.00002
    elif model in OPENAI_CHAT_MODELS:
        return 0.000002
    elif model in LLAMA_WEIGHTS:
        return 0.0
    else:
        raise ValueError("Model not found")
