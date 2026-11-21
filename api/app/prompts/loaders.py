from pathlib import Path

PROMPTS_DIR= Path(__file__).parent #__file__ tells the current path i.e api/app/prompts

def get_prompt(name:str, version: str = "v1")->None:
    return (PROMPTS_DIR/name/f"{version}.md").read_text(encoding="utf-8").strip()

#name: agent_name/prompt_name ?
#name: support/system

#version: v1
#path: api/app/prompts/support/system/v2