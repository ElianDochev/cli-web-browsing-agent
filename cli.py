from typing import Dict, Any
import argparse
import os
import contextlib
import io
import sys
from pathlib import Path
from urllib.parse import urlparse
from dotenv import load_dotenv
from pydantic import BaseModel
from crewai import Agent, Task, Crew
from crewai import LLM
from crewai.tools import tool
from crewai.flow.flow import Flow, start, listen
from browser_agent import browser_automation


load_dotenv()

def get_llm_model(env_var: str, default_model: str = "openai/gpt-5-mini") -> str:
    model_name = os.getenv(env_var)
    if model_name is None:
        return default_model
    model_name = model_name.strip()
    return model_name or default_model


def normalize_and_validate_url(raw_url: str) -> str | None:
    if not raw_url:
        return None
    candidate = raw_url.strip()
    if not candidate:
        return None
    if "://" not in candidate:
        candidate = f"https://{candidate}"
    parsed = urlparse(candidate)
    if parsed.scheme != "https" or not parsed.netloc:
        return None
    return candidate


# Define our LLMs for providing to agents
planner_llm = LLM(model=get_llm_model("PLANNER_LLM_MODEL"))
automation_llm = LLM(model=get_llm_model("AUTOMATION_LLM_MODEL"))
response_llm = LLM(model=get_llm_model("RESPONSE_LLM_MODEL"))


@tool("Stagehand Browser Tool")
def stagehand_browser_tool(task_description: str, website_url: str) -> str:
    """
    A tool that allows to interact with a web browser.
    The tool is used to perform browser automation tasks powered by Stagehand capabilities.

    Args:
        task_description (str): The task description for the agent to perform.
        website_url (str): The URL of the website to interact and navigate to.

    Returns:
        str: The result of the browser automation task.
    """
    return browser_automation(task_description, website_url)


class BrowserAutomationFlowState(BaseModel):
    query: str = ""
    result: str = ""
    verbose: bool = False


class AutomationPlan(BaseModel):
    task_description: str
    website_url: str


class BrowserAutomationFlow(Flow[BrowserAutomationFlowState]):
    """
    A CrewAI Flow to intelligently handle browser automation tasks
    through specialized agents using Stagehand tools.
    """

    @start()
    def start_flow(self) -> Dict[str, Any]:
        if self.state.verbose:
            print(f"Flow started with query: {self.state.query}")
        return {"query": self.state.query}

    @listen(start_flow)
    def plan_task(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if self.state.verbose:
            print("--- Using Automation Planner to plan the task ---")

        planner_agent = Agent(
            role="Automation Planner Specialist",
            goal="Plan the automation task for the user's query.",
            backstory="You are a browser automation specialist that plans the automation task for the user's query.",
            llm=planner_llm,
        )

        plan_task = Task(
            description=f"Analyze the following user query and determine the website url and the task description: '{inputs['query']}'.",
            agent=planner_agent,
            output_pydantic=AutomationPlan,
            expected_output=(
                "A JSON object with the following format:\n"
                "{\n"
                '  "task_description": "<brief description of what needs to be done>",\n'
                '  "website_url": "<URL of the target website>"\n'
                "}"
            ),
        )

        crew = Crew(agents=[planner_agent], tasks=[plan_task], verbose=self.state.verbose)
        result = crew.kickoff()

        # Add a fallback check to ensure we always have a valid website URL
        website_url = normalize_and_validate_url(result.pydantic.website_url)
        if website_url is None:
            website_url = "https://www.google.com"

        return {
            "task_description": result["task_description"],
            "website_url": website_url,
        }

    @listen(plan_task)
    def handle_browser_automation(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if self.state.verbose:
            print("--- Delegating to Browser Automation Specialist ---")

        automation_agent = Agent(
            role="Browser Automation Specialist",
            goal="Execute browser automation using the Stagehand tool",
            backstory="You specialize in executing user-defined automation tasks on websites using the Stagehand tool.",
            tools=[stagehand_browser_tool],
            llm=automation_llm,
        )

        automation_task = Task(
            description=(
                f"Perform the following browser automation task:\n\n"
                f"Website: {inputs['website_url']}\n"
                f"Task: {inputs['task_description']}\n\n"
                f"Use the Stagehand tool to complete this task accurately."
            ),
            agent=automation_agent,
            expected_output="A string containing the result of executing the browser automation task using Stagehand.",
            markdown=True,
        )

        crew = Crew(agents=[automation_agent], tasks=[automation_task], verbose=self.state.verbose)
        result = crew.kickoff()
        return {"result": str(result)}

    @listen(handle_browser_automation)
    def synthesize_result(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if self.state.verbose:
            print("--- Synthesizing Final Response ---")

        synthesis_agent = Agent(
            role="Response Synthesis Specialist",
            goal="Craft a clear, concise, and user-friendly response based on the tool calling output from the browser automation specialist.",
            backstory="An expert in communication and assistance.",
            llm=response_llm,
        )

        synthesis_task = Task(
            description=(
                "You are a summarizer. Produce a concise, relevant answer to the user's original query. "
                "Only include information that directly answers the question. If the result contains noise, ignore it. "
                "Keep the response under 5 sentences or 6 bullet points. Do not include raw page dumps or boilerplate.\n\n"
                f"Original query: {self.state.query}\n\n"
                f"Browser automation result:\n{inputs['result']}"
            ),
            expected_output=(
                "A short, relevant summary that directly answers the original query, with no irrelevant details."
            ),
            agent=synthesis_agent,
        )

        crew = Crew(agents=[synthesis_agent], tasks=[synthesis_task], verbose=self.state.verbose)
        final_result = crew.kickoff()
        return {"result": str(final_result)}


async def run_flow(query: str, output_path: str | None = None, verbose: bool = False) -> None:
    flow = BrowserAutomationFlow()
    flow.state.query = query
    flow.state.verbose = verbose
    os.environ["STAGEHAND_VERBOSE"] = "1" if verbose else "0"

    if not verbose:
        print(f"Prompt: {query}")
        print("Executing...")
    if verbose:
        result = await flow.kickoff_async()
    else:
        buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
                result = await flow.kickoff_async()
        except Exception:
            captured = buffer.getvalue().strip()
            if captured:
                sys.stderr.write(captured + "\n")
            raise

    print(f"\n{'='*50}")
    print("FINAL RESULT")
    print(f"{'='*50}")
    print(result["result"])

    if output_path:
        output_file = Path(output_path)
        output_file.write_text(
            f"PROMPT:\n{query}\n\nRESULT:\n{result['result']}\n",
            encoding="utf-8",
        )


def parse_args() -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    parser = argparse.ArgumentParser(description="Run browser automation flow.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("-p", "--prompt", help="Prompt text to run.")
    input_group.add_argument(
        "-f",
        "--file",
        help="Path to a file containing the prompt.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Optional output file to save the prompt and result.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show detailed execution steps.",
    )
    return parser, parser.parse_args()


def load_prompt(prompt: str | None, file_path: str | None, parser: argparse.ArgumentParser) -> str:
    if prompt is not None:
        prompt_text = prompt.strip()
        if not prompt_text:
            parser.error("Prompt cannot be empty.")
        return prompt_text
    if file_path:
        prompt_text = Path(file_path).read_text(encoding="utf-8").strip()
        if not prompt_text:
            parser.error("Prompt file is empty.")
        return prompt_text
    parser.error("Either -p/--prompt or -f/--file is required.")
    return ""


if __name__ == "__main__":
    import asyncio

    parser, args = parse_args()
    prompt_text = load_prompt(args.prompt, args.file, parser)
    asyncio.run(run_flow(prompt_text, args.output, args.verbose))
