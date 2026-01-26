import os
from stagehand import Stagehand, StagehandConfig

import nest_asyncio
import asyncio

# Allow nested loops in async (for environments like Jupyter or already-running loops)
nest_asyncio.apply()


def browser_automation(task_description: str, website_url: str) -> str:
    """Performs automated browser tasks using AI agent capabilities."""

    async def _execute_automation():
        stagehand = None

        try:
            config = StagehandConfig(
                env="LOCAL",
                model_name=os.getenv("STAGEHAND_MODEL", "gpt-4o"),
                self_heal=True,
                system_prompt="You are a browser automation assistant that helps users navigate websites effectively.",
                model_client_options={"apiKey": os.getenv("MODEL_API_KEY")},
                verbose=1,
            )

            stagehand = Stagehand(config)
            await stagehand.init()

            await stagehand.page.goto(website_url)

            # Use Stagehand's act/extract flows (avoid CUA-only agent models)
            await stagehand.page.act(
                f"Navigate and interact as needed to gather the information required. Task: {task_description}"
            )
            extraction = await stagehand.page.extract(
                f"Provide the answer to: {task_description}"
            )

            result_message = getattr(extraction, "extraction", None) or str(extraction)
            return f"Browser Automation Tool result:\n{result_message}"

        finally:
            if stagehand:
                await stagehand.close()

    # Run async in a sync context
    return asyncio.run(_execute_automation())
