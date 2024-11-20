import base64
import logging
import os
import time
import traceback
from datetime import datetime
from io import BytesIO
from typing import Any, cast, Final, List, Optional, Tuple, Type

from agentdesk.device_v1 import Desktop
from devicebay import Device
from mllm import Router
from pydantic import BaseModel
from rich.console import Console
from rich.json import JSON
from surfkit.agent import TaskAgent
from taskara import Task, TaskStatus
from tenacity import before_sleep_log, retry, stop_after_attempt
from threadmem import RoleThread
from toolfuse.util import AgentUtils

from anthropic import Anthropic
from anthropic.types.beta import (
    BetaMessageParam,
    BetaTextBlockParam,
    BetaToolResultBlockParam,
)

from .anthropic import ToolResult, response_to_params, make_api_tool_result

logging.basicConfig(level=logging.INFO)
logger: Final = logging.getLogger(__name__)
logger.setLevel(int(os.getenv("LOG_LEVEL", str(logging.DEBUG))))

console = Console(force_terminal=True)

if not os.environ.get("ANTHROPIC_API_KEY"):
    raise ValueError ("Please set the ANTHROPIC_API_KEY in your environment.")
client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

router = Router.from_env()

class SurfPizzaConfig(BaseModel):
    pass


class SurfPizza(TaskAgent):
    """A GUI desktop agent that slices up the image"""

    def solve_task(
        self,
        task: Task,
        device: Optional[Device] = None,
        max_steps: int = 30,
    ) -> Task:
        """Solve a task

        Args:
            task (Task): Task to solve.
            device (Device): Device to perform the task on.
            max_steps (int, optional): Max steps to try and solve. Defaults to 30.

        Returns:
            Task: The task
        """

        # Post a message to the default thread to let the user know the task is in progress
        task.post_message("assistant", f"Starting task '{task.description}'")

        # Create threads in the task to update the user
        console.print("creating threads...")
        task.ensure_thread("debug")
        task.post_message("assistant", "I'll post debug messages here", thread="debug")

        # Check that the device we received is one we support
        if not isinstance(device, Desktop):
            raise ValueError("Only desktop devices supported")

        # Add standard agent utils to the device
        device.merge(AgentUtils())

        # Get info about the desktop
        info = device.info()
        screen_size = info["screen_size"]
        console.print(f"Desktop info: {screen_size}")

        # Define Anthropic Computer Use tool. Refer to the docs at https://docs.anthropic.com/en/docs/build-with-claude/computer-use#computer-tool
        self.tools = [
            {
                "type": "computer_20241022",
                "name": "computer",
                "display_width_px": 1024,
                "display_height_px": 768,
                "display_number": 1,
            },
        ]

        console.print("tools: ", style="purple")
        console.print(JSON.from_data(self.tools))

        # Create our thread and start with the task description and system prompt
        messages = []
        messages.append(
            {
                "role": "user",
                "content": [BetaTextBlockParam(type="text", text=task.description)],
            })

        # The following prompt is a modified copy of the prompt from Anthropic's Computer Use Demo project
        # Some other code lines in this file are also copied from Anthropic's Computer Use Demo project
        # Original file: https://github.com/anthropics/anthropic-quickstarts/tree/main/computer-use-demo/computer_use_demo/tools

        SYSTEM_PROMPT = f"""<SYSTEM_CAPABILITY>
        * You are utilising an Linux virtual machine of screen size {screen_size} with internet access.
        * To open firefox, please just click on the web browser (globe) icon.
        * When viewing a page it can be helpful to zoom out so that you can see everything on the page.  Either that, or make sure you scroll down to see everything before deciding something isn't available.
        * When using your computer function calls, they take a while to run and send back to you.
        * The current date is {datetime.today().strftime('%A, %B %-d, %Y')}.
        </SYSTEM_CAPABILITY>

        <IMPORTANT>
        * When using Firefox, if a startup wizard appears, IGNORE IT.  Do not even click "skip this step".  Instead, click on the address bar where it says "Search or enter address", and enter the appropriate search term or URL there.
        * If the item you are looking at is a pdf, if after taking a single screenshot of the pdf it seems that you want to read the entire document instead of trying to continue to read the pdf from your screenshots + navigation, determine the URL, use curl to download the pdf, install and use pdftotext to convert it to a text file, and then read that text file directly with your StrReplaceEditTool.
        </IMPORTANT>"""

        self.system = BetaTextBlockParam(
            type="text",
            text=f"{SYSTEM_PROMPT}",
        )

        self.action_mapping = {
            "key": "hot_key",
            "type": "type_text",
            "mouse_move": "move_mouse",
            "left_click": "click",
            "left_click_drag": "drag_mouse",
            "right_click": "N/A",
            "middle_click": "N/A",
            "double_click": "double_click",
            "screenshot": "take_screenshots",
            "cursor_position": "mouse_coordinates",
        }

        for i in range(max_steps):
            console.print(f"-------step {i + 1}", style="green")

            try:
                messages, done = self.take_action(device, task, messages)
            except Exception as e:
                console.print(f"Error: {e}", style="red")
                task.status = TaskStatus.FAILED
                task.error = str(e)
                task.save()
                task.post_message("assistant", f"❗ Error taking action: {e}")
                return task

            if done:
                console.print("task is done", style="green")
                return task

            time.sleep(2)

        task.status = TaskStatus.FAILED
        task.save()
        task.post_message("assistant", "❗ Max steps reached without solving task")
        console.print("Reached max steps without solving task", style="red")

        return task

    @retry(
        stop=stop_after_attempt(5),
        before_sleep=before_sleep_log(logger, logging.INFO),
    )
    def take_action(
        self,
        device: Desktop,
        task: Task,
        messages: list[BetaMessageParam],
    ) -> Tuple[RoleThread, bool]:
        """Take an action

        Args:
            device (Desktop): Desktop to use
            task (str): Task to accomplish
            messages: Messages (LLM exchange thread) for the task

        Returns:
            bool: Whether the task is complete
        """
        try:
            # Check to see if the task has been cancelled
            if task.remote:
                task.refresh()
            console.print("task status: ", task.status.value)
            if (
                task.status == TaskStatus.CANCELING
                or task.status == TaskStatus.CANCELED
            ):
                console.print(f"task is {task.status}", style="red")
                if task.status == TaskStatus.CANCELING:
                    task.status = TaskStatus.CANCELED
                    task.save()
                return messages, True

            console.print("taking action...", style="white")

            raw_response = client.beta.messages.with_raw_response.create(
                max_tokens=4096,
                messages=messages,
                model="claude-3-5-sonnet-20241022",
                system=[self.system],
                tools=self.tools,
                betas=["computer-use-2024-10-22"],
            )

            try:
                response = raw_response.parse()
                response_params = response_to_params(response)

                messages.append(
                    {
                        "role": "assistant",
                        "content": response_params,
                    }
                )

            except Exception as e:
                console.print(f"Response failed to parse: {e}", style="red")
                raise

            # The agent will return 'end_turn' if it believes it's finished
            if response.stop_reason == "end_turn":
                console.print("final result: ", style="green")
                console.print(JSON.from_data(response_params[0]))
                task.post_message(
                    "assistant",
                    f"✅ I think the task is done, please review the result: {response_params[0]['text']}",
                )
                task.status = TaskStatus.FINISHED
                task.save()
                return messages, True

            tool_result_content: list[BetaToolResultBlockParam] = []

            for content_block in response_params:
                if content_block["type"] == "text":
                    task.post_message("assistant", f"👁️ {content_block.get('text')}")
                elif content_block["type"] == "tool_use":
                    input_args = cast(dict[str, Any], content_block["input"])
                    action_name = self.action_mapping[input_args["action"]]
                    console.print(f"found action: {action_name}", style="blue")
                    
                    task.post_message(
                        "assistant",
                        f"▶️ Taking action '{action_name}' with parameters: {input_args}",
                    )

                    action_params = input_args.copy()
                    del action_params["action"]

                    # Find the selected action in the tool
                    action = device.find_action(action_name)
                    console.print(f"found action: {action}", style="blue")
                    if not action:
                        console.print(f"action returned not found: {action_name}")
                        raise SystemError("action not found")

                    # Take the selected action
                    try:
                        if action_name != "take_screenshots": # Do not execute if the action is screenshot, coz we take the screenshot later anyway
                            action_params = self._get_mapped_action_params(action_name, action_params)
                            action_response = device.use(action, **action_params)

                            console.print(f"action output: {action_response}", style="blue")

                            if action_response:
                                task.post_message(
                                    "assistant", f"👁️ Result from taking action: {action_response}"
                                )
                    except Exception as e:
                        raise ValueError(f"Trouble using action: {e}")

                    # Take the screenshot after executing the action, or if the action itself is screenshot
                    screenshot_img = device.take_screenshots()[-1]
                    console.print(f"screenshot img type: {type(screenshot_img)}")

                    buffer = BytesIO()
                    screenshot_img.save(buffer, format='PNG')
                    base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

                    result = ToolResult(output=None, error=None, base64_image=base64_image)

                    task.post_message(
                        "assistant",
                        "current image",
                        # images=result.base64_image,
                        thread="debug",
                    )

                    tool_result_content.append(
                        make_api_tool_result(result, content_block["id"])
                    )
            
            if not tool_result_content:
                return messages, True

            messages.append({"content": tool_result_content, "role": "user"})

            return messages, False

        except Exception as e:
            console.print("Exception taking action: ", e)
            traceback.print_exc()
            task.post_message("assistant", f"⚠️ Error taking action: {e} -- retrying...")
            raise e

    def _get_mapped_action_params(self, action_name, action_params) -> ToolResult:
        if action_name != "screenshot":
            if action_name in ["move_mouse", "drag_mouse"] and "coordinate" in action_params:
                action_params["x"] = action_params["coordinate"][0]
                action_params["y"] = action_params["coordinate"][1]
                del action_params["coordinate"]
            if action_name == "hot_key" and "text" in action_params:
                action_params["keys"] = [action_params["text"]]
                del action_params["text"]
            return action_params

    @classmethod
    def supported_devices(cls) -> List[Type[Device]]:
        """Devices this agent supports

        Returns:
            List[Type[Device]]: A list of supported devices
        """
        return [Desktop]

    @classmethod
    def config_type(cls) -> Type[SurfPizzaConfig]:
        """Type of config

        Returns:
            Type[SurfPizzaConfig]: Config type
        """
        return SurfPizzaConfig

    @classmethod
    def from_config(cls, config: SurfPizzaConfig) -> "SurfPizza":
        """Create an agent from a config

        Args:
            config (SurfPizzaConfig): Agent config

        Returns:
            SurfPizza: The agent
        """
        return SurfPizza()

    @classmethod
    def default(cls) -> "SurfPizza":
        """Create a default agent

        Returns:
            SurfPizza: The agent
        """
        return SurfPizza()

    @classmethod
    def init(cls) -> None:
        """Initialize the agent class"""
        return


Agent = SurfPizza
