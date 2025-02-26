from io import BytesIO
from time import sleep

import helium
import requests
from PIL import Image
from huggingface_hub import login
from selenium import webdriver
from selenium.webdriver import Keys
from selenium.webdriver.common.by import By
from smolagents import (
    OpenAIServerModel,
    CodeAgent,
    tool,
    ActionStep,
    DuckDuckGoSearchTool,
)

from config import settings


def run_images():
    image_urls = [
        "https://upload.wikimedia.org/wikipedia/commons/e/e8/The_Joker_at_Wax_Museum_Plus.jpg",  # Joker image
        "https://upload.wikimedia.org/wikipedia/en/9/98/Joker_%28DC_Comics_character%29.jpg",  # Joker image
    ]

    images = []
    for url in image_urls:
        response = requests.get(url)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        images.append(image)

    model = OpenAIServerModel(model_id="gpt-4o")
    # Instantiate the agent
    agent = CodeAgent(tools=[], model=model, max_steps=20, verbosity_level=2)

    response = agent.run(
        """
        Describe the costume and makeup that the comic character in these photos is wearing and return the description.
        Tell me if the guest is The Joker or Wonder Woman.
        """,
        images=images,
    )


@tool
def search_item_ctrl_f(text: str, nth_result: int = 1) -> str:
    """
    Searches for text on the current page via Ctrl + F and jumps to the nth occurrence.
    Args:
        text: The text to search for
        nth_result: Which occurrence to jump to (default: 1)
    """
    elements = driver.find_elements(By.XPATH, f"//*[contains(text(), '{text}')]")
    if nth_result > len(elements):
        raise Exception(
            f"Match n°{nth_result} not found (only {len(elements)} matches found)"
        )
    result = f"Found {len(elements)} matches for '{text}'."
    elem = elements[nth_result - 1]
    driver.execute_script("arguments[0].scrollIntoView(true);", elem)
    result += f"Focused on element {nth_result} of {len(elements)}"
    return result


@tool
def go_back() -> None:
    """Goes back to previous page."""
    driver.back()


@tool
def close_popups() -> str:
    """
    Closes any visible modal or pop-up on the page. Use this to dismiss pop-up windows! This does not work on cookie consent banners.
    """
    webdriver.ActionChains(driver).send_keys(Keys.ESCAPE).perform()


def save_screenshot(step_log: ActionStep, agent: CodeAgent) -> None:
    sleep(1.0)  # Let JavaScript animations happen before taking the screenshot
    driver = helium.get_driver()
    current_step = step_log.step_number
    if driver is not None:
        for (
            step_logs
        ) in agent.logs:  # Remove previous screenshots from logs for lean processing
            if (
                isinstance(step_log, ActionStep)
                and step_log.step_number <= current_step - 2
            ):
                step_logs.observations_images = None
        png_bytes = driver.get_screenshot_as_png()
        image = Image.open(BytesIO(png_bytes))
        print(f"Captured a browser screenshot: {image.size} pixels")
        step_log.observations_images = [
            image.copy()
        ]  # Create a copy to ensure it persists, important!

    # Update observations with current URL
    url_info = f"Current url: {driver.current_url}"
    step_log.observations = (
        url_info
        if step_logs.observations is None
        else step_log.observations + "\n" + url_info
    )
    return


def initialize_driver():
    """Initialize the Selenium WebDriver."""
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--force-device-scale-factor=1")
    chrome_options.add_argument("--window-size=1000,1350")
    chrome_options.add_argument("--disable-pdf-viewer")
    chrome_options.add_argument("--window-position=0,0")
    return helium.start_chrome(headless=False, options=chrome_options)


helium_instructions = """
Use your web_search tool when you want to get Google search results.
Then you can use helium to access websites. Don't use helium for Google search, only for navigating websites!
Don't bother about the helium driver, it's already managed.
We've already ran "from helium import *"
Then you can go to pages!
Code:
```py
go_to('github.com/trending')
```<end_code>
You can directly click clickable elements by inputting the text that appears on them.
Code:
```py
click("Top products")
```<end_code>
If it's a link:
Code:
```py
click(Link("Top products"))
```<end_code>
If you try to interact with an element and it's not found, you'll get a LookupError.
In general stop your action after each button click to see what happens on your screenshot.
Never try to login in a page.
To scroll up or down, use scroll_down or scroll_up with as an argument the number of pixels to scroll from.
Code:
```py
scroll_down(num_pixels=1200) # This will scroll one viewport down
```<end_code>
When you have pop-ups with a cross icon to close, don't try to click the close icon by finding its element or targeting an 'X' element (this most often fails).
Just use your built-in tool `close_popups` to close them:
Code:
```py
close_popups()
```<end_code>
You can use .exists() to check for the existence of an element. For example:
Code:
```py
if Text('Accept cookies?').exists():
    click('I accept')
```<end_code>
Proceed in several steps rather than trying to solve the task in one shot.
And at the end, only when you have your answer, return your final answer.
Code:
```py
final_answer("YOUR_ANSWER_HERE")
```<end_code>
If pages seem stuck on loading, you might have to wait, for instance `import time` and run `time.sleep(5.0)`. But don't overuse this!
To list elements on page, DO NOT try code-based element searches like 'contributors = find_all(S("ol > li"))': just look at the latest screenshot you have and read it visually, or use your tool search_item_ctrl_f.
Of course, you can act on buttons like a user would do when navigating.
After each code blob you write, you will be automatically provided with an updated screenshot of the browser and the current browser url.
But beware that the screenshot will only be taken at the end of the whole action, it won't see intermediate states.
Don't kill the browser.
When you have modals or cookie banners on screen, you should get rid of them before you can click anything else.
"""


def run_browser():
    model = OpenAIServerModel(model_id="gpt-4o")

    agent = CodeAgent(
        tools=[DuckDuckGoSearchTool(), go_back, close_popups, search_item_ctrl_f],
        model=model,
        additional_authorized_imports=["helium"],
        step_callbacks=[save_screenshot],
        max_steps=20,
        verbosity_level=2,
    )
    agent.python_executor("from helium import *", agent.state)
    agent.run(
        """
    I am Alfred, the butler of Wayne Manor, responsible for verifying the identity of guests at party. A superhero has arrived at the entrance claiming to be Wonder Woman, but I need to confirm if she is who she says she is.

    Please search for images of Wonder Woman and generate a detailed visual description based on those images. Additionally, navigate to Wikipedia to gather key details about her appearance. With this information, I can determine whether to grant her access to the event.
    """
        + helium_instructions
    )


if __name__ == "__main__":
    login(token=settings.HF_TOKEN)

    # run_images()

    global driver
    driver = initialize_driver()

    run_browser()
