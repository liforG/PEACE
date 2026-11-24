import os
import cv2
import time
import random
import base64
import common
import logging
from functools import lru_cache
from io import BytesIO
from uuid import uuid4
from PIL import Image
from mimetypes import guess_type
import openai
from openai import AzureOpenAI, OpenAI

azure_endpoint = ""# to be filled.
identity_id = ""# to be filled.
api_version = "2024-08-01-preview"


@lru_cache(maxsize=1)
def _get_client():
    if common.llm_provider == "opencode":
        api_key = os.getenv("OPENCODE_API_KEY")
        if not api_key or not api_key.strip():
            raise ValueError("Set OPENCODE_API_KEY to use the OpenCode provider.")

        base_url = os.getenv(
            "OPENCODE_BASE_URL", "https://opencode.ai/inference/openai/v1"
        ).rstrip("/")
        headers = {}
        if base_url == "https://opencode.ai/inference/go/openai/v1":
            headers["x-opencode-session"] = f"ses_{uuid4().hex}"

        return OpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers=headers,
            max_retries=0,
        )

    if common.llm_provider != "azure":
        raise ValueError("PEACE_LLM_PROVIDER must be 'azure' or 'opencode'.")

    from azure.identity import DefaultAzureCredential, get_bearer_token_provider

    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(managed_identity_client_id=identity_id),
        "https://cognitiveservices.azure.com/.default",
    )
    return AzureOpenAI(
        azure_endpoint=azure_endpoint,
        azure_ad_token_provider=token_provider,
        api_version=api_version,
        max_retries=0,
    )


def answer_wrapper(messages, max_tks=2048, temperature=0.0, structured=False, tools=None):
    client = _get_client()
    request_options = {}
    if structured or common.llm_provider == "azure":
        request_options["response_format"] = {"type": "json_object" if structured else "text"}
    if tools is not None:
        request_options.update(tools=tools, tool_choice="auto")

    models = [common.model_name]
    max_trial = len(models)
    current = random.randint(0, len(models)-1)
    answer = None
    for i in range(max_trial):
        try:
            response = None
            response = client.chat.completions.create(
                model=models[current],
                messages=messages,
                temperature=temperature,
                max_tokens=max_tks,
                **request_options,
            )
            message = response.choices[0].message
            if tools is not None:
                answer = message
            else:
                answer = message.content
        # https://github.com/openai/openai-python/blob/main/openai/error.py
        except openai.RateLimitError as e:
            logging.warning(f"RateLimitError: {e}. Retrying...")
            time.sleep(2)
        except openai.BadRequestError as e:
            if e.code == "content_filter":
                logging.warning(f"BadRequestError:content_filter: {e}. Skipping...")
                break
            elif e.code == "context_length_exceeded":
                logging.warning(f"BadRequestError:context_length_exceeded: {e}. Skipping...")
                break
            else:
                logging.warning(f"BadRequestError: {e}. Skipping...")
                break
        except openai.APITimeoutError as e:
            logging.warning(f"APITimeoutError: {e}. Retrying...")
            time.sleep(2)
        except openai.APIConnectionError as e:
            logging.warning(f"Connection aborted: {e}.")
            time.sleep(2)
        except Exception as e:
            if response is not None and response.choices[0].finish_reason == "content_filter":
                print(messages)
                print(response.choices[0].content_filter_results)
                logging.warning(f"{type(e).__name__}:content_filter: {e}. Skipping...")
                break
            else:
                logging.warning(f"{type(e).__name__}: {e}. Retrying...")
                time.sleep(2)
        if answer is not None:
            break
    return answer

def input_image_to_data_url(image_rgb_array, image_format="PNG") :
    # Convert numpy array to PIL Image
    image = Image.fromarray(image_rgb_array)
    # Convert the image to a bytes object
    buffered = BytesIO()
    image.save(buffered, format=image_format)
    # Guess the MIME type from the image format
    mime_type, _ = guess_type(f"dummy.{image_format.lower()}")
    if mime_type is None:
        mime_type = "application/octet-stream"  # Default MIME type if none is found
    # Encode the image to base64
    base64_encoded_data = base64.b64encode(buffered.getvalue()).decode("utf-8")
    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"

def local_image_to_data_url(image_path):
    api_image_size_limit = 20 * 1024 * 1024# 20M
    if os.path.getsize(image_path) >= api_image_size_limit:
        img = cv2.imread(image_path)
        h, w, _ = img.shape
        resized_img = cv2.resize(img, (w//2, h//2))
        resized_img_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        return input_image_to_data_url(resized_img_rgb)

    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = "application/octet-stream"  # Default MIME type if none is found
    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode("utf-8")
    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"
