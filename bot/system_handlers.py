import asyncio
import logging
from aiogram import Bot, Dispatcher, types
import io
import numpy as np
import os
from PIL import Image
from aiogram import F, Router
from aiogram.types import Message
from aiogram.filters import Command, CommandObject, CommandStart
from aiogram.types import FSInputFile, URLInputFile, BufferedInputFile
from aiogram.enums import ParseMode

router = Router()

multimae_img = FSInputFile("./figs/multimae_fig.png")


@router.message(Command("start"))
async def cmd_start(message: types.Message):
    greeting_message = """
    <b>👋 Hello there! Welcome to the Depth Prediction Bot! 🤖</b>

    I'm here to assist you in predicting depth from images. If you need help or information about available commands, you can use the <code>/help</code> command.

    Feel free to explore the capabilities of the bot and enjoy your experience! 🚀
    """
    await message.answer(greeting_message, parse_mode=ParseMode.HTML)


@router.message(Command("help"))
async def cmd_help(message: types.Message):
    help_message = """
    <b> Welcome to the Depth Prediction Bot!🤖 </b>

    <b><u>Commands:</u></b>
    <code>/rgb</code> - Predict depth from an RGB image. Send an RGB picture with this command.\n
    <code>/multi</code> - Predict depth from RGB and semantic images. Send two pictures in this order: RGB and semantic mask.\n
    <code>/rate</code> - Leave a review for the bot.\n
    <code>/bot_stats</code> - Check the statistics of the bot.\n
    <code>/model_info</code> - Check model information.\n
    <code>/author_info</code> - Check author information.


    <b><u>Examples:</u></b>
    - To predict depth from an RGB image:\n
      <code>/rgb</code> (send an RGB picture)

    - To leave a review:\n
      <code>/rate</code>

    - To check bot statistics:\n
      <code>/bot_stats</code>

    - To get model info:\n
      <code>/model_info</code>

    - To get author info:\n
      <code>/author_info</code>

    Feel free to explore the capabilities of the bot and enjoy your experience! 🚀
    """
    await message.answer(help_message, parse_mode=ParseMode.HTML)


@router.message(Command("model_info"))
async def cmd_model_info(message: types.Message):
    model_info_message = """
<b>🌟 Model Information 🌟</b>

The backbone of this depth prediction model is the <i>MultiMAE</i> developed by the EPFL Lab. You can find details about it in the paper titled "<a href="https://arxiv.org/abs/2204.01678">MultiMAE: Multi-modal Multi-task Masked Autoencoders</a>."

The model's architecture is based on the Vision Transformer (ViT) with the following specifications:
- Patch Size: 16
- Image Size: 224
- Hidden Dimension: 768
- Number of Heads: 12
- Number of Layers: 12
- Layer Normalization is applied

<b>Adapters:</b>
- RGB Adapter: Utilizes a 2D convolution, simulating patching, followed by a linear layer.
- Semantic Segmentation Adapter: Maps each class to a learned embedding and employs a 2D convolution.

<b>Depth Output Adapter:</b>
The depth output adapter is a <i>DPT Adapter</i>, inspired by the "Vision Transformers for Dense Prediction" <a href="https://arxiv.org/pdf/2103.13413.pdf">paper</a> from Intel Labs.
The model was fine-tuned on the CLEVR dataset for 5 epochs with a learning rate of 3e-4, achieving near-perfect score.

You can check the architecture on the following graph.
"""
    await message.answer(model_info_message, parse_mode=ParseMode.HTML)

    await message.answer_photo(multimae_img, caption="MultiMAE Architecture")


@router.message(Command("author_info"))
async def cmd_author_info(message: types.Message):
    author_info_message = """
<b>👨‍💻 Author Information</b>

Hello! My name is <i>Viktor Moskvoretskii</i>, and I am a researcher at Skoltech and a master's student at HSE.

This bot was developed as part of my master's thesis in collaboration with the EPFL Lab. Our primary research focus is on Multimodal Deep Learning and its applications.

Currently, only Depth Estimation is available, but stay tuned for more exciting features in the future!

<b>About Me:</b>
Apart from Multimodal Deep Learning, my research interests include:

- Large Language Models (LLM)
- Reinforcement Learning (RL), especially for LLMs
- Theory of Generative Modeling

If you have valuable ideas for joint research or collaboration in these areas, I would be thrilled to hear from you:

- Telegram: <a href="https://t.me/VityaVitalichDS">@VityaVitalichDS</a>
- Email: <a href="mailto:vvmoskvoretskiy@yandex.ru">vvmoskvoretskiy@yandex.ru</a>

Let's explore innovative possibilities in AI together!
"""

    await message.answer(author_info_message, parse_mode=ParseMode.HTML)
