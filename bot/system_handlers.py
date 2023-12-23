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

@router.message(Command('start'))
async def cmd_start(message: types.Message):
    greeting_message = """
    <b>👋 Hello there! Welcome to the Depth Prediction Bot! 🤖</b>

    I'm here to assist you in predicting depth from images. If you need help or information about available commands, you can use the <code>/help</code> command.

    Feel free to explore the capabilities of the bot and enjoy your experience! 🚀
    """
    await message.answer(
        greeting_message,
        parse_mode=ParseMode.HTML)

@router.message(Command('help'))
async def cmd_help(message: types.Message):
    help_message = """
    <b> Welcome to the Depth Prediction Bot!🤖 </b>

    <b><u>Commands:</u></b>
    <code>/rgb</code> - Predict depth from an RGB image. Send an RGB picture with this command.\n
    <code>/multi</code> - Predict depth from RGB and semantic images. Send two pictures in this order: RGB and semantic mask.\n
    <code>/rate</code> - Leave a review for the bot.\n
    <code>/bot_stats</code> - Check the statistics of the bot.\n
    <code>/info</code> - Check model parameters, model, and author info.

    <b><u>Examples:</u></b>
    - To predict depth from an RGB image:\n
      <code>/rgb</code> (send an RGB picture)

    - To predict depth from RGB and semantic images:\n
      <code>/multi</code> (send an RGB picture, then send a semantic mask picture)

    - To leave a review:\n
      <code>/rate</code>

    - To check bot statistics:\n
      <code>/bot_stats</code>

    - To get model and author info:\n
      <code>/info</code>

    Feel free to explore the capabilities of the bot and enjoy your experience! 🚀
    """
    await message.answer(
        help_message,
        parse_mode=ParseMode.HTML)