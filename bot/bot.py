import asyncio
import logging
from aiogram import Bot, Dispatcher, types
import io
import numpy as np
import os
from pathlib import Path
from PIL import Image
from aiogram import F
from aiogram.types import Message
from messages import greeting_message
from aiogram.filters import Command, CommandObject, CommandStart
from aiogram.types import FSInputFile, URLInputFile, BufferedInputFile
import model_utils
from aiogram.enums import ParseMode
import rate_handler
from rate_handler import overall_rating, overall_times_rated
import system_handlers

# Включаем логирование, чтобы не пропустить важные сообщения
logging.basicConfig(level=logging.INFO)
# Объект бота
BOT_TOKEN = os.environ.get("BOT_TOKEN")
bot = Bot(token=BOT_TOKEN)
# Диспетчер
dp = Dispatcher()
dp.include_routers(rate_handler.router, system_handlers.router)

model = model_utils.init_rgb_model()

rgb_calls = 0

tmp_path = Path("./tmp/")
tmp_path.mkdir(parents=True, exist_ok=True)


@dp.message(F.photo, Command("rgb"))
async def handle_depth_from_rgb(message: types.Message):
    global rgb_calls
    rgb_calls += 1

    try:
        
        file_id = message.photo[-1].file_id
        file = await bot.get_file(file_id)
        download_file = await bot.download_file(file.file_path)
        img = Image.open(download_file)
        tmp_path = f"./tmp/{file_id}.jpg"

        out = model_utils.inference(img, model)
        model_utils.save_predictions(out, tmp_path)
        depth_image = FSInputFile(tmp_path)

        # Send the depth image as a reply
        await message.answer_photo(depth_image, caption="Predicted depth")
        os.remove(tmp_path)

    except:
        await message.reply("Error processing the image. Please try again.")


@dp.message(Command("bot_stats"))
async def bot_stats(message: types.Message):
    global rgb_calls
    global overall_rating
    global overall_times_rated

    if rate_handler.overall_times_rated == 0:
        stats_message = """
        <b>📊 Overall Bot Statistics</b>

        <b>RGB Model Usage:</b> {}

        <b>Mean Rating:</b> Rating is not set Yet

        """.format(
                rgb_calls
            )

        await message.answer(stats_message, parse_mode=ParseMode.HTML)
    else:
        rating = rate_handler.overall_rating / rate_handler.overall_times_rated
        stats_message = """
    <b>📊 Overall Bot Statistics</b>

    <b>RGB Model Usage:</b> {} times

    <b>Mean Rating:</b> {}

    """.format(
            rgb_calls, rating
        )

        await message.answer(stats_message, parse_mode=ParseMode.HTML)

@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer(greeting_message, parse_mode=ParseMode.HTML)



async def main():
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
