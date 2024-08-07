from aiogram import types
from aiogram import Router
from aiogram.filters import Command
from aiogram.types import FSInputFile
from aiogram.enums import ParseMode
from messages import (
    help_message,
    model_info_message,
    author_info_message,
)

router = Router()

multimae_img_path = "./figs/multimae_fig.png"
multimae_img = FSInputFile(multimae_img_path)


@router.message(Command("help"))
async def cmd_help(message: types.Message):
    await message.answer(help_message, parse_mode=ParseMode.HTML)


@router.message(Command("model_info"))
async def cmd_model_info(message: types.Message):
    await message.answer(model_info_message, parse_mode=ParseMode.HTML)

    await message.answer_photo(multimae_img, caption="MultiMAE Architecture")


@router.message(Command("author_info"))
async def cmd_author_info(message: types.Message):
    await message.answer(author_info_message, parse_mode=ParseMode.HTML)
