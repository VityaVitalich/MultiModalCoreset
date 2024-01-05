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

router = Router()
overall_rating = 0
overall_times_rated = 0


@router.message(Command("rate"))
async def cmd_rate(message: types.Message):
    kb = [
        [types.KeyboardButton(text="Отлично!")],
        [types.KeyboardButton(text="Лучше среднего")],
        [types.KeyboardButton(text="Сойдет")],
        [types.KeyboardButton(text="Хуже среднего")],
        [types.KeyboardButton(text="Очень плохо!")],
    ]
    keyboard = types.ReplyKeyboardMarkup(
        keyboard=kb,
        resize_keyboard=True,
        input_field_placeholder="Выберите подходящую оценку",
    )
    await message.answer("Оцените предсказание глубины", reply_markup=keyboard)


@router.message(F.text == "Отлично!")
async def rate_excelent(message: types.Message):
    global overall_times_rated
    overall_times_rated += 1
    global overall_rating
    overall_rating += 5
    await message.reply(
        "Спасибо! Мы рады, что Вам понравилось",
        reply_markup=types.ReplyKeyboardRemove(),
    )


@router.message(F.text == "Лучше среднего")
async def rate_good(message: types.Message):
    global overall_times_rated
    overall_times_rated += 1
    global overall_rating
    overall_rating += 4
    await message.reply(
        "Спасибо! Мы работаем, чтобы стало идеально",
        reply_markup=types.ReplyKeyboardRemove(),
    )


@router.message(F.text == "Сойдет")
async def rate_okey(message: types.Message):
    global overall_times_rated
    overall_times_rated += 1
    global overall_rating

    overall_rating += 3
    await message.reply(
        "Спасибо! Мы постараемся улучшить наш сервис!",
        reply_markup=types.ReplyKeyboardRemove(),
    )


@router.message(F.text == "Хуже среднего")
async def rate_bad(message: types.Message):
    global overall_times_rated
    overall_times_rated += 1
    global overall_rating

    overall_rating += 2
    await message.reply(
        "Спасибо! Нам жаль, что сервис справился плохо, в скором времени станет лучше",
        reply_markup=types.ReplyKeyboardRemove(),
    )


@router.message(F.text == "Очень плохо!")
async def rate_worst(message: types.Message):
    global overall_times_rated
    overall_times_rated += 1
    global overall_rating

    overall_rating += 1
    await message.reply(
        "Спасибо! Нам очень жаль, что сервис не смог справиться, с последующими обновлениями качество вырастет",
        reply_markup=types.ReplyKeyboardRemove(),
    )
