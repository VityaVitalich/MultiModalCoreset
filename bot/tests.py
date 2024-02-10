import pytest
import sys
import os
from aiogram.filters import Command
import bot
import rate_handler
from system_handlers import (
    cmd_help,
    cmd_model_info,
    multimae_img_path,
    cmd_author_info
)
from messages import (
    greeting_message,
    help_message,
    model_info_message,
    author_info_message,
)

sys.path.append("./aiogram_tests")
sys.path.append("./aiogram_tests/aiogram_tests")
sys.path.append("./aiogram_tests/tests")

from aiogram_tests import MockedBot  # noqa:E402
from aiogram_tests.handler import MessageHandler  # noqa:E402
from aiogram_tests.types.dataset import MESSAGE, MESSAGE_WITH_PHOTO  # noqa:E402, E501


@pytest.mark.asyncio
async def test_command_handler():
    requester = MockedBot(MessageHandler(bot.cmd_start,
                                         Command(commands=["start"])))
    calls = await requester.query(MESSAGE.as_object(text="/start"))
    answer_message = calls.send_message.fetchone().text
    assert answer_message == greeting_message


@pytest.mark.asyncio
async def test_command_help():
    requester = MockedBot(MessageHandler(cmd_help, Command(commands=["help"])))
    calls = await requester.query(MESSAGE.as_object(text="/help"))
    answer_message = calls.send_message.fetchone().text
    assert answer_message == help_message


@pytest.mark.asyncio
async def test_command_cmd_author_info():
    requester = MockedBot(
        MessageHandler(cmd_author_info, Command(commands=["author_info"]))
    )
    calls = await requester.query(MESSAGE.as_object(text="/author_info"))
    answer_message = calls.send_message.fetchone().text
    assert answer_message == author_info_message


@pytest.mark.asyncio
async def test_command_model_info():
    requester = MockedBot(
        MessageHandler(cmd_model_info, Command(commands=["model_info"]))
    )
    calls = await requester.query(MESSAGE.as_object(text="/model_info"))
    answer_message = calls.send_message.fetchone()
    assert answer_message.text == model_info_message


@pytest.mark.asyncio
async def test_command_model_photo_exists():
    assert os.path.exists(multimae_img_path)


@pytest.mark.asyncio
async def test_command_model_photo():
    requester = MockedBot(
        MessageHandler(cmd_model_info, Command(commands=["model_info"]))
    )
    calls = await requester.query(MESSAGE.as_object(text="/model_info"))
    answer_message = calls.send_photo.fetchone()
    assert answer_message.photo.path == multimae_img_path


@pytest.mark.asyncio
async def test_bot_stats_1_call():
    bot.rgb_calls = 1

    requester = MockedBot(
        MessageHandler(bot.bot_stats, Command(commands=["bot_stats"]))
    )

    calls = await requester.query(MESSAGE.as_object(text="/bot_stats"))

    answer_message = calls.send_message.fetchone()

    stats_message = """
        <b>📊 Overall Bot Statistics</b>

        <b>RGB Model Usage:</b> {}

        <b>Mean Rating:</b> Rating is not set Yet

        """.format(
        bot.rgb_calls
    )
    assert answer_message.text == stats_message


@pytest.mark.asyncio
async def test_bot_stats_non_zero_rating():
    global overall_rating
    global overall_times_rated

    bot.rgb_calls = 1
    bot.rate_handler.overall_rating = 5
    bot.rate_handler.overall_times_rated = 1
    desired_rating = 5.0

    requester = MockedBot(
        MessageHandler(bot.bot_stats, Command(commands=["bot_stats"]))
    )

    calls = await requester.query(MESSAGE.as_object(text="/bot_stats"))

    answer_message = calls.send_message.fetchone()

    stats_message = """
    <b>📊 Overall Bot Statistics</b>

    <b>RGB Model Usage:</b> {} times

    <b>Mean Rating:</b> {}

    """.format(
        bot.rgb_calls, desired_rating
    )
    assert answer_message.text == stats_message


@pytest.mark.asyncio
async def test_rgb_error():
    requester = MockedBot(
        MessageHandler(bot.handle_depth_from_rgb, Command(commands=["rgb"]))
    )

    calls = await requester.query(MESSAGE_WITH_PHOTO.as_object(text="/rgb"))

    answer_message = calls.send_message.fetchone()
    message = "Error processing the image. Please try again."
    assert answer_message.text == message


@pytest.mark.asyncio
async def test_command_cmd_rate():
    requester = MockedBot(
        MessageHandler(rate_handler.cmd_rate, Command(commands=["rate"]))
    )
    calls = await requester.query(MESSAGE.as_object(text="/rate"))
    answer_message = calls.send_message.fetchone().text
    assert answer_message == "Оцените предсказание глубины"


@pytest.mark.asyncio
async def test_command_cmd_rate_excelent():
    requester = MockedBot(MessageHandler(rate_handler.rate_excelent))
    calls = await requester.query(MESSAGE.as_object(text="Отлично!"))
    answer_message = calls.send_message.fetchone().text
    assert answer_message == "Спасибо! Мы рады, что Вам понравилось"


@pytest.mark.asyncio
async def test_command_cmd_rate_excelent_increment():
    rate_handler.overall_rating = 0
    requester = MockedBot(MessageHandler(rate_handler.rate_excelent))
    await requester.query(MESSAGE.as_object(text="Отлично!"))

    assert rate_handler.overall_rating == 5


@pytest.mark.asyncio
async def test_command_cmd_rate_good():
    requester = MockedBot(MessageHandler(rate_handler.rate_good))
    calls = await requester.query(MESSAGE.as_object(text="Лучше среднего"))
    answer_message = calls.send_message.fetchone().text
    assert answer_message == "Спасибо! Мы работаем, чтобы стало идеально"


@pytest.mark.asyncio
async def test_command_cmd_rate_good_increment():
    rate_handler.overall_rating = 0
    requester = MockedBot(MessageHandler(rate_handler.rate_good))
    await requester.query(MESSAGE.as_object(text="Лучше среднего"))

    assert rate_handler.overall_rating == 4


@pytest.mark.asyncio
async def test_command_cmd_rate_okey():
    requester = MockedBot(MessageHandler(rate_handler.rate_okey))
    calls = await requester.query(MESSAGE.as_object(text="Сойдет"))
    answer_message = calls.send_message.fetchone().text
    assert answer_message == "Спасибо! Мы постараемся улучшить наш сервис!"


@pytest.mark.asyncio
async def test_command_cmd_rate_okey_increment():
    rate_handler.overall_rating = 0
    requester = MockedBot(MessageHandler(rate_handler.rate_okey))
    await requester.query(MESSAGE.as_object(text="Сойдет"))

    assert rate_handler.overall_rating == 3


@pytest.mark.asyncio
async def test_command_cmd_rate_bad():
    requester = MockedBot(MessageHandler(rate_handler.rate_bad))
    calls = await requester.query(MESSAGE.as_object(text="Хуже среднего"))
    answer_message = calls.send_message.fetchone().text
    message = "Спасибо! Нам жаль, что сервис справился плохо, в скором времени станет лучше"  # noqa: E501
    assert (
        answer_message
        == message
    )


@pytest.mark.asyncio
async def test_command_cmd_rate_bad_increment():
    rate_handler.overall_rating = 0
    requester = MockedBot(MessageHandler(rate_handler.rate_bad))
    await requester.query(MESSAGE.as_object(text="Хуже среднего"))

    assert rate_handler.overall_rating == 2


@pytest.mark.asyncio
async def test_command_cmd_rate_worst():
    requester = MockedBot(MessageHandler(rate_handler.rate_worst))
    calls = await requester.query(MESSAGE.as_object(text="Очень плохо!"))
    answer_message = calls.send_message.fetchone().text
    message = "Спасибо! Нам очень жаль, что сервис не смог справиться, с последующими обновлениями качество вырастет"  # noqa: E501
    assert (
        answer_message
        == message
    )


@pytest.mark.asyncio
async def test_command_cmd_rate_worst_increment():
    rate_handler.overall_rating = 0
    requester = MockedBot(MessageHandler(rate_handler.rate_worst))
    await requester.query(MESSAGE.as_object(text="Очень плохо!"))

    assert rate_handler.overall_rating == 1
