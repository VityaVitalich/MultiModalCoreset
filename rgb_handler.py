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

# Handler for image messages with the specified command
@router.message(F.photo, Command('test'))
async def handle_depth_from_rgb(message: types.Message):

    global bot 
    # # Download the image
    # await bot.download(
    #     message.photo[-1],
    #     destination=f"./tmp/{message.photo[-1].file_id}.jpg"
    # )
    file_id = message.photo[-1].file_id
    file = await bot.get_file(file_id)
    download_file= await bot.download_file(file.file_path)
    img = Image.open(download_file)
    tmp_path = f'./tmp/{file_id}.jpg'
    img.save(tmp_path)



    # # Convert the image to a numpy array
    # image = Image.open(io.BytesIO(image_data))
    # image = np.array(image)

    # Perform inference on the model
    # depth_image = predict(model, image)  # Replace with your actual prediction function
    depth_image = FSInputFile(tmp_path)

    # Convert the depth image array to bytes
    # depth_image_bytes = io.BytesIO()
    # Image.fromarray(depth_image).save(depth_image_bytes, format='PNG')
    # depth_image_bytes.seek(0)

    # Send the depth image as a reply
    await message.answer_photo(
        depth_image,
        caption="Изображение из файла на компьютере"
    )
    os.remove(tmp_path)

   # await bot.send_photo(message.chat.id, depth_image_bytes, caption="Depth prediction")

    # except exceptions.BadRequest:
    #     await message.reply("Error processing the image. Please try again.")
