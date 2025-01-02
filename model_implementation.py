from io import BytesIO
import telebot
from PIL import Image
import numpy as np
import tensorflow as tf
from telebot.states import State, StatesGroup
from telebot.storage import StateMemoryStorage
from telebot.states.sync.middleware import StateMiddleware
from telebot.states.sync.context import StateContext
from telebot import custom_filters


TOKEN = '7729564646:AAHy-_9E2djmtOdldwTiCxOdd6_9PDeEKX0'

state_storage = StateMemoryStorage()

model = tf.keras.models.load_model('beta_binary_classifier.h5')

bot = telebot.TeleBot(TOKEN, state_storage=state_storage, use_class_middlewares=True)


class UploadStates(StatesGroup):
    waiting_for_file = State()
    processing_file = State()


class PhotoStates(StatesGroup):
    waiting_for_photo = State()
    resizing_photo = State()


@bot.message_handler(commands=['start'])
def start_handler(msg):
    bot.send_message(msg.chat.id, 'Привет! Я бот для анализа рентгеновских снимков.'
                                  'Вы можете загрузить свое изображение, и я помогу вам с его анализом.'
                                  ' Напишите /help, чтобы узнать больше о командах.')


@bot.message_handler(commands=['upload'])
def upload(msg, state: StateContext):
    state.set(UploadStates.waiting_for_file)
    bot.send_message(msg.chat.id, 'Пожалуйста, загрузите рентгеновский снимок в формате JPG или PNG.')


@bot.message_handler(state=UploadStates.waiting_for_file, content_types=['photo', 'text', 'document'])
def photo_handler(msg, state: StateContext):
    if msg.text:
        bot.send_message(msg.chat.id, "Пожалуйста, отправьте изображение, а не текст.")
        return

    elif msg.document:
        file_extension = msg.document.file_name.split(".")[-1].lower()
        if file_extension not in ['jpg', 'jpeg', 'png']:
            bot.send_message(msg.chat.id, "Пожалуйста, отправьте изображение в формате JPG или PNG.")
            return
        file_id = msg.document.file_id

    elif msg.photo:
        file_id = msg.photo[-1].file_id
    
    bot.send_message(msg.chat.id, "Спасибо! Я получил ваш рентгеновский снимок."
                                    "Начинаю анализ... Пожалуйста, подождите.")
    bot.send_chat_action(msg.chat.id, 'typing')
    state.set(UploadStates.processing_file)

    
    file_info = bot.get_file(file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    image = Image.open(BytesIO(downloaded_file))
    try:
        image = Image.open(BytesIO(downloaded_file))
        modified_image = image.convert('L').resize((512, 512), resample=Image.Resampling.LANCZOS)
    except Exception as e:
        bot.send_message(msg.chat.id, "Произошла ошибка при обработке изображения. Попробуйте загрузить его снова.")
        state.delete()
        return

    modified_image = image.convert('L').resize((512, 512), resample=Image.Resampling.LANCZOS)

    image_array = np.array(modified_image)
    image_array1 = np.expand_dims(image_array, axis=-1)
    image_array2 = np.expand_dims(image_array1, axis=0)

    try:
        prediction = model.predict(image_array2)
    except Exception as e:
        bot.send_message(msg.chat.id, "Произошла ошибка при анализе изображения. Попробуйте снова.")
        return
    finally:
        state.delete()
        


    if prediction[0][0] >= 0.7:
        bot.send_message(msg.chat.id,
                            'Анализ завершен!'
                            'На представленном рентгеновском снимке присутствует перелом.')
    else:
        bot.send_message(msg.chat.id,
                            'Анализ завершен! На представленном рентгеновском снимке отсутствует перелом.')

    state.delete()
    return


@bot.message_handler(commands=['status'])
def check_status(msg, state: StateContext):
    current_state = state.get()
    if current_state == 'UploadStates:processing_file':
        bot.send_message(msg.chat.id, 'Анализ вашего снимка еще не завершен. '
                                      'Ожидайте, пожалуйста.' 
                                      'Обычно это занимает от 1 до 5 минут.')
    else:
        bot.send_message(msg.chat.id, 'Вы не вызвали /upload.')


@bot.message_handler(commands=['help'])
def show_help(msg):
    help_text = (
        "*Доступные команды:*\n"
        "🔹 /upload - загрузить рентгеновский снимок для анализа\n"
        "🔹 /status - проверить статус анализа\n"
        "🔹 /help - получить помощь по использованию бота"
    )
    bot.send_message(msg.chat.id, help_text, parse_mode='Markdown')


@bot.message_handler(commands=['feedback'])
def send_feedback_form(message):
    markup = telebot.types.InlineKeyboardMarkup()
    markup.add(telebot.types.InlineKeyboardButton("⭐️", callback_data="rating_1"),
               telebot.types.InlineKeyboardButton("⭐️⭐️", callback_data="rating_2"),
               telebot.types.InlineKeyboardButton("⭐️⭐️⭐️", callback_data="rating_3"),
               telebot.types.InlineKeyboardButton("⭐️⭐️⭐️⭐️", callback_data="rating_4"),
               telebot.types.InlineKeyboardButton("⭐️⭐️⭐️⭐️⭐️", callback_data="rating_5"))
    bot.reply_to(message, "Оцените наше приложение:", reply_markup=markup)


@bot.callback_query_handler(func=lambda call: call.data.startswith("rating_"))
def process_rating(call):
    rating = call.data.split("_")[1]
    with open("feedback.txt", "a", encoding="utf-8") as f:
        f.write(f"Пользователь {call.from_user.id}: {rating} звезд\n")
    bot.edit_message_text(chat_id=call.message.chat.id, message_id=call.message.id,
                          text=f"Спасибо за вашу оценку! Вы поставили {rating} звезд.")


bot.add_custom_filter(custom_filters.StateFilter(bot))

bot.setup_middleware(StateMiddleware(bot))

bot.polling(none_stop=True, long_polling_timeout=60, timeout=20, interval=1)
