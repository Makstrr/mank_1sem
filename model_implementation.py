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

TOKEN = ''

# Хранилище состояний
state_storage = StateMemoryStorage()

# Загрузка модели
model = tf.keras.models.load_model('beta_binary_classifier.h5')

# Инициализация бота
bot = telebot.TeleBot(TOKEN, state_storage=state_storage, use_class_middlewares=True)

# Состояния для загрузки и обработки изображения
class UploadStates(StatesGroup):
    waiting_for_file = State()
    processing_file = State()

# Предобработка изображения
def preprocess_image(file_bytes):
    try:
        image = Image.open(BytesIO(file_bytes))
        image = image.convert('L').resize((512, 512), resample=Image.Resampling.LANCZOS)
        image_array = np.expand_dims(np.expand_dims(np.array(image), axis=-1), axis=0)
        return image_array
    except Exception as e:
        return None

# Команда: /start
@bot.message_handler(commands=['start'])
def start_handler(msg):
    bot.send_message(msg.chat.id, (
        'Привет! Я бот для анализа рентгеновских снимков.\n'
        'Вы можете загрузить изображение для анализа.\n'
        'Напишите /help, чтобы узнать больше о командах.'
    ))

# Команда: /upload
@bot.message_handler(commands=['upload'])
def upload(msg, state: StateContext):
    state.set(UploadStates.waiting_for_file)
    bot.send_message(msg.chat.id, 'Пожалуйста, загрузите рентгеновский снимок в формате JPG или PNG.')

# Хэндлер загруженного фото
@bot.message_handler(state=UploadStates.waiting_for_file, content_types=['photo', 'document'])
def photo_handler(msg, state: StateContext):
    file_id = None
    if msg.photo:
        file_id = msg.photo[-1].file_id
    elif msg.document:
        if msg.document.file_name.split(".")[-1].lower() not in ['jpg', 'jpeg', 'png']:
            bot.send_message(msg.chat.id, "Пожалуйста, отправьте изображение в формате JPG или PNG.")
            return
        file_id = msg.document.file_id

    if not file_id:
        bot.send_message(msg.chat.id, "Пожалуйста, отправьте изображение, а не текст.")
        return

    bot.send_message(msg.chat.id, "Спасибо! Начинаю анализ... Пожалуйста, подождите.")
    bot.send_chat_action(msg.chat.id, 'typing')
    state.set(UploadStates.processing_file)

    try:
        file_info = bot.get_file(file_id)
        file_bytes = bot.download_file(file_info.file_path)
        processed_image = preprocess_image(file_bytes)
        if processed_image is None:
            raise ValueError("Ошибка при обработке изображения")

        prediction = model.predict(processed_image)
        result = ('перелом присутствует.' if prediction[0][0] >= 0.7 else 'перелом отсутствует.')
        bot.send_message(msg.chat.id, f'Анализ завершен! На снимке {result}')
    except Exception as e:
        bot.send_message(msg.chat.id, "Произошла ошибка. Попробуйте снова.")
    finally:
        state.delete()

# Команда: /status
@bot.message_handler(commands=['status'])
def check_status(msg, state: StateContext):
    current_state = state.get()
    if current_state == UploadStates.processing_file:
        bot.send_message(msg.chat.id, 'Анализ еще не завершен. Ожидайте, пожалуйста.')
    else:
        bot.send_message(msg.chat.id, 'Вы не вызвали /upload.')

# Команда: /help
@bot.message_handler(commands=['help'])
def show_help(msg):
    help_text = (
        "*Доступные команды:*\n"
        "🔹 /upload - загрузить рентгеновский снимок для анализа\n"
        "🔹 /status - проверить статус анализа\n"
        "🔹 /help - получить помощь по использованию бота"
    )
    bot.send_message(msg.chat.id, help_text, parse_mode='Markdown')

# Команда: /feedback
@bot.message_handler(commands=['feedback'])
def send_feedback_form(msg):
    markup = telebot.types.InlineKeyboardMarkup()
    for i in range(1, 6):
        markup.add(telebot.types.InlineKeyboardButton(f"⭐️" * i, callback_data=f"rating_{i}"))
    bot.send_message(msg.chat.id, "Оцените наше приложение:", reply_markup=markup)

@bot.callback_query_handler(func=lambda call: call.data.startswith("rating_"))
def process_rating(call):
    rating = call.data.split("_")[1]
    with open("feedback.txt", "a", encoding="utf-8") as f:
        f.write(f"Пользователь {call.from_user.id}: {rating} звезд\n")
    bot.edit_message_text(chat_id=call.message.chat.id, message_id=call.message.id,
                          text=f"Спасибо за вашу оценку! Вы поставили {rating} звезд.")

# Добавление фильтра состояний и middleware (промежуточного слоя)
bot.add_custom_filter(custom_filters.StateFilter(bot))
bot.setup_middleware(StateMiddleware(bot))

# Поллинг
bot.polling(none_stop=True, long_polling_timeout=60, timeout=20, interval=1)
