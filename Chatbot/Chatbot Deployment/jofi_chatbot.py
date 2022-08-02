# import library
import numpy as np
import string
import logging
import pyfiglet
import logging.config
import os
import re
import joblib
from util import JSONParser
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters                  

def preprocess(chat):
    # konversi ke lowercase
    chat = chat.lower()
    # menghapus tanda baca
    tandabaca = tuple(string.punctuation)
    chat = ''.join(ch for ch in chat if ch not in tandabaca)
    return chat

def bot_response(chat, model_pipe, parser):
    chat = preprocess(chat)
    res = model_pipe.predict_proba([chat])
    max_prob = max(res[0])
    if max_prob < 0.1:
        return "Maaf kak, aku ga ngerti :(" , None
    else:
        max_id = np.argmax(res[0])
        pred_tag = model_pipe.classes_[max_id]
        return parser.get_response(pred_tag), pred_tag

def start(update, context):
    update.message.reply_text("Selamat!, kakak telah terhubung JoFiBot")

def respons(update, context):
    chat = update.message.text

    # load data
    path = "data/intents.json"
    parser = JSONParser()
    parser.parse(path)
    data = parser.get_dataframe()

    # Load Chatbot Machine Learning Model
    model = joblib.load("chatbot_model.pkl")

    res, tag = bot_response(chat, model, parser)
    update.message.reply_text(res)

def error(update, context):
    # Log Errors caused by Updates.
    logging.warning('Update "%s" ', update)
    logging.exception(context.error)

def main():
    updater = Updater(DefaultConfig.TELEGRAM_TOKEN, use_context=True)
    dp = updater.dispatcher

    # Command handlers
    dp.add_handler(CommandHandler("start",start))

    # Message handler
    dp.add_handler(MessageHandler(Filters.text, respons))

    # log all errors
    dp.add_error_handler(error)

    # Start the Bot
    if DefaultConfig.MODE == 'webhook':

        updater.start_webhook(listen="0.0.0.0",
                              port=int(DefaultConfig.PORT),
                              url_path=DefaultConfig.TELEGRAM_TOKEN)
        updater.bot.setWebhook(DefaultConfig.WEBHOOK_URL + DefaultConfig.TELEGRAM_TOKEN)
        
        logging.info(f"Start webhook mode on port {DefaultConfig.PORT}")
    else:
        updater.start_polling()
        logging.info(f"Start polling mode")

    updater.idle()

class DefaultConfig:
    PORT = int(os.environ.get("PORT", 3978))
    TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "YourBotTokenAPI")
    MODE = os.environ.get("MODE", "polling")
    WEBHOOK_URL = os.environ.get("WEBHOOK_URL", "YourHerokuLink")

    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO').upper()

    @staticmethod
    def init_logging():
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                            level=DefaultConfig.LOG_LEVEL)
        #logging.config.fileConfig('logging.conf')

if __name__ == '__main__':
    ascii_banner = pyfiglet.figlet_format("AdvancedTelegramBot")
    print(ascii_banner)

    # Enable logging
    DefaultConfig.init_logging()

    main()
