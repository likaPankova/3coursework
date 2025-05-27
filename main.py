# bot
import telebot
from models import predict_pois, predict_xgb, predict_lgb, clubs

BOT_TOKEN = "7905350226:AAHRfxW62hDGl9_K1aZMNbMXvPPpIEZM_9k"
bot = telebot.TeleBot(BOT_TOKEN)

user_state = {}  # Храним выбранные команды и модель


@bot.message_handler(commands=["start", "predict"])
def start(message):
    text = "Выбери домашнюю команду :"
    if message.text == "/start":
        text = "Привет, любитель азарта! 🤠" + text
    markup = telebot.types.ReplyKeyboardMarkup(one_time_keyboard=True)
    for club in clubs:
        markup.add(club)
    bot.send_message(message.chat.id, text, reply_markup=markup)
    user_state[message.chat.id] = {}


@bot.message_handler(func=lambda m: m.chat.id in user_state and "home" not in user_state[m.chat.id])
def select_home(message):
    if message.text not in clubs:
        return bot.send_message(message.chat.id, "Такой команды нет. Попробуй снова.")
    user_state[message.chat.id]["home"] = message.text
    markup = telebot.types.ReplyKeyboardMarkup(one_time_keyboard=True)
    for club in clubs:
        if club != message.text:
            markup.add(club)
    bot.send_message(message.chat.id, "Теперь выбери гостевую команду :", reply_markup=markup)


@bot.message_handler(func=lambda m: m.chat.id in user_state and "home" in user_state[m.chat.id] and "away" not in user_state[m.chat.id])
def select_away(message):
    if message.text not in clubs:
        return bot.send_message(message.chat.id, "Такой команды нет. Попробуй снова.")
    user_state[message.chat.id]["away"] = message.text
    markup = telebot.types.ReplyKeyboardMarkup(one_time_keyboard=True)
    markup.add("Poisson", "XGBoost", "LightGBM")
    bot.send_message(message.chat.id, "Выбери модель предсказания :", reply_markup=markup)


@bot.message_handler(func=lambda m: m.chat.id in user_state and "away" in user_state[m.chat.id])
def select_model(message):
    tag_map = {"Poisson": "pois", "XGBoost": "xgb", "LightGBM": "lgb"}
    if message.text not in tag_map:
        return bot.send_message(message.chat.id, "Выбери одну из моделей: Poisson, XGBoost или LightGBM.")
    state = user_state[message.chat.id]
    state["model"] = tag_map[message.text]

    home, away, model = state["home"], state["away"], state["model"]
    pred_func = {"pois": predict_pois, "xgb": predict_xgb, "lgb": predict_lgb}[model]
    pred = pred_func(home, away)

    winner = home if pred["HW"] > max(pred["D"], pred["AW"]) else (
        away if pred["AW"] > max(pred["HW"], pred["D"]) else "Ничья")

    response = (
        f"🏟 Матч: {home} vs {away}\n"
        f"\n⚽ Предсказание:\n"
        f"  {home}: {pred['HG']} гол(ов)\n"
        f"  {away}: {pred['AG']} гол(ов)\n"
        f"\n📊 Вероятности:\n"
        f"  Победа {home}: {pred['HW']:.2%}\n"
        f"  Ничья: {pred['D']:.2%}\n"
        f"  Победа {away}: {pred['AW']:.2%}\n"
        f"\n✅ Победитель: {winner}🎉\n"
        f"📈 Модель: {message.text}"
    )

    bot.send_message(message.chat.id, response)
    # user_state.pop(message.chat.id, None)  # Очистим состояние

bot.polling()