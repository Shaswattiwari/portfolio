import streamlit as st
import telebot


bot_token = "7082167682:AAFS_8X78RoSWdkvXjz6E2WEE3vFIu1_2oM"
bot = telebot.TeleBot(bot_token)
def chat():
    st.subheader('Your queries are welcome here!')
    st.markdown("Ask a question, share a thought, or leave a review - Say Hi!")
    with st.container():
        user_name = st.text_input("", placeholder="Your Name")
        user_sub = st.text_input("", placeholder="Subject")
        user_mail = st.text_input("", placeholder="Your Email")
        user_text = st.text_area("", placeholder="Your Message")

        # Send user input to Telegram when a button is clicked
        if st.button("SEND"):
            send_to_telegram(user_name, user_sub, user_mail, user_text)

def send_to_telegram(user_name, user_sub, user_mail, user_text):
    message = f'''name-{user_name}
    subject-{user_sub}
    mail id-{user_mail}
    text-{user_text}'''
    try:
        # Send the message to your Telegram bot
        bot.send_message("5711562852", message)
        st.success("Message sent successfully!")
    except Exception as e:
        st.error(f"Error sending message: {str(e)}")
