from pathlib import Path
import streamlit as st
from factorial import factory
import os
BASE_DIR = Path(__file__).parent
USER_FILE = BASE_DIR / "user.txt"

def load_user():
    try:
        if USER_FILE.exists():
            with open(USER_FILE, "r", encoding="utf-8") as f:
                users = [line.strip() for line in f.readlines() if line.strip()]
                st.write(f"wtf {users}")
            return users

        else:
            st.error("File user.txt không tồn tại!")
            return []

    except Exception as e:
        st.error(f"Error reading the file {e}")
        return []

def login_page():
    st.title("LOGIN")

    username = st.text_input("Enter your name: ")
    if st.button("Login"):
        if username:
            users = load_user()
            if username in users:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.rerun()
            else:
                st.session_state.show_greeting = True
                st.session_state.username = username
                # st.rerun()

        else:
            st.warning("Please enter your name")

def factorial_calculator():
    st.title("Factorial Calculator")
    st.write(f"Hello {st.session_state.username}")

    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ''
        st.rerun()

    st.divider()

    number = st.number_input("Enter number: ", min_value=0, max_value=90)
    if st.button("Calulate"):
        result = factory(number)
        st.write(f"Factory of {number} is {result}")

def greeting_page():
    st.title(f"Hello! {st.session_state.username}")
    st.write("You cannot access the factorial calculator")

    if st.button("Logout"):
        st.session_state.show_greeting = False
        st.session_state.username = ''
        st.rerun()

def main():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'username' not in st.session_state:
        st.session_state.username = ''
    if 'show_greeting' not in st.session_state:
        st.session_state.show_greeting = False

    if st.session_state.logged_in:
        factorial_calculator()
    elif st.session_state.show_greeting:
        greeting_page()
    else:
        login_page()

if __name__ == "__main__":
    main()