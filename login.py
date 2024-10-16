import streamlit as st
from recommendations import authenticate
from app import main

# Initialize session state for Spotify client and login status
if 'sp' not in st.session_state:
    st.session_state.sp = None
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = None

# Set background image
page_bg_img = '''
<style>
body {
    background-image: url("https://images.unsplash.com/photo-1542281286-9e0a16bb7366");
    background-size: cover;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: white'><b>Emotion-based music recommendation</b></h2>",
            unsafe_allow_html=True)

def login():
    """Handles the Spotify login process."""
    if st.button("Log In"):
        try:
            sp = authenticate()
            if sp is not None:
                st.session_state.sp = sp
                st.session_state.logged_in = True
                st.session_state.username = sp.current_user()['display_name']
                st.success("Spotify logged in successfully!", icon="✅")
                st.rerun()
                return True
            else:
                st.warning("Failed to log in to Spotify. Please try again.")
        except Exception as e:
            st.error(f"An error occurred during login: {str(e)}")
    return False

# Main logic
if not st.session_state.logged_in:
    st.header("Please log in to continue")
    login()
else:
    st.header(f"Welcome, {st.session_state.username}!")
    main()