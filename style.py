import streamlit as st
from PIL import Image
import base64


def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()
def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

def Style(title,header):
    # im = Image.open(r'images/images.ico')
    # st.set_page_config(page_title=title, page_icon=im ,layout='wide')
    st.set_page_config(page_title=title, layout='wide')
    # st.sidebar.image(im, caption='Pitney Bowes', width=150)
    set_background(r'images/image.png')
    css = '''
    <style>
        [data-testid="stSidebar"]{
            min-width: 200px;
            max-width: 22spx;
        }
    </style>
    '''
    st.markdown(css, unsafe_allow_html=True)
    st.markdown(
        f'<p class="small-font" style="font-size: 33px; color:#FFC300;font-weight:630;">{header}</p>',
        unsafe_allow_html=True)