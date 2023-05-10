import logging
import logging.config
import datetime
import os
from tempfile import NamedTemporaryFile

import streamlit as st
import whisper
import yaml

def init_logger():
    with open(os.path.join(os.path.dirname(__file__), "cfg/logging_config.yml"), "r") as logging_config_file:
        logging_config = yaml.safe_load(logging_config_file.read())
        logging.config.dictConfig(logging_config)
        return logging.getLogger(__name__)


hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

init_logger()
logger = logging.getLogger(__name__)
supported_file_types = ["mp3", "wav", "m4a", "webm", "mpga", "mpeg"]
audio_file = st.file_uploader("选择一个音频文件:", type=supported_file_types)


@st.cache_resource(ttl=3600, show_spinner=False)
def load_model():
    model = whisper.load_model("small")
    return model

def format_time(num_str):
    return str(datetime.timedelta(seconds=int(num_str)))

def transcribe(audio_file):
    with NamedTemporaryFile() as temp:
        with st.spinner("请耐心等待 ..."):

            temp.write(audio_file.getvalue())
            temp.seek(0)

            # return { "text": temp.name }

    #         audio_file = temp.name
    #         try:
    #             result = load_model().transcribe(temp.name, temperature=0)
    #         except Exception:
    #             logger.error("The file could not be transcribed, file: " + temp.name)
    #     return result

            model = load_model()

            audio = whisper.load_audio(temp.name)
            audio = whisper.pad_or_trim(audio)

            # fp16 should be False if use CPU as it is not supported on CPU
            return model.transcribe(audio, verbose=True, fp16 = False)

            mel = whisper.log_mel_spectrogram(audio).to(model.device)

            _, probs = model.detect_language(mel)

            options = whisper.DecodingOptions(fp16 = False)
            result = whisper.decode(model, mel, options)

            return result
        

if audio_file is not None:
    logger.debug("transcript started...")
    result = transcribe(audio_file)

    # if result["text"] == "":
    #     output = "Speech was not detected"
    # else:
    #     output = result["text"]
    logger.debug("transcript is: " + str(result["text"]))

    st.text_area(label='Text & Timeline', value='\n'.join([ f"[{format_time(segment['start'])} --> {format_time(segment['end'])}]  {segment['text']}" for segment in result["segments"]]), height=300)

    st.text_area(label='Only Text', value='\n'.join([ f"{segment['text']}" for segment in result["segments"]]), height=300)
    
    # st.text_area(label="Raw Obj", value=str(result), height=300)