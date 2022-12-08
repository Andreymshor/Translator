import streamlit as st
from googletrans import Translator
from seq_2_seq import *


if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False
st.set_page_config(page_title="Eng2Rus Translate")


st.title('A simple English to Russian Translator App using Seq2Seq Model')
hide_default_format = """
       <style>
       footer {visibility: hidden;}
       </style>
       """
#MainMenu {visibility: hidden; }
st.markdown(hide_default_format, unsafe_allow_html=True)


text_input = str(st.text_input(
        "Enter some text 👇",
        label_visibility=st.session_state.visibility,
        disabled=st.session_state.disabled,
        placeholder='English Text',
    ))

if text_input:
    with open('input_lang.pkl', 'rb') as file:
        input_lang = pickle.load(file)
    
    with open('output_lang.pkl', 'rb') as file:
        output_lang = pickle.load(file)
    
    with open('pairs.pkl', 'rb') as file:
        pairs = pickle.load(file)
    
    hidden_size=256
    
    with st.spinner('Translating...'):

        translator = Translator(service_urls=['translate.google.com'])
        device = torch.device("cpu")
        # input_lang, output_lang, pairs = prepareData('eng', 'rus', False)

        #st.write("You entered: ", text_input)
        #print(text_input)
        result = translator.translate(text_input.lower(), src='en', dest='ru').text
        #print(result)
        encoder2 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
        encoder2.load_state_dict(torch.load('results/encoder_1000000'))

        attn_decoder2 = AttnDecoderRNN(hidden_size, output_lang.n_words).to(device)
        attn_decoder2.load_state_dict(torch.load('results/decoder_1000000'))
        decoded_words = evaluateAndShowAttention(encoder2, attn_decoder2, input_sentence=text_input.lower(), input_lang=input_lang, output_lang=output_lang)
        st.subheader(f"Using Translate:\n             {result}")
        st.subheader(f"Using Seq2Seq:\n               {decoded_words}")
        st.caption('A \'?\' means that the word was not found in Seq2Seq model vocabulary')
        st.subheader('Attention Mechanism:')
    
    st.image('attention.png', caption='Visualization of Attention Mechanism')
    st.subheader('Explaining Why Seq2Seq Model Does Not Perform Well:')
    st.write(f'In the translation of \"**{text_input}**\", the model struggled to evaluate the due to the limited vocabulary and the inability to find latent patterns between the rules of English and Russian Language. This can partially be explained by the large train error:')
    st.image('training_plot.png', caption='Training Plot for Seq2Seq Model')
    st.write("In our training, the smallest error we managed to get was 1.98. Ideally, we want a training plot that has an error less than one. This may be difficult to achieve for some language pairs as these models struggle with longer input, even when expanding the number of hidden states used (256 for this one) and utilizing attention.")
    #print(decoded_words)


