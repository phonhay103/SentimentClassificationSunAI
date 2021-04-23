import tensorflow as tf
import numpy as np
import re
import string
import streamlit as st
from tensorflow import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from joblib import load
from PIL import Image

VN_CHARS_LOWER = u'ạảãàáâậầấẩẫăắằặẳẵóòọõỏôộổỗồốơờớợởỡéèẻẹẽêếềệểễúùụủũưựữửừứíìịỉĩýỳỷỵỹđð'

replace_list = {
    'òa': 'oà', 'óa': 'oá', 'ỏa': 'oả', 'õa': 'oã', 'ọa': 'oạ', 'òe': 'oè', 'óe': 'oé','ỏe': 'oẻ',
    'õe': 'oẽ', 'ọe': 'oẹ', 'ùy': 'uỳ', 'úy': 'uý', 'ủy': 'uỷ', 'ũy': 'uỹ','ụy': 'uỵ', 'uả': 'ủa',
    'ả': 'ả', 'ố': 'ố', 'u´': 'ố','ỗ': 'ỗ', 'ồ': 'ồ', 'ổ': 'ổ', 'ấ': 'ấ', 'ẫ': 'ẫ', 'ẩ': 'ẩ',
    'ầ': 'ầ', 'ỏ': 'ỏ', 'ề': 'ề','ễ': 'ễ', 'ắ': 'ắ', 'ủ': 'ủ', 'ế': 'ế', 'ở': 'ở', 'ỉ': 'ỉ',
    'ẻ': 'ẻ', 'àk': u' à ','aˋ': 'à', 'iˋ': 'ì', 'ă´': 'ắ','ử': 'ử', 'e˜': 'ẽ', 'y˜': 'ỹ', 'a´': 'á',
    #Emoji: positive or negative
    "👹": "negative", "👻": "positive", "💃": "positive",'🤙': ' positive ', '👍': ' positive ',
    "💄": "positive", "💎": "positive", "💩": "positive","😕": "negative", "😱": "negative", "😸": "positive",
    "😾": "negative", "🚫": "negative",  "🤬": "negative","🧚": "positive", "🧡": "positive",'🐶':' positive ',
    '👎': ' negative ', '😣': ' negative ','✨': ' positive ', '❣': ' positive ','☀': ' positive ',
    '♥': ' positive ', '🤩': ' positive ', 'like': ' positive ', '💌': ' positive ',
    '🤣': ' positive ', '🖤': ' positive ', '🤤': ' positive ', ':(': ' negative ', '😢': ' negative ',
    '❤': ' positive ', '😍': ' positive ', '😘': ' positive ', '😪': ' negative ', '😊': ' positive ',
    '?': ' ? ', '😁': ' positive ', '💖': ' positive ', '😟': ' negative ', '😭': ' negative ',
    '💯': ' positive ', '💗': ' positive ', '♡': ' positive ', '💜': ' positive ', '🤗': ' positive ',
    '^^': ' positive ', '😨': ' negative ', '☺': ' positive ', '💋': ' positive ', '👌': ' positive ',
    '😖': ' negative ', '😀': ' positive ', ':((': ' negative ', '😡': ' negative ', '😠': ' negative ',
    '😒': ' negative ', '🙂': ' positive ', '😏': ' negative ', '😝': ' positive ', '😄': ' positive ',
    '😙': ' positive ', '😤': ' negative ', '😎': ' positive ', '😆': ' positive ', '💚': ' positive ',
    '✌': ' positive ', '💕': ' positive ', '😞': ' negative ', '😓': ' negative ', '️🆗️': ' positive ',
    '😉': ' positive ', '😂': ' positive ', ':v': '  positive ', '=))': '  positive ', '😋': ' positive ',
    '💓': ' positive ', '😐': ' negative ', ':3': ' positive ', '😫': ' negative ', '😥': ' negative ',
    '😃': ' positive ', '😬': ' 😬 ', '😌': ' 😌 ', '💛': ' positive ', '🤝': ' positive ', '🎈': ' positive ',
    '😗': ' positive ', '🤔': ' negative ', '😑': ' negative ', '🔥': ' negative ', '🙏': ' negative ',
    '🆗': ' positive ', '😻': ' positive ', '💙': ' positive ', '💟': ' positive ',
    '😚': ' positive ', '❌': ' negative ', '👏': ' positive ', ';)': ' positive ', '<3': ' positive ',
    '🌝': ' positive ',  '🌷': ' positive ', '🌸': ' positive ', '🌺': ' positive ',
    '🌼': ' positive ', '🍓': ' positive ', '🐅': ' positive ', '🐾': ' positive ', '👉': ' positive ',
    '💐': ' positive ', '💞': ' positive ', '💥': ' positive ', '💪': ' positive ',
    '💰': ' positive ',  '😇': ' positive ', '😛': ' positive ', '😜': ' positive ',
    '🙃': ' positive ', '🤑': ' positive ', '🤪': ' positive ','☹': ' negative ',  '💀': ' negative ',
    '😔': ' negative ', '😧': ' negative ', '😩': ' negative ', '😰': ' negative ', '😳': ' negative ',
    '😵': ' negative ', '😶': ' negative ', '🙁': ' negative ', 't_t':' negative ', '✔': ' positive ',
    '🙈': ' positive ',
    # Sentiment words/English words
    ':))': '  positive ', ':)': ' positive ', 'ô kêi': ' ok ', 'okie': ' ok ', ' o kê ': ' ok ',
    'okey': ' ok ', 'ôkê': ' ok ', 'oki': ' ok ', ' oke ':  ' ok ',' okay':' ok ','okê':' ok ',
    ' tks ': u' cám ơn ', 'thks': u' cám ơn ', 'thanks': u' cám ơn ', 'ths': u' cám ơn ', 'thank': u' cám ơn ',
    '⭐': 'star ', '*': 'star ', '🌟': 'star ', '🎉': u' positive ',
    'kg ': u' không ','not': u' không ', u' kg ': u' không ', '"k ': u' không ',' kh ':u' không ','kô':u' không ','hok':u' không ',' kp ': u' không phải ',u' kô ': u' không ', 
    '"ko ': u' không ', u' ko ': u' không ', u' k ': u' không ', 'khong': u' không ', u' hok ': u' không ',
    'he he': ' positive ','hehe': ' positive ','hihi': ' positive ', 'haha': ' positive ', 'hjhj': ' positive ',
    ' lol ': ' negative ',' cc ': ' negative ','cute': u' dễ thương ','huhu': ' negative ', ' vs ': u' với ', 'wa': ' quá ', 'wá': u' quá', 'j': u' gì ', '“': ' ',
    ' sz ': u' cỡ ', 'size': u' cỡ ', u' đx ': u' được ', 'dk': u' được ', 'dc': u' được ', 'đk': u' được ',
    'đc': u' được ','authentic': u' chuẩn chính hãng ',u' aut ': u' chuẩn chính hãng ', u' auth ': u' chuẩn chính hãng ', 'thick': u' positive ', 'store': u' cửa hàng ',
    'shop': u' cửa hàng ', 'sp': u' sản phẩm ', 'gud': u' tốt ','god': u' tốt ','wel done':' tốt ', 'good': u' tốt ', 'gút': u' tốt ',
    'sấu': u' xấu ','gut': u' tốt ', u' tot ': u' tốt ', u' nice ': u' tốt ', 'perfect': 'rất tốt', 'bt': u' bình thường ',
    'time': u' thời gian ', 'qá': u' quá ', u' ship ': u' giao hàng ', u' m ': u' mình ', u' mik ': u' mình ',
    'ể': 'ể', 'product': 'sản phẩm', 'quality': 'chất lượng','chat':' chất ', 'excelent': 'hoàn hảo', 'bad': 'tệ','fresh': ' tươi ','sad': ' tệ ',
    'date': u' hạn sử dụng ', 'hsd': u' hạn sử dụng ','quickly': u' nhanh ', 'quick': u' nhanh ','fast': u' nhanh ','delivery': u' giao hàng ',u' síp ': u' giao hàng ',
    'beautiful': u' đẹp tuyệt vời ', u' tl ': u' trả lời ', u' r ': u' rồi ', u' shopE ': u' cửa hàng ',u' order ': u' đặt hàng ',
    'chất lg': u' chất lượng ',u' sd ': u' sử dụng ',u' dt ': u' điện thoại ',u' nt ': u' nhắn tin ',u' tl ': u' trả lời ',u' sài ': u' xài ',u'bjo':u' bao giờ ',
    'thik': u' thích ',u' sop ': u' cửa hàng ', ' fb ': ' facebook ', ' face ': ' facebook ', ' very ': u' rất ',u'quả ng ':u' quảng  ',
    'dep': u' đẹp ',u' xau ': u' xấu ','delicious': u' ngon ', u'hàg': u' hàng ', u'qủa': u' quả ',
    'iu': u' yêu ','fake': u' giả mạo ', 'trl': 'trả lời', '><': u' positive ', 'hih':'hình',
    ' por ': u' tệ ',' poor ': u' tệ ', 'ib':u' nhắn tin ', 'rep':u' trả lời ',u'fback':' feedback ','fedback':' feedback ',
    # Star
    '6 sao': ' 5star ','6 star': ' 5star ', '5star': ' 5star ','5 sao': ' 5star ','5sao': ' 5star ',
    'starstarstarstarstar': ' 5star ', '1 sao': ' 1star ', '1sao': ' 1star ','2 sao':' 1star ','2sao':' 1star ',
    '2 starstar':' 1star ','1star': ' 1star ', '0 sao': ' 1star ', '0star': ' 1star ',}

def remove_accents(text):
    __INTAB = [ch for ch in VN_CHARS_LOWER]
    __OUTTAB = "a"*17 + "o"*17 + "e"*11 + "u"*11 + "i"*5 + "y"*5 + "d"*2
    __OUTTAB += "A"*17 + "O"*17 + "E"*11 + "U"*11 + "I"*5 + "Y"*5 + "D"*2
    __r = re.compile("|".join(__INTAB))
    __replaces_dict = dict(zip(__INTAB, __OUTTAB))
    result = __r.sub(lambda m: __replaces_dict[m.group(0)], text)
    return result

def remove_punctuations(text):
    # Convert punctuation to space
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    s = text.translate(translator)
    
    s = s.replace(u'"', u' ')
    s = s.replace('🏻','')
    s = ' '.join(s.strip().split())
    return s

def normalize_text(text):
    # convert_to_lowercase
    s = text.lower()
    
    for k, v in replace_list.items():
        s = s.replace(k, v)
    
    # remove_stretched_characters
    s = re.sub(r'([A-Z])\1+', lambda m: m.group(1).lower(), s, flags=re.IGNORECASE)
    s = remove_accents(s)
    s = remove_punctuations(s)
    return s

def normalize_data(data):
    return [normalize_text(text) for text in data]

max_length = 100
trunc_type='post'
padding_type='post'

tokenizer = load('models/tokenizer.joblib')
model = tf.keras.models.load_model('models/my_model.h5')

st.title('Sentiment Classification App')
st.write("""Sentiment classification is the automated process of identifying opinions in text and labeling them as positive, or negative, based on 
            the emotions customers express within them. Using NLP to interpret subjective data, sentiment classification can help you understand 
            how customers feel about your products, services, or brand.
""")

imageLocation = st.empty()
image_path = './images/'
# imageLocation.image(Image.open(image_path + 'default' + '.jpg'))
st.write('Please enter a review in the left sidebar and see the results below: ')

sentence = st.sidebar.text_input('Enter a review here:')
show_steps = st.sidebar.radio('Show steps:', options=[True, False], index=1)
prob = st.sidebar.radio('Returns the predicted probability:', options=[True, False], index=0)
if not prob:
    threshold = st.sidebar.slider("Threshold prediction", 0.01, 0.99, 0.5, 0.01)
    st.sidebar.text('Bad \t\t\t\t   Good')

image_select = st.sidebar.selectbox(
    "Choose the background image",
    ("Default", "Girl1", "Girl2", "Girl3", "Girl4", "Girl5", 
        "Bg1", "Bg2", "Bg3", "Bg4", "Bg5", "Meme1")
)
imageLocation.image(Image.open(image_path + image_select + '.jpg'))

if sentence:
    test_normalize = normalize_data([sentence])
    test_tokenizer = tokenizer.texts_to_sequences(test_normalize)
    test_tokenizer = pad_sequences(test_tokenizer, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    if show_steps:
        st.write('**Input: **')
        st.write(f'<p style="color:blue"><i>{sentence}</i></p>', unsafe_allow_html=True)
        st.write('**Normalize: **')
        st.write(f'<p style="color:green"><i>{test_normalize[0]}</i></p>', unsafe_allow_html=True)
        st.write('**Tokenizer: **')
        st.write(test_tokenizer)

    predict = model.predict(test_tokenizer)[0][0]

    if prob:
        st.write('**Probability predicted: **')
        st.write(f'*Good: {1-predict:.2f}*')
        st.write(f'*Bad: {predict:.2f}*')
    else:
        if predict < threshold:
            st.write('***This is a good review***')
        else:
            st.write('***This is a bad review***')
