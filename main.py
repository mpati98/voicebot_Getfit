from flask import Flask, jsonify, render_template, request
from flask_restful import Resource, Api
import nltk
nltk.download('popular')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
# Load data
from keras.models import load_model
import json
model = load_model('data/model/modelGetfit_0623.h5')
intents = json.loads(open('data/intents/intents_Getfit_062023.json', encoding="utf8").read())
words = pickle.load(open('data/model/textsGetfit_0623.pkl','rb'))
classes = pickle.load(open('data/model/labelsGetfit_0623.pkl','rb'))

def transText(text_input, scr_input='user'):
    from googletrans import Translator
    # define a translate object
    translate = Translator()
    if scr_input == "bot":
        result = translate.translate(text_input, src='en', dest='vi')
        result = result.text
    elif scr_input == "user":
        result = translate.translate(text_input, src='vi', dest='en')
        result = result.text
    else:
        result = "We not support this language, please use English or Vietnamese!"
    return result

def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    print(res)
    # AMBIGOUS_THRESHOLD = 0.0
    CERTAIN_THRESHOLD = 0.7
    results = [[i,r] for i,r in enumerate(res) if r>CERTAIN_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    # print(results)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = i['responses']
            break
    return result, tag

def chatbot_response(msg):
    ints = predict_class(msg, model)
    print(ints)
    if ints:
        res, tag = getResponse(ints, intents)
    else:
        res = ["Rất xin lỗi vì thông tin bạn cần không tồn tại trong hệ thống, chúng tôi sẽ kiểm tra và cập nhật trong thời gian tới. Bạn còn muốn biết thêm thông tin gì khác không?", "930e5fa5-827a-454f-bcac-84e1b9dd5b4f"]
        tag = "Other"
    return res, tag
def chat_rulebased_01(msg):
    if "làm gì" in msg.lower() or "mục đích" in msg.lower():
        res = ["Tập gym chính là một hình thức tập luyện được nhiều người lựa chọn với mục đích hướng tới việc tăng cân, duy trì được vóc dáng, giúp giải tỏa căng thẳng sau một ngày làm việc. Nhờ nhu cầu đó đã thúc đẩy việc các phòng tập luyện ra đời với các hình thức tập vô cùng đa dạng như dance, yoga, fitness,…. đặc biệt cũng có những hình thức tập luyện chuyên sâu vào từng mục đích cụ thể hơn như: tăng cơ cho nam, giảm mỡ bụng, giảm cân cho nữ,… và điều này cũng tùy thuộc vào mục đích và nhu cầu của từng đối tượng cụ thể.", "b60e58c0-fecd-41cf-809e-10d132947668"]
        tag = "GetFit_gym_practice"
    elif "lợi ích" in msg.lower():
        res = ["Lợi ích của việc tập Gym mang lại là: Phát triển cơ bắp, giúp kích thích và xây dựng cơ bắp khi bạn kết hợp với lượng protein đầy đủ. Điều này cũng là do khi tập gym cơ thể giải phóng được lượng hormone giúp thúc đẩy quá trình hấp thụ các axit amin cho cơ bắp làm cho cơ bắp phát triển. Cải thiện được tâm trạng, khi tập gym tâm trạng thoải mái hơn sẽ làm giảm đi cảm giác chán nản, căng thẳng. Hơn nữa, việc tập thể dục đã được chứng minh rằng sẽ giúp làm giảm các triệu chứng về lo lắng. Hơn thế nữa, nó cũng giúp chúng ta có những nhận thức đúng đắn hơn về tinh thần và phân tâm được nỗi sợ hãi. Giúp giảm cân, một số nghiên cứu đã chỉ ra rằng việc không hoạt động cũng là yếu tố khiến cho nhiều người tăng cân, béo phì. Để hiểu rõ hơn tác dụng của tập thể dục đối với việc giảm cân thì chúng ta cần phải biết được mối quan hệ giữa tập thể dục và sự tiêu hao năng lượng. Cơ thể của chúng ta dành năng lượng theo 3 cách: tiêu hóa thức ăn, duy trì các chức năng trong cơ thể và tập thể dục. Khi chúng ta giảm cân bằng việc ăn kiêng thì lượng calo sẽ giảm và kéo theo tốc độ trao đổi chất giảm theo, điều này sẽ trì hoãn được việc giảm cân. Ngược lại, khi tập gym thường xuyên cũng làm tăng quá trình trao đổi chất, điều này sẽ làm calo được đốt cháy nhanh hơn và sẽ giúp bạn giảm cân.", "d62f62b2-fce6-4bdd-9fdb-bbd2efe5e993"]
        tag = "GetFit_gym_benefit" 
    else:
        res = ["Gym chính là một từ có nguồn gốc từ Hy Lạp cổ đại, bắt nguồn của nó chính là từ “gymnasium” nó có nghĩa liên quan đến việc tập thể dục, thể thao ở trong nhà. Ngày nay, từ Gym chính là một khái niệm để chỉ cho việc tập thể hình nói chung. Những người tập gym hay các gymer chính là ám chỉ những người tập thể hình ở các không gian rộng, thoáng với các thiết bị hỗ trợ như máy tập luyện, tạ,… hướng tới mục tiêu phát triển cơ bắp, giảm cân,…", "faf1786a-4fcd-4f86-83e7-bc9bd119e505"]
        tag = "GetFit_gym_definition" 
    return res, tag


app = Flask(__name__)
api = Api(app)



@app.route("/")
def home():
    return render_template("index.html")

@app.route('/welcome', methods=["POST"])
def voice_welcome():
    resp = "Getfit gym và Yoga xin kính chào quý khách, em là trợ lý ảo của câu lạc bộ, quý khách cần hỗ trợ gì ạ?"
    output = {
            "res_text": resp,
            "res_audio": "GetFit_welcome"
        }
    return jsonify(output)


class Chatbot(Resource):

    def post(self):
        text_input = request.get_json().get("message")
        if "ghim" in text_input.lower() or "diêm" in text_input.lower():
            resp, tag = chat_rulebased_01(text_input)
        else:
            text_input = transText(text_input)
            try:
                resp, tag = chatbot_response(text_input)
            except:
                resp = ["Tín hiệu không ổn định, vui lòng lặp lại rõ hơn nhé", "fbad6e35-3933-4388-be7b-d6dda276e114"]
                tag = "Error"
            print(resp)
        output = {
            "res_text": resp[0],
            "audio_token": resp[1],
            "res_audio": tag
        }
        return jsonify(output)

api.add_resource(Chatbot, '/response')

if __name__ == "__main__":
    app.run(debug=True)