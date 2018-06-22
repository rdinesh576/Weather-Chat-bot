from flask import Flask, render_template, request, jsonify
import requests
import datetime
import nltk
from nltk.stem.lancaster import LancasterStemmer
from chatterbot.trainers import ListTrainer
from chatterbot import ChatBot
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
app = Flask(__name__)
#nltk.download('punkt')
# word stemmer
stemmer = LancasterStemmer()

# 3 classes of training data
training_data = []
training_data.append({"class":"greeting", "sentence":"how are you?"})
training_data.append({"class":"greeting", "sentence":"how is your day?"})
training_data.append({"class":"greeting", "sentence":"good day"})
training_data.append({"class":"greeting", "sentence":"Hello"})
training_data.append({"class":"greeting", "sentence":"Good Morning Evening night afternoon noon!"})
training_data.append({"class":"greeting", "sentence":"Hi"})
training_data.append({"class":"greeting", "sentence":"How are you doing?"})
training_data.append({"class":"greeting", "sentence":"How do you do?"})
training_data.append({"class":"greeting", "sentence":"how is it going today?"})
training_data.append({"class":"greeting", "sentence":"have a nice day"})
training_data.append({"class":"greeting", "sentence":"have a good day"})
training_data.append({"class":"greeting", "sentence":"see you later"})
training_data.append({"class":"greeting", "sentence":"have a nice day"})
training_data.append({"class":"greeting", "sentence":"talk to you soon"})
training_data.append({"class":"greeting", "sentence":"see you soon"})
training_data.append({"class":"greeting", "sentence":"see you"})
training_data.append({"class":"greeting", "sentence":"bye"})

training_data.append({"class":"question", "sentence":"How is the weather in Akron?"})
training_data.append({"class":"question", "sentence":"How is the weather tomorrow at parsi?"})
training_data.append({"class":"question", "sentence":"How is weather today in states?"})
training_data.append({"class":"question", "sentence":"can you give me the temperature at cali?"})
training_data.append({"class":"question", "sentence":"what is temperature at chicago?"})
training_data.append({"class":"question", "sentence":"what is weather in jersey?"})

training_data.append({"class":"question", "sentence":"Will it rain today?"})
training_data.append({"class":"question", "sentence":"Do I need to carry an umbrella tomorrow?"})
training_data.append({"class":"question", "sentence":"Will it be sunny tomorrow?"})
training_data.append({"class":"question", "sentence":"is it sunny tomorrow?"})
training_data.append({"class":"question", "sentence":"is there snow day after tomorrow?"})
training_data.append({"class":"question", "sentence":"Do i need a cap tomorrow?"})

# capture unique stemmed words in the training corpus
corpus_words = {}
class_words = {}
# turn a list into a set (of unique items) and then a list again (this removes duplicates)
classes = list(set([a['class'] for a in training_data]))
for c in classes:
    # prepare a list of words within each class
    class_words[c] = []

# loop through each sentence in our training data
for data in training_data:
    # tokenize each sentence into words
    for word in nltk.word_tokenize(data['sentence']):
        # ignore a some things
        if word not in ["?", "'s"]:
            # stem and lowercase each word
            stemmed_word = stemmer.stem(word.lower())
            # have we not seen this word already?
            if stemmed_word not in corpus_words:
                corpus_words[stemmed_word] = 1
            else:
                corpus_words[stemmed_word] += 1

            # add the word to our words in class list
            class_words[data['class']].extend([stemmed_word])


# calculate a score for a given class taking into account word commonality
def calculate_class_score(sentence, class_name, show_details=True):
    score = 0
    # tokenize each word in our new sentence
    for word in nltk.word_tokenize(sentence):
        # check to see if the stem of the word is in any of our classes
        if stemmer.stem(word.lower()) in class_words[class_name]:
            # treat each word with relative weight
            score += (1 / corpus_words[stemmer.stem(word.lower())])

            if show_details:
                print ("   match: %s (%s)" % (stemmer.stem(word.lower()), 1 / corpus_words[stemmer.stem(word.lower())]))
    return score

# calculate a score for a given class taking into account word commonality
def calculate_class_score_commonality(sentence, class_name, show_details=True):
    score = 0
    # tokenize each word in our new sentence
    for word in nltk.word_tokenize(sentence):
        # check to see if the stem of the word is in any of our classes
        if stemmer.stem(word.lower()) in class_words[class_name]:
            # treat each word with relative weight
            score += (1 / corpus_words[stemmer.stem(word.lower())])

            if show_details:
                print ("   match: %s (%s)" % (stemmer.stem(word.lower()), 1 / corpus_words[stemmer.stem(word.lower())]))
    return score

def greetings(input_sentence):
    chatbot = ChatBot("weather chatbot")
    conversation = [
        "Hello",
        "Hi there!",
        "How are you doing?",
        "I'm doing great.",
        "That is good to hear",
        "Thank you.",
        "You're welcome.",
        "bye",
        "see you",
        "Take care!"
    ]
    chatbot.set_trainer(ListTrainer)
    chatbot.train(conversation)
    response = chatbot.get_response(input_sentence)
    print(response)
    return response

# return the class with highest score for sentence
def classify(sentence):
    high_class = None
    high_score = 0
    # loop through our classes
    for c in class_words.keys():
        # calculate score of sentence for each class
        score = calculate_class_score_commonality(sentence, c, show_details=True)
        # keep track of highest score
        if score > high_score:
            high_class = c
            high_score = score
    return high_class

print ("Corpus words and counts: %s \n" % corpus_words)
# also we have all words in each class
print ("Class words: %s" % class_words)

def weather(input_sentence):
    A = {'weather', '?', 'temperature', 'tell', 'give', 'get'}
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(input_sentence)
    stop_words.update(A)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    tagged = nltk.pos_tag(filtered_sentence)
    print(tagged)
    nouns = []
    weather_condition = []
    zipcodes = []
    zipcode = ''
    for word, pos in tagged:
        if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS'):
            nouns.append(word)
    for word, pos in tagged:
        if (pos == 'JJ'):
            weather_condition.append(word)
    for word, pos in tagged:
        if (pos == 'CD'):
            zipcodes.append(word)
    if(nouns!=[]):
        city = nouns[0]
        print('you have entered the city: ' + city)
    #s = ', '.join(filtered_sentence)
    #s = s.replace(',', ' ')
    #zipcode = ''.join(re.findall(r'\d+', s))
    if zipcodes:
        return byzip(zipcodes)
    #if zipcode:
    #    print('zipcode: ' + zipcode)
    #    return byzip(zipcode)
    else:
        return bycity(city,weather_condition)

def bycity(city,weather_condition):
    r = requests.get('https://api.openweathermap.org/data/2.5/weather?q=' + city + '&appid=102dd395d4b9ccdd2d42c2292b28f67a')
    json_object = r.json()
    if(str(json_object['cod'])=="200"):
        place_k = str(json_object['name'])
        weather_list = json_object['weather']
        weather_k = weather_list[0]['main']
        weather_detail_k = weather_list[0]['description']
        sunrise_k = json_object['sys']['sunrise']
        sunset_k = json_object['sys']['sunset']
        sunrise = datetime.datetime.fromtimestamp(int(sunrise_k)).strftime('%H:%M:%S')
        sunset = datetime.datetime.fromtimestamp(int(sunset_k)).strftime('%H:%M:%S')
        temp_k = float(json_object['main']['temp'])
        temp_f = (temp_k - 273.15) * 1.8 + 32
        return_string = "Weather at " + place_k + " is " + weather_k + " and the current temperature is : " + str(
            temp_f) + "°F and Sun rises, sets at " + str(sunrise) + ", " + str(sunset) + " respectively"
        if (weather_condition == []):
            return_string = " The current temperature at " + place_k + " is : " + str(
                temp_f) + " and the weather is " + weather_k + "°F and Sun rises, sets at " + str(sunrise) + ", " + str(
                sunset) + " respectively"
        else:
            return_string = "Weather at " + place_k + " is " + weather_k + " and in detail it is " +weather_detail_k
    else:
        return_string = 'Sorry, ' + str(json_object['message'])
    return return_string

def byzip(zipcodes):
    return_string=''
    for zipcode in zipcodes:
        print('zipcode: ' + zipcode)
        r = requests.get(
            'https://api.openweathermap.org/data/2.5/weather?zip=' + zipcode + ',us&appid=102dd395d4b9ccdd2d42c2292b28f67a')
        json_object = r.json()
        if (str(json_object['cod']) == "200"):
            print(json_object)
            place_k = str(json_object['name'])
            weather_list = json_object['weather']
            weather_k = weather_list[0]['main']
            sunrise_k = json_object['sys']['sunrise']
            sunset_k = json_object['sys']['sunset']
            sunrise = datetime.datetime.fromtimestamp(int(sunrise_k)).strftime('%H:%M:%S')
            sunset = datetime.datetime.fromtimestamp(int(sunset_k)).strftime('%H:%M:%S')
            temp_k = float(json_object['main']['temp'])
            temp_f = (temp_k - 273.15) * 1.8 + 32
            return_string = return_string + "Weather at " + place_k + " is " + weather_k + " and the current temperature is : " + str(
                temp_f) + "°F and Sun rises, sets at " + str(sunrise) + ", " + str(sunset) + " respectively <br/>"
        else:
            return_string = 'Sorry, ' + str(json_object['message'])
    return return_string

@app.route('/temperature', methods=['POST'])
def temperature():
    print("call hit")
    message = str(request.form['messageText'])
    print(message)
    return jsonify({'status': 'OK', 'answer': "Please enter some valid input"})

@app.route('/alltext', methods=['POST'])
def alltext():
    user_input = str(request.form['messageText'])
    print(user_input)
    result1 = classify(user_input)
    print(result1)
    if result1 == 'greeting':
        x=''
        x = greetings(user_input)
        return_string = str(x)
    elif result1 == 'question':
        return_string = weather(user_input)
    else:
        return_string = 'I am not sure I got you. thank you'
    return jsonify({'status': 'OK', 'answer': return_string})

@app.route('/')
def index():
	return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)