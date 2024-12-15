from flask import Flask, render_template, request, jsonify
from main import process_question  # Assuming you have a function to process the question

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('./index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('question')
    response = process_question(user_input)  # Call your main function to process the question
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)