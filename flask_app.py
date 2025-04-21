from flask import Flask, request, jsonify
from flask_cors import CORS
from test import FraudDetectionMonitor, transcribe_audio
import os

app = Flask(__name__)
CORS(app)

monitor = FraudDetectionMonitor(
    model_path=r'bert_classifier.pt',
    bert_path=r'bert-base-chinese',
    window_size=3,
    high_confidence_threshold=0.7,
    required_high_confidence=3
)


@app.route('/detect', methods=['POST'])
def detect_audio():
    if 'audio' not in request.files:
        print("No audio file received")
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    audio_path = f'temp_audio_{audio_file.filename}'
    print(f"Received audio: {audio_file.filename}")

    try:
        audio_file.save(audio_path)
        text = transcribe_audio(audio_path)
        print(f"Transcribed: {text}")

        if not text:
            print("Transcription failed")
            return jsonify({'error': 'Transcription failed'}), 500

        prediction, confidence, fraud_alert = monitor.process_message(text)
        print(f"Result: {prediction, confidence, fraud_alert}")

        return jsonify({
            'transcription': text,
            'prediction': '诈骗' if prediction == 1 else '正常',
            'confidence': float(confidence),
            'fraudAlert': fraud_alert
        })
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)


@app.route('/detect_text', methods=['POST'])
def detect_text():
    if not request.json or 'text' not in request.json:
        print("No text provided")
        return jsonify({'error': 'No text provided'}), 400

    text = request.json['text']
    print(f"Received text: {text[:50]}...")

    prediction, confidence, fraud_alert = monitor.process_message(text)
    print(f"Result: {prediction, confidence, fraud_alert}")

    return jsonify({
        'prediction': '诈骗' if prediction == 1 else '正常',
        'confidence': float(confidence),
        'fraudAlert': fraud_alert
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)