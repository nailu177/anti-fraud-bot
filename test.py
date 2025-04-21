import whisper
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from collections import deque
import os

# 设置 ffmpeg 路径（确保 Whisper 能找到 ffmpeg）
os.environ["PATH"] += os.pathsep + r"C:\ProgramData\chocolatey\bin"


# BERT 分类器模型（你的原始代码，未修改）
class BertClassifier(nn.Module):
    def __init__(self, dropout=0.5, num_labels=2, model_path=None):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_path)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        out = self.dropout(pooled_output)
        return self.classifier(out)


# 诈骗检测监控器（你的原始代码，未修改）
class FraudDetectionMonitor:
    def __init__(self, model_path=r'D:\anti-fraud\Bert\bert_classifier.pt', bert_path="D:\\anti-fraud\\bert-base-chinese",
                 window_size=5, high_confidence_threshold=0.8, required_high_confidence=3):
        """
        Initialize the fraud detection monitor
        Args:
            model_path: Path to the saved model state dict
            bert_path: Path to BERT model files
            window_size: Number of recent messages to consider
            high_confidence_threshold: Threshold for high confidence fraud detection
            required_high_confidence: Minimum number of high confidence fraud detections needed
        """
        # Load tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.model = BertClassifier(num_labels=2, model_path=bert_path)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Settings
        self.window_size = window_size
        self.high_confidence_threshold = high_confidence_threshold
        self.required_high_confidence = required_high_confidence

        # History tracking
        self.recent_messages = deque(maxlen=window_size)
        self.recent_predictions = deque(maxlen=window_size)
        self.recent_confidences = deque(maxlen=window_size)

    def predict(self, text):
        """Predict if a given text is fraudulent"""
        # Tokenize input
        inputs = self.tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors="pt")
        mask = inputs['attention_mask'].to(self.device)
        input_ids = inputs['input_ids'].to(self.device)

        # Get prediction
        with torch.no_grad():
            output = self.model(input_ids, mask)
            probabilities = torch.softmax(output, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][prediction].item()

        return prediction, confidence

    def process_message(self, text):
        """
        Process a new message and determine if it's part of a fraud pattern
        Returns:
            tuple: (is_fraud, confidence, fraud_alert)
        """
        # Get prediction
        prediction, confidence = self.predict(text)

        # Update history
        self.recent_messages.append(text)
        self.recent_predictions.append(prediction)
        self.recent_confidences.append(confidence)

        # Check for fraud pattern
        fraud_alert = self._check_fraud_pattern()

        return prediction, confidence, fraud_alert

    def _check_fraud_pattern(self):
        """Check if recent messages match a suspicious fraud pattern"""
        # Need at least window_size messages
        if len(self.recent_predictions) < self.window_size:
            return False

        # Count fraudulent messages
        fraud_count = sum(1 for p in self.recent_predictions if p == 1)

        # Count high confidence fraud detections
        high_confidence_count = sum(1 for p, c in zip(self.recent_predictions, self.recent_confidences)
                                    if p == 1 and c > self.high_confidence_threshold)

        # Check if pattern matches criteria
        if fraud_count >= self.window_size and high_confidence_count >= self.required_high_confidence:
            return True
        return False


def transcribe_audio(audio_path, model_name="base"):
    """
    Transcribe audio file to text using Whisper
    Args:
        audio_path: Path to the audio file
        model_name: Whisper model size (e.g., 'base', 'small')
    Returns:
        str: Transcribed text
    """
    try:
        # Load Whisper model
        model = whisper.load_model(model_name)
        # Transcribe audio
        result = model.transcribe(audio_path)
        return result['text']
    except Exception as e:
        print(f"Error transcribing audio {audio_path}: {e}")
        return None


def detect_fraud(text, monitor):
    """
    Detect fraud risk in text
    Args:
        text: Input text to analyze
        monitor: FraudDetectionMonitor instance
    Returns:
        tuple: (prediction, confidence, fraud_alert)
    """
    if not text:
        return None, None, None
    return monitor.process_message(text)


def main(audio_files=None):
    """
    Main function to process audio files and detect fraud
    Args:
        audio_files: List of audio file paths, or None for interactive mode
    """
    # Initialize Whisper and FraudDetectionMonitor
    try:
        monitor = FraudDetectionMonitor(
            model_path=r'/Bert/bert_classifier.pt',
            bert_path="/bert-base-chinese",
            window_size=3,
            high_confidence_threshold=0.7,
            required_high_confidence=3
        )
    except Exception as e:
        print(f"Error initializing FraudDetectionMonitor: {e}")
        return

    # Process audio files if provided
    if audio_files:
        print("===== 音频转录与诈骗检测开始 =====")
        for audio_path in audio_files:
            print(f"\n处理音频: {audio_path}")
            # Transcribe audio
            text = transcribe_audio(audio_path, model_name="base")
            if text:
                print(f"转录文本: {text}")
                # Detect fraud
                prediction, confidence, fraud_alert = detect_fraud(text, monitor)
                result = "诈骗" if prediction == 1 else "正常"
                print(f"预测: {result} (置信度: {confidence:.4f})")
                if fraud_alert:
                    print("\n 警告: 当前通话检测到诈骗!")
                    # print(" 最近的通话中包含多条高置信度诈骗信息")
                    print(" 请提高警惕，不要泄露个人信息，不要点击可疑链接，涉及钱财一律不信!\n")
            else:
                print("转录失败，跳过诈骗检测")
            print("-" * 50)

    # Interactive mode
    print("\n===== 实时文本检测模式 =====")
    print("输入任意文本进行诈骗检测 (输入'退出'结束)")
    while True:
        user_input = input("\n请输入文本: ")
        if user_input.lower() in ["退出", "exit", "quit"]:
            break
        prediction, confidence, fraud_alert = detect_fraud(user_input, monitor)
        result = "诈骗" if prediction == 1 else "正常"
        print(f"预测: {result} (置信度: {confidence:.4f})")
        if fraud_alert:
            print("\n 警告: 当前通话检测到诈骗!")
            # print(" 最近的通话中包含多条高置信度诈骗信息")
            print(" 请提高警惕，不要泄露个人信息，不要点击可疑链接，涉及钱财一律不信!\n")


if __name__ == "__main__":
    # 示例音频文件列表
    audio_files = [
        r"D:\anti-fraud\whisper\testa.m4a",
        r"D:\anti-fraud\whisper\testb.m4a",
        r"D:\anti-fraud\whisper\testc.m4a",
        r"D:\anti-fraud\whisper\testd.m4a",
        r"D:\anti-fraud\whisper\teste.m4a"
        # 可添加更多音频文件路径
    ]
    main(audio_files)