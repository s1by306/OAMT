from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
device = torch.device('cuda')

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-xlarge-mnli",local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-xlarge-mnli",local_files_only=True).to(device)
def nli_inference(premise, hypothesis,with_logits= False):
    # Encode the premise and hypothesis
    inputs = tokenizer.encode_plus(premise, hypothesis, return_tensors='pt').to(device)

    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    if with_logits:
        probabilities = torch.softmax(logits[0], dim=0)
        return probabilities
    predicted_class = torch.argmax(logits, dim=1).item()

    # Convert the predicted class to a label
    labels = ["contradiction", "neutral", "entailment"]
    return labels[predicted_class]


if __name__ == '__main__':
    premise = "i sit in kitchen"
    hypothesis = "i stand in kitchen"

    # Get the NLI prediction
    result = nli_inference(premise, hypothesis,True)
    print(f"Prediction: {result}")