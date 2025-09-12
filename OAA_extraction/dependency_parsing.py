import spacy
from spacy import displacy

# Load the English language model
nlp = spacy.load("en_core_web_sm")

# Define a function to perform dependency parsing
def perform_dependency_parsing(text):
    # Process the text with SpaCy
    doc = nlp(text)

    # Print the dependency parsing results
    for token in doc:
        print(f"Token: {token.text}, Head: {token.head.text}, Dependency: {token.dep_}, POS: {token.pos_}")

    # Visualize the dependency tree
    displacy.render(doc, style="dep", jupyter=True, options={"compact": True})
    return doc



if __name__ == "__main__":
    text = "a bathroom with a toilet, sink, and shower"
    a = perform_dependency_parsing(text)